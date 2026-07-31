"""Shared effect shell utilities for synthetic benchmark runners."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from benchmark_incremental import IncrementalExperimentSaver
from benchmark_logging import BenchmarkLogger
from sampling_zoo.core.experiment.artifact_runtime import (
    capture_run_identity,
    materialize_experiment_artifact_manifest,
)
from sampling_zoo.core.experiment.errors import (
    UnavailableExperimentDependencyError,
)


class SyntheticArtifactBuilder(Protocol):
    def build(
        self,
        records: Sequence[Mapping[str, Any]],
        output_root: str | Path,
    ) -> Mapping[str, Path]:
        ...


def create_synthetic_incremental_saver(
    *,
    logger: BenchmarkLogger,
    config: Any,
    grid_size: int,
    experiment_name: str,
    records_filename: str,
    artifact_builder: SyntheticArtifactBuilder,
    snapshot_every: int,
    repo_root: str | Path,
) -> IncrementalExperimentSaver:
    """Attach durable records, snapshots, metadata, and manifest lifecycle."""

    effective_config = config_payload(config)
    run_identity = capture_run_identity(
        run_id=logger.run_id,
        effective_config=effective_config,
        repo_root=repo_root,
        ignored_git_paths=(logger.paths.root,),
    )

    def build_artifacts(records: Sequence[Mapping[str, Any]]) -> None:
        artifact_builder.build(records, logger.paths.root)

    def build_metadata(
        records: Sequence[Mapping[str, Any]],
        status: str,
    ) -> Mapping[str, Any]:
        statuses = [str(record.get("status", "unknown")) for record in records]
        return {
            "run_id": logger.run_id,
            "experiment": experiment_name,
            "status": status,
            "grid_size": int(grid_size),
            "completed_leaf_runs": statuses.count("completed"),
            "failed_leaf_runs": statuses.count("failed"),
            "skipped_leaf_runs": statuses.count("skipped"),
            "effective_config": effective_config,
            "run_identity": run_identity.to_dict(),
        }

    saver = IncrementalExperimentSaver(
        records_path=logger.paths.metrics / records_filename,
        metadata_path=logger.paths.root / "run_meta.json",
        snapshot_hooks=(build_artifacts,),
        metadata_builder=build_metadata,
        json_ready=json_ready,
        rebuild_every=snapshot_every,
    )

    def materialize_manifest(
        records: Sequence[Mapping[str, Any]],
        status: str,
    ) -> None:
        materialize_experiment_artifact_manifest(
            run_dir=logger.paths.root,
            run_identity=run_identity,
            status=status,
            records=records,
        )

    saver.add_lifecycle_hook(materialize_manifest)
    return saver


def resolve_backend_device(
    *,
    backend: str,
    requested_device: str,
    error_scope: str,
) -> str:
    if backend == "numpy":
        return "cpu"
    try:
        import torch
    except Exception as exc:
        raise UnavailableExperimentDependencyError(
            scope=error_scope,
            code="torch_backend_unavailable",
            message="Torch backend is unavailable for this leaf run",
            details={"backend": backend},
        ) from exc
    requested = str(requested_device)
    if requested != "auto":
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise UnavailableExperimentDependencyError(
                scope=error_scope,
                code="torch_device_unavailable",
                message=f"Requested torch device is unavailable: {requested}",
                details={"backend": backend, "device": requested},
            )
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def config_payload(config: Any) -> dict[str, Any]:
    if is_dataclass(config):
        return json_ready(asdict(config))
    if isinstance(config, Mapping):
        return json_ready(config)
    raise TypeError("Synthetic benchmark config must be a dataclass or mapping")


def json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value
