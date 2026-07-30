from __future__ import annotations

import json

import pytest

from examples.benchmark.benchmark_incremental import IncrementalExperimentSaver
from examples.benchmark.benchmark_logging import BenchmarkLogger
import examples.benchmark.rmt_regression_medium_datasets as rmt_runner
import sampling_zoo.core.experiment.artifact_runtime as artifact_runtime
from sampling_zoo.core.experiment.artifact_manifest import (
    AcceleratorIdentity,
    ArtifactManifestBuildError,
    ArtifactManifestParseFailure,
    ExperimentConfigRef,
    PackageVersion,
    RunArtifactRef,
    RunArtifactRegistry,
    RunIdentity,
    RunStatus,
    build_artifact_manifest,
    parse_artifact_manifest,
    render_artifact_manifest,
)
from sampling_zoo.core.experiment.artifact_runtime import (
    canonical_payload_sha256,
    materialize_experiment_artifact_manifest,
    sha256_file,
)


_EMPTY_SHA256 = "0" * 64


def _run_identity() -> RunIdentity:
    return RunIdentity(
        project="sampling-zoo",
        run_id="run_test",
        created_at_utc="2026-07-30T10:00:00+00:00",
        git_commit="a" * 40,
        git_dirty=False,
        python_version="3.11.9",
        platform="test-platform",
        packages=(
            PackageVersion(name="numpy", version="2.0.0"),
            PackageVersion(name="torch", version="2.3.1+cu121"),
        ),
        accelerator=AcceleratorIdentity(
            torch_version="2.3.1+cu121",
            cuda_runtime="12.1",
            cuda_available=True,
            device_count=1,
            device_name="NVIDIA Test GPU",
        ),
        config=ExperimentConfigRef(
            sha256=_EMPTY_SHA256,
            seed=42,
            max_train_rows=1000,
            suite_ids=(269,),
            requested_tasks=("diamonds",),
        ),
    )


def _artifact(
    *,
    role: str,
    path: str,
    optional: bool = False,
) -> RunArtifactRef:
    return RunArtifactRef(
        role=role,
        path=path,
        sha256=_EMPTY_SHA256,
        size_bytes=10,
        producer_stage="write_metadata",
        optional=optional,
    )


def test_artifact_manifest_round_trip_is_canonical() -> None:
    registry = RunArtifactRegistry.from_artifacts(
        (
            _artifact(role="raw_runs", path="metrics/runs.jsonl"),
            _artifact(role="run_metadata", path="run_meta.json"),
        )
    )
    manifest = build_artifact_manifest(
        run=_run_identity(),
        status=RunStatus.COMPLETED,
        record_count=2,
        registry=registry,
    )

    rendered = render_artifact_manifest(manifest)
    parsed = parse_artifact_manifest(json.loads(rendered))

    assert parsed == manifest
    assert render_artifact_manifest(parsed) == rendered


def test_artifact_registry_rejects_duplicate_role_scope() -> None:
    with pytest.raises(ArtifactManifestBuildError) as exc_info:
        RunArtifactRegistry.from_artifacts(
            (
                _artifact(role="run_metadata", path="run_meta.json"),
                _artifact(role="run_metadata", path="metadata.json"),
            )
        )

    assert exc_info.value.violations[0].code == "duplicate_artifact_role"


def test_canonical_config_hash_is_order_independent() -> None:
    first = {
        "models": ("lightgbm",),
        "seed": 42,
        "budgets": [0.1, 0.3],
    }
    second = {
        "budgets": [0.1, 0.3],
        "seed": 42,
        "models": ("lightgbm",),
    }

    assert canonical_payload_sha256(first) == canonical_payload_sha256(second)
    assert canonical_payload_sha256({"tasks": {"b", "a"}}) == (
        canonical_payload_sha256({"tasks": {"a", "b"}})
    )


def test_parser_returns_structured_failure_for_invalid_boolean() -> None:
    registry = RunArtifactRegistry.from_artifacts(
        (_artifact(role="run_metadata", path="run_meta.json"),)
    )
    manifest = build_artifact_manifest(
        run=_run_identity(),
        status="running",
        record_count=0,
        registry=registry,
    ).to_dict()
    manifest["run"]["git_dirty"] = "false"

    parsed = parse_artifact_manifest(manifest)

    assert isinstance(parsed, ArtifactManifestParseFailure)
    assert parsed.violations[0].code == "invalid_run_identity"


def test_git_dirty_ignores_only_current_run_directory(
    tmp_path,
    monkeypatch,
) -> None:
    class _Completed:
        returncode = 0
        stdout = "?? results/current/manifest.json\n"

    monkeypatch.setattr(
        artifact_runtime.subprocess,
        "run",
        lambda *args, **kwargs: _Completed(),
    )

    assert artifact_runtime._git_dirty(
        tmp_path,
        ignored_paths=(tmp_path / "results" / "current",),
    ) is False

    _Completed.stdout += " M sampling_zoo/core/experiment/contracts.py\n"
    assert artifact_runtime._git_dirty(
        tmp_path,
        ignored_paths=(tmp_path / "results" / "current",),
    ) is True


def test_materialized_manifest_tracks_hashes_and_openml_provenance(
    tmp_path,
) -> None:
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    (tmp_path / "run_meta.json").write_text(
        '{"status": "completed"}\n',
        encoding="utf-8",
    )
    raw_runs_path = metrics_dir / "rmt_regression_runs.jsonl"
    raw_runs_path.write_text('{"dataset": "diamonds"}\n', encoding="utf-8")
    records = [
        {
            "dataset": "diamonds",
            "extra": {
                "problem_type": "regression",
                "suite_id": 269,
                "task_id": 233211,
                "task_name": "diamonds",
                "dataset_id": 42225,
                "split_label": "holdout",
                "openml_repeat": 0,
                "openml_fold": 0,
                "openml_sample": 0,
            },
        }
    ]

    manifest_path = materialize_experiment_artifact_manifest(
        run_dir=tmp_path,
        run_identity=_run_identity(),
        status="completed",
        records=records,
    )
    first_bytes = manifest_path.read_bytes()
    payload = json.loads(first_bytes)

    assert payload["schema_version"] == 1
    assert payload["status"] == "completed"
    assert payload["record_count"] == 1
    assert payload["datasets"] == [
        {
            "dataset_id": 42225,
            "name": "diamonds",
            "openml_fold": 0,
            "openml_repeat": 0,
            "openml_sample": 0,
            "problem_type": "regression",
            "split_labels": ["holdout"],
            "suite_id": 269,
            "task_id": 233211,
            "task_name": "diamonds",
        }
    ]
    artifacts = {item["role"]: item for item in payload["artifacts"]}
    assert artifacts["raw_runs"]["sha256"] == sha256_file(raw_runs_path)
    assert artifacts["run_metadata"]["path"] == "run_meta.json"

    materialize_experiment_artifact_manifest(
        run_dir=tmp_path,
        run_identity=_run_identity(),
        status="completed",
        records=records,
    )
    assert manifest_path.read_bytes() == first_bytes


def test_rmt_orchestrator_attaches_manifest_to_incremental_lifecycle(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        rmt_runner,
        "capture_run_identity",
        lambda **_: _run_identity(),
    )
    orchestrator = rmt_runner.RMTRegressionExperimentOrchestrator(
        rmt_runner.RMTRegressionExperimentConfig(
            show_progress=False,
            synthetic_smoke=True,
        )
    )
    orchestrator._build_experiment_plan()
    logger = BenchmarkLogger(run_id="run_test", artifacts_root=tmp_path)
    saver = IncrementalExperimentSaver(
        records_path=logger.paths.metrics / "rmt_regression_runs.jsonl",
        metadata_path=logger.paths.root / "run_meta.json",
        metadata_builder=lambda records, status: {
            "run_id": logger.run_id,
            "status": status,
        },
    )

    orchestrator._configure_artifact_tracking(logger, saver)
    saver.start()
    start_manifest = json.loads(
        (logger.paths.root / "artifact_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert start_manifest["status"] == "running"
    assert start_manifest["record_count"] == 0

    saver.record({"dataset": "synthetic", "extra": {"problem_type": "regression"}})
    saver.finalize()

    metadata = json.loads(
        (logger.paths.root / "run_meta.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (logger.paths.root / "artifact_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert metadata["run_identity"]["run_id"] == "run_test"
    assert metadata["artifact_manifest_schema_version"] == 1
    assert manifest["status"] == "completed"
    assert manifest["record_count"] == 1
