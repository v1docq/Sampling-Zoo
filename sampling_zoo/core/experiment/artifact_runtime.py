"""Runtime shell for provenance capture and artifact manifest materialization."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from enum import Enum
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform as platform_module
import subprocess
import sys
from typing import Any, Callable, Iterable, Mapping, Sequence

from .artifact_manifest import (
    AcceleratorIdentity,
    ExperimentConfigRef,
    PackageVersion,
    RunArtifactRef,
    RunArtifactRegistry,
    RunIdentity,
    build_artifact_manifest,
    dataset_refs_from_records,
    render_artifact_manifest,
)


ARTIFACT_MANIFEST_FILENAME = "artifact_manifest.json"
DEFAULT_VERSIONED_PACKAGES = (
    "sampling-zoo",
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "openml",
    "lightgbm",
    "tabpfn",
    "torch",
)


@dataclass(frozen=True)
class ArtifactDeclaration:
    role: str
    relative_pattern: str
    producer_stage: str
    optional: bool
    schema_name: str | None = None
    schema_version: int | None = None
    scoped: bool = False


DEFAULT_ARTIFACT_DECLARATIONS = (
    ArtifactDeclaration(
        role="run_metadata",
        relative_pattern="run_meta.json",
        producer_stage="write_metadata",
        optional=False,
        schema_name="run_metadata",
        schema_version=1,
    ),
    ArtifactDeclaration(
        role="raw_runs",
        relative_pattern="metrics/*_runs.jsonl",
        producer_stage="run_datasets",
        optional=False,
        schema_name="run_records",
        schema_version=1,
    ),
    ArtifactDeclaration(
        role="strategy_log",
        relative_pattern="logs/strategy_runs.jsonl",
        producer_stage="run_datasets",
        optional=True,
        schema_name="run_records",
        schema_version=1,
    ),
    ArtifactDeclaration(
        role="benchmark_report",
        relative_pattern="report.md",
        producer_stage="build_reports",
        optional=True,
    ),
    ArtifactDeclaration(
        role="analysis_report",
        relative_pattern="*.tex",
        producer_stage="build_reports",
        optional=True,
        scoped=True,
    ),
    ArtifactDeclaration(
        role="metrics_table",
        relative_pattern="metrics/*.csv",
        producer_stage="build_reports",
        optional=True,
        scoped=True,
    ),
)


def capture_run_identity(
    *,
    run_id: str,
    effective_config: Mapping[str, Any],
    repo_root: str | Path,
    project: str = "sampling-zoo",
    clock: Callable[[], datetime] | None = None,
    package_names: Sequence[str] = DEFAULT_VERSIONED_PACKAGES,
    ignored_git_paths: Sequence[str | Path] = (),
) -> RunIdentity:
    now = (clock or _utc_now)()
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    created_at = now.astimezone(timezone.utc).isoformat()
    repo_path = Path(repo_root)
    return RunIdentity(
        project=project,
        run_id=run_id,
        created_at_utc=created_at,
        git_commit=_git_output(repo_path, "rev-parse", "HEAD"),
        git_dirty=_git_dirty(repo_path, ignored_paths=ignored_git_paths),
        python_version=platform_module.python_version(),
        platform=platform_module.platform(),
        packages=_capture_package_versions(package_names),
        accelerator=_capture_accelerator_identity(),
        config=_build_config_ref(effective_config),
    )


def canonical_payload_sha256(payload: Any) -> str:
    normalized = _json_ready(payload)
    encoded = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_run_artifacts(
    run_dir: str | Path,
    declarations: Iterable[
        ArtifactDeclaration
    ] = DEFAULT_ARTIFACT_DECLARATIONS,
) -> RunArtifactRegistry:
    root = Path(run_dir)
    root_resolved = root.resolve()
    artifacts: list[RunArtifactRef] = []
    registered_paths: set[str] = set()
    for declaration in declarations:
        for path in sorted(root.glob(declaration.relative_pattern)):
            if not path.is_file():
                continue
            resolved = path.resolve()
            try:
                resolved.relative_to(root_resolved)
            except ValueError as exc:
                raise ValueError(
                    f"Artifact path escapes run directory: {path}"
                ) from exc
            relative_path = path.relative_to(root).as_posix()
            if relative_path in registered_paths:
                continue
            scope = (
                path.relative_to(root).with_suffix("").as_posix()
                if declaration.scoped
                else None
            )
            artifacts.append(
                RunArtifactRef(
                    role=declaration.role,
                    path=relative_path,
                    sha256=sha256_file(path),
                    size_bytes=path.stat().st_size,
                    producer_stage=declaration.producer_stage,
                    optional=declaration.optional,
                    schema_name=declaration.schema_name,
                    schema_version=declaration.schema_version,
                    scope=scope,
                )
            )
            registered_paths.add(relative_path)
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative_path = path.relative_to(root).as_posix()
        if relative_path in registered_paths or relative_path in {
            ARTIFACT_MANIFEST_FILENAME,
            f".{ARTIFACT_MANIFEST_FILENAME}.tmp",
        }:
            continue
        artifacts.append(
            RunArtifactRef(
                role="unclassified_artifact",
                path=relative_path,
                sha256=sha256_file(path),
                size_bytes=path.stat().st_size,
                producer_stage="experiment_runtime",
                optional=True,
                scope=relative_path,
            )
        )
    return RunArtifactRegistry.from_artifacts(artifacts)


def materialize_experiment_artifact_manifest(
    *,
    run_dir: str | Path,
    run_identity: RunIdentity,
    status: str,
    records: Sequence[Mapping[str, Any]],
    declarations: Iterable[
        ArtifactDeclaration
    ] = DEFAULT_ARTIFACT_DECLARATIONS,
) -> Path:
    root = Path(run_dir)
    registry = discover_run_artifacts(root, declarations)
    manifest = build_artifact_manifest(
        run=run_identity,
        status=status,
        record_count=len(records),
        registry=registry,
        datasets=dataset_refs_from_records(records),
    )
    target = root / ARTIFACT_MANIFEST_FILENAME
    temporary = root / f".{ARTIFACT_MANIFEST_FILENAME}.tmp"
    temporary.write_bytes(render_artifact_manifest(manifest))
    os.replace(temporary, target)
    return target


def _build_config_ref(
    effective_config: Mapping[str, Any],
) -> ExperimentConfigRef:
    suite_ids = []
    for key in ("classification_suite", "regression_suite"):
        value = effective_config.get(key)
        if value is not None:
            suite_ids.append(int(value))
    requested_tasks = []
    for key in ("classification_tasks", "regression_tasks"):
        values = effective_config.get(key)
        if values:
            if isinstance(values, str):
                requested_tasks.append(values)
            else:
                requested_tasks.extend(str(value) for value in values)
    return ExperimentConfigRef(
        sha256=canonical_payload_sha256(effective_config),
        seed=_optional_int(effective_config.get("seed")),
        max_train_rows=_optional_int(
            effective_config.get("max_train_rows")
        ),
        suite_ids=tuple(sorted(set(suite_ids))),
        requested_tasks=tuple(dict.fromkeys(requested_tasks)),
    )


def _capture_package_versions(
    package_names: Sequence[str],
) -> tuple[PackageVersion, ...]:
    packages: list[PackageVersion] = []
    for package_name in package_names:
        try:
            version = metadata.version(package_name)
        except metadata.PackageNotFoundError:
            continue
        packages.append(PackageVersion(package_name, version))
    return tuple(sorted(packages))


def _capture_accelerator_identity() -> AcceleratorIdentity:
    try:
        import torch
    except Exception:
        return AcceleratorIdentity(
            torch_version=None,
            cuda_runtime=None,
            cuda_available=False,
            device_count=0,
            device_name=None,
        )

    try:
        cuda_available = bool(torch.cuda.is_available())
        device_count = int(torch.cuda.device_count()) if cuda_available else 0
        device_name = (
            str(torch.cuda.get_device_name(0))
            if cuda_available and device_count > 0
            else None
        )
    except Exception:
        cuda_available = False
        device_count = 0
        device_name = None
    return AcceleratorIdentity(
        torch_version=str(torch.__version__),
        cuda_runtime=(
            None if torch.version.cuda is None else str(torch.version.cuda)
        ),
        cuda_available=cuda_available,
        device_count=device_count,
        device_name=device_name,
    )


def _git_output(repo_root: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    value = completed.stdout.strip()
    return value or None


def _git_dirty(
    repo_root: Path,
    *,
    ignored_paths: Sequence[str | Path] = (),
) -> bool | None:
    ignored_relative_paths = _relative_git_paths(
        repo_root,
        ignored_paths,
    )
    try:
        completed = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    changed_paths = (
        _porcelain_entry_path(line)
        for line in completed.stdout.splitlines()
        if line.strip()
    )
    return any(
        not _is_ignored_git_path(path, ignored_relative_paths)
        for path in changed_paths
    )


def _relative_git_paths(
    repo_root: Path,
    paths: Sequence[str | Path],
) -> tuple[str, ...]:
    relative_paths: list[str] = []
    root = repo_root.resolve()
    for path in paths:
        resolved = Path(path).resolve()
        try:
            relative = resolved.relative_to(root).as_posix()
        except ValueError:
            continue
        relative_paths.append(relative.rstrip("/"))
    return tuple(sorted(set(relative_paths)))


def _porcelain_entry_path(line: str) -> str:
    path = line[3:].strip().strip('"')
    if " -> " in path:
        path = path.rsplit(" -> ", 1)[1].strip('"')
    return path.replace("\\", "/")


def _is_ignored_git_path(
    path: str,
    ignored_paths: Sequence[str],
) -> bool:
    return any(
        path == ignored or path.startswith(f"{ignored}/")
        for ignored in ignored_paths
    )


def _json_ready(value: Any) -> Any:
    if is_dataclass(value):
        return _json_ready(asdict(value))
    if isinstance(value, Mapping):
        return {
            str(key): _json_ready(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, set):
        normalized = [_json_ready(item) for item in value]
        return sorted(
            normalized,
            key=lambda item: json.dumps(
                item,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)
