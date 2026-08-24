"""Typed contracts and pure transforms for experiment artifact manifests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import json
import math
from pathlib import PurePosixPath, PureWindowsPath
import re
from typing import Any, Iterable, Mapping, Sequence


ARTIFACT_MANIFEST_SCHEMA_VERSION = 1
_IDENTIFIER_PATTERN = re.compile(r"^[a-z][a-z0-9_.-]*$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class RunStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True, order=True)
class PackageVersion:
    name: str
    version: str

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "version": self.version}


@dataclass(frozen=True)
class AcceleratorIdentity:
    torch_version: str | None
    cuda_runtime: str | None
    cuda_available: bool
    device_count: int
    device_name: str | None

    def __post_init__(self) -> None:
        if self.device_count < 0:
            raise ValueError("accelerator.device_count must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "torch_version": self.torch_version,
            "cuda_runtime": self.cuda_runtime,
            "cuda_available": self.cuda_available,
            "device_count": self.device_count,
            "device_name": self.device_name,
        }


@dataclass(frozen=True)
class ExperimentConfigRef:
    sha256: str
    seed: int | None = None
    max_train_rows: int | None = None
    suite_ids: tuple[int, ...] = ()
    requested_tasks: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validate_sha256(self.sha256, field_name="config.sha256")
        if self.max_train_rows is not None and self.max_train_rows < 1:
            raise ValueError("config.max_train_rows must be positive when provided")

    def to_dict(self) -> dict[str, Any]:
        return {
            "sha256": self.sha256,
            "seed": self.seed,
            "max_train_rows": self.max_train_rows,
            "suite_ids": list(self.suite_ids),
            "requested_tasks": list(self.requested_tasks),
        }


@dataclass(frozen=True)
class RunIdentity:
    project: str
    run_id: str
    created_at_utc: str
    git_commit: str | None
    git_dirty: bool | None
    python_version: str
    platform: str
    packages: tuple[PackageVersion, ...]
    accelerator: AcceleratorIdentity
    config: ExperimentConfigRef

    def __post_init__(self) -> None:
        if not self.project.strip():
            raise ValueError("project must be non-empty")
        if not self.run_id.strip():
            raise ValueError("run_id must be non-empty")
        if not self.created_at_utc.strip():
            raise ValueError("created_at_utc must be non-empty")
        try:
            created_at = datetime.fromisoformat(
                self.created_at_utc.replace("Z", "+00:00")
            )
        except ValueError as exc:
            raise ValueError(
                "created_at_utc must be an ISO-8601 timestamp"
            ) from exc
        if created_at.tzinfo is None:
            raise ValueError("created_at_utc must include a timezone")
        if not self.python_version.strip():
            raise ValueError("python_version must be non-empty")
        if not self.platform.strip():
            raise ValueError("platform must be non-empty")
        package_names = [package.name for package in self.packages]
        if len(package_names) != len(set(package_names)):
            raise ValueError("package names must be unique")

    def to_dict(self) -> dict[str, Any]:
        return {
            "project": self.project,
            "run_id": self.run_id,
            "created_at_utc": self.created_at_utc,
            "git_commit": self.git_commit,
            "git_dirty": self.git_dirty,
            "python_version": self.python_version,
            "platform": self.platform,
            "packages": [package.to_dict() for package in self.packages],
            "accelerator": self.accelerator.to_dict(),
            "config": self.config.to_dict(),
        }


@dataclass(frozen=True)
class DatasetRunRef:
    name: str
    problem_type: str | None = None
    suite_id: int | None = None
    task_id: int | None = None
    task_name: str | None = None
    dataset_id: int | None = None
    split_labels: tuple[str, ...] = ()
    openml_repeat: int | None = None
    openml_fold: int | None = None
    openml_sample: int | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("dataset.name must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in {
                "name": self.name,
                "problem_type": self.problem_type,
                "suite_id": self.suite_id,
                "task_id": self.task_id,
                "task_name": self.task_name,
                "dataset_id": self.dataset_id,
                "split_labels": list(self.split_labels),
                "openml_repeat": self.openml_repeat,
                "openml_fold": self.openml_fold,
                "openml_sample": self.openml_sample,
            }.items()
            if value is not None
        }


@dataclass(frozen=True)
class ArtifactManifestViolation:
    code: str
    path: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "path": self.path, "message": self.message}


@dataclass(frozen=True)
class ArtifactManifestParseFailure:
    violations: tuple[ArtifactManifestViolation, ...]


class ArtifactManifestBuildError(ValueError):
    def __init__(self, violations: Iterable[ArtifactManifestViolation]) -> None:
        self.violations = tuple(violations)
        super().__init__(
            "; ".join(
                f"{violation.path}: {violation.message}"
                for violation in self.violations
            )
        )


@dataclass(frozen=True)
class RunArtifactRef:
    role: str
    path: str
    sha256: str
    size_bytes: int
    producer_stage: str
    optional: bool
    schema_name: str | None = None
    schema_version: int | None = None
    scope: str | None = None

    def __post_init__(self) -> None:
        _validate_identifier(self.role, field_name="role")
        _validate_identifier(self.producer_stage, field_name="producer_stage")
        _validate_relative_path(self.path, field_name="path")
        _validate_sha256(self.sha256, field_name="sha256")
        if self.size_bytes < 0:
            raise ValueError("size_bytes must be non-negative")
        if (self.schema_name is None) != (self.schema_version is None):
            raise ValueError(
                "schema_name and schema_version must be provided together"
            )
        if self.schema_name is not None:
            _validate_identifier(self.schema_name, field_name="schema_name")
            if self.schema_version is None or self.schema_version < 1:
                raise ValueError("schema_version must be positive")
        if self.scope is not None:
            _validate_relative_path(self.scope, field_name="scope")

    @property
    def role_key(self) -> tuple[str, str | None]:
        return self.role, self.scope

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "role": self.role,
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "producer_stage": self.producer_stage,
            "optional": self.optional,
        }
        if self.schema_name is not None:
            payload["schema_name"] = self.schema_name
            payload["schema_version"] = self.schema_version
        if self.scope is not None:
            payload["scope"] = self.scope
        return payload


@dataclass(frozen=True)
class RunArtifactRegistry:
    artifacts: tuple[RunArtifactRef, ...] = ()

    @classmethod
    def from_artifacts(
        cls,
        artifacts: Iterable[RunArtifactRef],
    ) -> "RunArtifactRegistry":
        ordered = tuple(
            sorted(
                artifacts,
                key=lambda item: (
                    item.role,
                    item.scope or "",
                    item.path,
                ),
            )
        )
        violations: list[ArtifactManifestViolation] = []
        path_counts: dict[str, int] = {}
        role_counts: dict[tuple[str, str | None], int] = {}
        for artifact in ordered:
            path_counts[artifact.path] = path_counts.get(artifact.path, 0) + 1
            role_counts[artifact.role_key] = (
                role_counts.get(artifact.role_key, 0) + 1
            )
        for path, count in sorted(path_counts.items()):
            if count > 1:
                violations.append(
                    ArtifactManifestViolation(
                        code="duplicate_artifact_path",
                        path="artifacts",
                        message=f"Artifact path {path!r} occurs {count} times",
                    )
                )
        for (role, scope), count in sorted(
            role_counts.items(),
            key=lambda item: (item[0][0], item[0][1] or ""),
        ):
            if count > 1:
                violations.append(
                    ArtifactManifestViolation(
                        code="duplicate_artifact_role",
                        path="artifacts",
                        message=(
                            f"Artifact role {role!r} in scope {scope!r} "
                            f"occurs {count} times"
                        ),
                    )
                )
        if violations:
            raise ArtifactManifestBuildError(violations)
        return cls(artifacts=ordered)

    def required_role_violations(
        self,
        required_roles: Iterable[str],
    ) -> tuple[ArtifactManifestViolation, ...]:
        violations: list[ArtifactManifestViolation] = []
        for role in required_roles:
            matches = [
                artifact
                for artifact in self.artifacts
                if artifact.role == role
                and artifact.scope is None
                and not artifact.optional
            ]
            if len(matches) != 1:
                violations.append(
                    ArtifactManifestViolation(
                        code="required_artifact_role_count",
                        path="artifacts",
                        message=(
                            f"Required root role {role!r} must occur exactly "
                            f"once; found {len(matches)}"
                        ),
                    )
                )
        return tuple(violations)


@dataclass(frozen=True)
class ExperimentArtifactManifest:
    run: RunIdentity
    status: RunStatus
    record_count: int
    registry: RunArtifactRegistry
    datasets: tuple[DatasetRunRef, ...] = ()
    schema_version: int = ARTIFACT_MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ARTIFACT_MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported artifact manifest schema version: "
                f"{self.schema_version}"
            )
        if self.record_count < 0:
            raise ValueError("record_count must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run": self.run.to_dict(),
            "status": self.status.value,
            "record_count": self.record_count,
            "datasets": [dataset.to_dict() for dataset in self.datasets],
            "artifacts": [
                artifact.to_dict() for artifact in self.registry.artifacts
            ],
        }


def build_artifact_manifest(
    *,
    run: RunIdentity,
    status: str | RunStatus,
    record_count: int,
    registry: RunArtifactRegistry,
    datasets: Sequence[DatasetRunRef] = (),
) -> ExperimentArtifactManifest:
    normalized_status = (
        status if isinstance(status, RunStatus) else RunStatus(str(status))
    )
    required_roles = ["run_metadata"]
    if record_count > 0:
        required_roles.append("raw_runs")
    violations = registry.required_role_violations(required_roles)
    if violations:
        raise ArtifactManifestBuildError(violations)
    return ExperimentArtifactManifest(
        run=run,
        status=normalized_status,
        record_count=int(record_count),
        registry=registry,
        datasets=tuple(sorted(datasets, key=_dataset_sort_key)),
    )


def render_artifact_manifest(manifest: ExperimentArtifactManifest) -> bytes:
    text = json.dumps(
        manifest.to_dict(),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    return f"{text}\n".encode("utf-8")


def parse_artifact_manifest(
    payload: Any,
) -> ExperimentArtifactManifest | ArtifactManifestParseFailure:
    violations: list[ArtifactManifestViolation] = []
    if not isinstance(payload, Mapping):
        return ArtifactManifestParseFailure(
            (
                ArtifactManifestViolation(
                    code="manifest_not_object",
                    path="artifact_manifest",
                    message="Manifest must be an object",
                ),
            )
        )
    if payload.get("schema_version") != ARTIFACT_MANIFEST_SCHEMA_VERSION:
        violations.append(
            ArtifactManifestViolation(
                code="unsupported_schema_version",
                path="schema_version",
                message=(
                    "Expected schema version "
                    f"{ARTIFACT_MANIFEST_SCHEMA_VERSION}"
                ),
            )
        )

    run = _parse_run_identity(payload.get("run"), violations)
    status = _parse_status(payload.get("status"), violations)
    record_count = _parse_non_negative_int(
        payload.get("record_count"),
        path="record_count",
        violations=violations,
    )
    datasets = _parse_datasets(payload.get("datasets"), violations)
    registry = _parse_registry(payload.get("artifacts"), violations)

    if violations or run is None or status is None or record_count is None:
        return ArtifactManifestParseFailure(tuple(violations))
    try:
        return build_artifact_manifest(
            run=run,
            status=status,
            record_count=record_count,
            registry=registry,
            datasets=datasets,
        )
    except (ArtifactManifestBuildError, ValueError) as exc:
        nested = (
            exc.violations
            if isinstance(exc, ArtifactManifestBuildError)
            else (
                ArtifactManifestViolation(
                    code="invalid_manifest",
                    path="artifact_manifest",
                    message=str(exc),
                ),
            )
        )
        return ArtifactManifestParseFailure(tuple(nested))


def dataset_refs_from_records(
    records: Sequence[Mapping[str, Any]],
) -> tuple[DatasetRunRef, ...]:
    grouped: dict[tuple[str, int | None], dict[str, Any]] = {}
    for record in records:
        extra = record.get("extra", {})
        if not isinstance(extra, Mapping):
            extra = {}
        name = str(record.get("dataset", "dataset"))
        task_id = _optional_int(extra.get("task_id"))
        key = (name, task_id)
        current = grouped.setdefault(
            key,
            {
                "name": name,
                "problem_type": _optional_str(extra.get("problem_type")),
                "suite_id": _optional_int(extra.get("suite_id")),
                "task_id": task_id,
                "task_name": _optional_str(extra.get("task_name")),
                "dataset_id": _optional_int(extra.get("dataset_id")),
                "split_labels": set(),
                "openml_repeat": _optional_int(extra.get("openml_repeat")),
                "openml_fold": _optional_int(extra.get("openml_fold")),
                "openml_sample": _optional_int(extra.get("openml_sample")),
            },
        )
        split_label = _optional_str(extra.get("split_label"))
        if split_label is not None:
            current["split_labels"].add(split_label)

    return tuple(
        sorted(
            (
                DatasetRunRef(
                    name=value["name"],
                    problem_type=value["problem_type"],
                    suite_id=value["suite_id"],
                    task_id=value["task_id"],
                    task_name=value["task_name"],
                    dataset_id=value["dataset_id"],
                    split_labels=tuple(sorted(value["split_labels"])),
                    openml_repeat=value["openml_repeat"],
                    openml_fold=value["openml_fold"],
                    openml_sample=value["openml_sample"],
                )
                for value in grouped.values()
            ),
            key=_dataset_sort_key,
        )
    )


def _dataset_sort_key(dataset: DatasetRunRef) -> tuple[Any, ...]:
    return (
        dataset.name,
        -1 if dataset.suite_id is None else dataset.suite_id,
        -1 if dataset.task_id is None else dataset.task_id,
        dataset.task_name or "",
        -1 if dataset.dataset_id is None else dataset.dataset_id,
    )


def _validate_identifier(value: str, *, field_name: str) -> None:
    if not _IDENTIFIER_PATTERN.fullmatch(value):
        raise ValueError(
            f"{field_name} must match {_IDENTIFIER_PATTERN.pattern!r}"
        )


def _validate_sha256(value: str, *, field_name: str) -> None:
    if not _SHA256_PATTERN.fullmatch(value):
        raise ValueError(
            f"{field_name} must be a lowercase 64-character SHA-256 digest"
        )


def _validate_relative_path(value: str, *, field_name: str) -> None:
    if not value or "\\" in value:
        raise ValueError(f"{field_name} must be a non-empty POSIX path")
    posix_path = PurePosixPath(value)
    windows_path = PureWindowsPath(value)
    if posix_path.is_absolute() or windows_path.is_absolute():
        raise ValueError(f"{field_name} must be relative")
    if any(part in {"", ".", ".."} for part in posix_path.parts):
        raise ValueError(f"{field_name} must be canonical and contained")
    if posix_path.as_posix() != value:
        raise ValueError(f"{field_name} must be canonical")


def _parse_run_identity(
    payload: Any,
    violations: list[ArtifactManifestViolation],
) -> RunIdentity | None:
    if not isinstance(payload, Mapping):
        violations.append(
            ArtifactManifestViolation(
                code="run_not_object",
                path="run",
                message="run must be an object",
            )
        )
        return None
    try:
        package_payload = payload.get("packages", [])
        if not isinstance(package_payload, Sequence) or isinstance(
            package_payload, (str, bytes)
        ):
            raise ValueError("packages must be an array")
        if any(not isinstance(item, Mapping) for item in package_payload):
            raise ValueError("every package must be an object")
        packages = tuple(
            sorted(
                PackageVersion(
                    name=str(item["name"]),
                    version=str(item["version"]),
                )
                for item in package_payload
            )
        )
        accelerator_payload = payload.get("accelerator")
        config_payload = payload.get("config")
        if not isinstance(accelerator_payload, Mapping):
            raise ValueError("accelerator must be an object")
        if not isinstance(config_payload, Mapping):
            raise ValueError("config must be an object")
        cuda_available = accelerator_payload.get("cuda_available")
        if not isinstance(cuda_available, bool):
            raise ValueError("accelerator.cuda_available must be a boolean")
        git_dirty = payload.get("git_dirty")
        if git_dirty is not None and not isinstance(git_dirty, bool):
            raise ValueError("git_dirty must be a boolean or null")
        accelerator = AcceleratorIdentity(
            torch_version=_optional_str(
                accelerator_payload.get("torch_version")
            ),
            cuda_runtime=_optional_str(
                accelerator_payload.get("cuda_runtime")
            ),
            cuda_available=cuda_available,
            device_count=int(accelerator_payload.get("device_count", 0)),
            device_name=_optional_str(
                accelerator_payload.get("device_name")
            ),
        )
        config = ExperimentConfigRef(
            sha256=str(config_payload.get("sha256", "")),
            seed=_optional_int(config_payload.get("seed")),
            max_train_rows=_optional_int(
                config_payload.get("max_train_rows")
            ),
            suite_ids=tuple(
                int(value) for value in config_payload.get("suite_ids", [])
            ),
            requested_tasks=tuple(
                str(value)
                for value in config_payload.get("requested_tasks", [])
            ),
        )
        return RunIdentity(
            project=str(payload.get("project", "")),
            run_id=str(payload.get("run_id", "")),
            created_at_utc=str(payload.get("created_at_utc", "")),
            git_commit=_optional_str(payload.get("git_commit")),
            git_dirty=git_dirty,
            python_version=str(payload.get("python_version", "")),
            platform=str(payload.get("platform", "")),
            packages=packages,
            accelerator=accelerator,
            config=config,
        )
    except (KeyError, TypeError, ValueError) as exc:
        violations.append(
            ArtifactManifestViolation(
                code="invalid_run_identity",
                path="run",
                message=str(exc),
            )
        )
        return None


def _parse_status(
    value: Any,
    violations: list[ArtifactManifestViolation],
) -> RunStatus | None:
    try:
        return RunStatus(str(value))
    except ValueError:
        violations.append(
            ArtifactManifestViolation(
                code="invalid_status",
                path="status",
                message=f"Unsupported run status: {value!r}",
            )
        )
        return None


def _parse_non_negative_int(
    value: Any,
    *,
    path: str,
    violations: list[ArtifactManifestViolation],
) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = -1
    if parsed < 0:
        violations.append(
            ArtifactManifestViolation(
                code="invalid_non_negative_integer",
                path=path,
                message="Value must be a non-negative integer",
            )
        )
        return None
    return parsed


def _parse_datasets(
    payload: Any,
    violations: list[ArtifactManifestViolation],
) -> tuple[DatasetRunRef, ...]:
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
        violations.append(
            ArtifactManifestViolation(
                code="datasets_not_array",
                path="datasets",
                message="datasets must be an array",
            )
        )
        return ()
    datasets: list[DatasetRunRef] = []
    for index, item in enumerate(payload):
        if not isinstance(item, Mapping):
            violations.append(
                ArtifactManifestViolation(
                    code="dataset_not_object",
                    path=f"datasets.{index}",
                    message="Dataset entry must be an object",
                )
            )
            continue
        try:
            datasets.append(
                DatasetRunRef(
                    name=str(item.get("name", "")),
                    problem_type=_optional_str(item.get("problem_type")),
                    suite_id=_optional_int(item.get("suite_id")),
                    task_id=_optional_int(item.get("task_id")),
                    task_name=_optional_str(item.get("task_name")),
                    dataset_id=_optional_int(item.get("dataset_id")),
                    split_labels=tuple(
                        str(value)
                        for value in item.get("split_labels", [])
                    ),
                    openml_repeat=_optional_int(
                        item.get("openml_repeat")
                    ),
                    openml_fold=_optional_int(item.get("openml_fold")),
                    openml_sample=_optional_int(
                        item.get("openml_sample")
                    ),
                )
            )
        except (TypeError, ValueError) as exc:
            violations.append(
                ArtifactManifestViolation(
                    code="invalid_dataset",
                    path=f"datasets.{index}",
                    message=str(exc),
                )
            )
    return tuple(datasets)


def _parse_registry(
    payload: Any,
    violations: list[ArtifactManifestViolation],
) -> RunArtifactRegistry:
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
        violations.append(
            ArtifactManifestViolation(
                code="artifacts_not_array",
                path="artifacts",
                message="artifacts must be an array",
            )
        )
        return RunArtifactRegistry()
    artifacts: list[RunArtifactRef] = []
    for index, item in enumerate(payload):
        if not isinstance(item, Mapping):
            violations.append(
                ArtifactManifestViolation(
                    code="artifact_not_object",
                    path=f"artifacts.{index}",
                    message="Artifact entry must be an object",
                )
            )
            continue
        try:
            optional = item.get("optional")
            if not isinstance(optional, bool):
                raise ValueError("optional must be a boolean")
            artifacts.append(
                RunArtifactRef(
                    role=str(item.get("role", "")),
                    path=str(item.get("path", "")),
                    sha256=str(item.get("sha256", "")),
                    size_bytes=int(item.get("size_bytes", -1)),
                    producer_stage=str(item.get("producer_stage", "")),
                    optional=optional,
                    schema_name=_optional_str(item.get("schema_name")),
                    schema_version=_optional_int(
                        item.get("schema_version")
                    ),
                    scope=_optional_str(item.get("scope")),
                )
            )
        except (TypeError, ValueError) as exc:
            violations.append(
                ArtifactManifestViolation(
                    code="invalid_artifact",
                    path=f"artifacts.{index}",
                    message=str(exc),
                )
            )
    try:
        return RunArtifactRegistry.from_artifacts(artifacts)
    except ArtifactManifestBuildError as exc:
        violations.extend(exc.violations)
        return RunArtifactRegistry()


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return int(parsed)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
