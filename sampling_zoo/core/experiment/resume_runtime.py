"""Effectful loading and validation for resumable experiment runs."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .artifact_manifest import (
    ArtifactManifestParseFailure,
    ExperimentArtifactManifest,
    RunArtifactRef,
    RunStatus,
    parse_artifact_manifest,
)
from .artifact_runtime import ARTIFACT_MANIFEST_FILENAME, sha256_file
from .errors import ResumeCompatibilityError, ResumeRecordError
from .resume import ResumePlan, ResumePolicy, build_resume_plan


DEFAULT_RAW_RECORDS_PATH = "metrics/rmt_regression_runs.jsonl"


@dataclass(frozen=True)
class ResumeSession:
    run_dir: Path
    manifest: ExperimentArtifactManifest
    plan: ResumePlan
    raw_records_path: Path
    raw_hash_matches_manifest: bool | None
    trailing_partial_line_ignored: bool
    accepted_config_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_run_id": self.manifest.run.run_id,
            "source_status": self.manifest.status.value,
            "source_record_count": self.manifest.record_count,
            "raw_records_path": self.raw_records_path.relative_to(
                self.run_dir
            ).as_posix(),
            "raw_hash_matches_manifest": self.raw_hash_matches_manifest,
            "trailing_partial_line_ignored": (
                self.trailing_partial_line_ignored
            ),
            "accepted_config_hash": self.accepted_config_hash,
            "plan": self.plan.to_dict(),
        }


def load_resume_session(
    run_dir: str | Path,
    *,
    expected_config_hashes: Sequence[str],
    policy: str | ResumePolicy,
    default_seed: int,
    default_raw_records_path: str = DEFAULT_RAW_RECORDS_PATH,
) -> ResumeSession:
    root = Path(run_dir).expanduser().resolve()
    manifest = _load_manifest(root)
    _validate_run_identity(root, manifest)
    accepted_config_hash = _match_config_hash(
        manifest,
        expected_config_hashes,
    )
    raw_artifact = _root_artifact(manifest, role="raw_runs")
    raw_path = _resolve_raw_records_path(
        root,
        raw_artifact,
        default_relative_path=default_raw_records_path,
    )
    hash_matches = _validate_raw_hash(
        raw_path,
        raw_artifact,
        status=manifest.status,
    )
    records, ignored_partial_line = _load_records(
        raw_path,
        recover_trailing_line=manifest.status
        in {RunStatus.RUNNING, RunStatus.FAILED},
    )
    if (
        manifest.status == RunStatus.COMPLETED
        and len(records) != manifest.record_count
    ):
        raise ResumeCompatibilityError(
            scope="experiment.resume.manifest.record_count",
            code="resume_record_count_mismatch",
            message=(
                "Completed run manifest record_count does not match "
                "the raw JSONL record count"
            ),
            details={
                "manifest_record_count": manifest.record_count,
                "raw_record_count": len(records),
            },
        )
    plan = build_resume_plan(
        records,
        policy=policy,
        default_seed=default_seed,
    )
    return ResumeSession(
        run_dir=root,
        manifest=manifest,
        plan=plan,
        raw_records_path=raw_path,
        raw_hash_matches_manifest=hash_matches,
        trailing_partial_line_ignored=ignored_partial_line,
        accepted_config_hash=accepted_config_hash,
    )


def _load_manifest(root: Path) -> ExperimentArtifactManifest:
    manifest_path = root / ARTIFACT_MANIFEST_FILENAME
    if not root.is_dir():
        raise ResumeCompatibilityError(
            scope="experiment.resume.run_dir",
            code="resume_run_directory_missing",
            message=f"Resume directory does not exist: {root}",
        )
    if not manifest_path.is_file():
        raise ResumeCompatibilityError(
            scope="experiment.resume.manifest",
            code="resume_manifest_missing",
            message=(
                f"Resume directory has no {ARTIFACT_MANIFEST_FILENAME}"
            ),
        )
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResumeCompatibilityError(
            scope="experiment.resume.manifest",
            code="resume_manifest_unreadable",
            message=str(exc),
        ) from exc
    parsed = parse_artifact_manifest(payload)
    if isinstance(parsed, ArtifactManifestParseFailure):
        raise ResumeCompatibilityError(
            scope="experiment.resume.manifest",
            code="resume_manifest_invalid",
            message="Artifact manifest failed typed validation",
            details={
                "violations": [
                    violation.to_dict()
                    for violation in parsed.violations
                ]
            },
        )
    return parsed


def _validate_run_identity(
    root: Path,
    manifest: ExperimentArtifactManifest,
) -> None:
    if manifest.run.run_id != root.name:
        raise ResumeCompatibilityError(
            scope="experiment.resume.run_identity",
            code="resume_run_id_mismatch",
            message=(
                "Manifest run_id does not match the resume directory name"
            ),
            details={
                "manifest_run_id": manifest.run.run_id,
                "directory_name": root.name,
            },
        )


def _match_config_hash(
    manifest: ExperimentArtifactManifest,
    expected_hashes: Sequence[str],
) -> str:
    normalized = tuple(dict.fromkeys(str(value) for value in expected_hashes))
    persisted = manifest.run.config.sha256
    if persisted not in normalized:
        raise ResumeCompatibilityError(
            scope="experiment.resume.config",
            code="resume_config_mismatch",
            message=(
                "Resume run was created with an incompatible experiment "
                "configuration"
            ),
            details={
                "manifest_config_sha256": persisted,
                "expected_config_sha256": list(normalized),
            },
        )
    return persisted


def _root_artifact(
    manifest: ExperimentArtifactManifest,
    *,
    role: str,
) -> RunArtifactRef | None:
    matches = [
        artifact
        for artifact in manifest.registry.artifacts
        if artifact.role == role and artifact.scope is None
    ]
    return matches[0] if matches else None


def _resolve_raw_records_path(
    root: Path,
    artifact: RunArtifactRef | None,
    *,
    default_relative_path: str,
) -> Path:
    relative_path = (
        artifact.path if artifact is not None else default_relative_path
    )
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ResumeCompatibilityError(
            scope="experiment.resume.raw_records",
            code="resume_artifact_path_escape",
            message="Raw records path escapes the resume directory",
        ) from exc
    if artifact is not None and not candidate.is_file():
        raise ResumeCompatibilityError(
            scope="experiment.resume.raw_records",
            code="resume_raw_records_missing",
            message=f"Manifest raw_runs artifact is missing: {relative_path}",
        )
    return candidate


def _validate_raw_hash(
    path: Path,
    artifact: RunArtifactRef | None,
    *,
    status: RunStatus,
) -> bool | None:
    if artifact is None or not path.is_file():
        return None
    matches = sha256_file(path) == artifact.sha256
    if not matches and status == RunStatus.COMPLETED:
        raise ResumeCompatibilityError(
            scope="experiment.resume.raw_records",
            code="resume_raw_records_hash_mismatch",
            message=(
                "Completed run raw JSONL hash does not match the artifact "
                "manifest"
            ),
            details={"path": artifact.path},
        )
    return matches


def _load_records(
    path: Path,
    *,
    recover_trailing_line: bool,
) -> tuple[list[Mapping[str, Any]], bool]:
    if not path.exists():
        return [], False
    try:
        lines = path.read_bytes().splitlines()
    except OSError as exc:
        raise ResumeRecordError(
            scope="experiment.resume.raw_records",
            code="resume_raw_records_unreadable",
            message=str(exc),
        ) from exc

    nonempty_indices = [
        index for index, line in enumerate(lines) if line.strip()
    ]
    last_nonempty = nonempty_indices[-1] if nonempty_indices else None
    records: list[Mapping[str, Any]] = []
    ignored_partial = False
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            payload = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            if recover_trailing_line and index == last_nonempty:
                ignored_partial = True
                break
            raise ResumeRecordError(
                scope=f"experiment.resume.raw_records.{index}",
                code="resume_jsonl_invalid",
                message=str(exc),
                details={"line_number": index + 1},
            ) from exc
        if not isinstance(payload, Mapping):
            raise ResumeRecordError(
                scope=f"experiment.resume.raw_records.{index}",
                code="resume_record_not_object",
                message="Every raw JSONL record must be an object",
                details={"line_number": index + 1},
            )
        records.append(dict(payload))
    return records, ignored_partial
