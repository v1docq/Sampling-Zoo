"""Pure contracts for resumable, gate-controlled research programs."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence, Tuple


class ResearchStageStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class ResearchStageSpec:
    stage_id: str
    title: str
    artifact_directory: str
    expected_records: int
    dependencies: Tuple[str, ...] = ()
    gate_dependencies: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        stage_id = str(self.stage_id).strip()
        title = str(self.title).strip()
        artifact_directory = str(self.artifact_directory).strip()
        dependencies = tuple(str(value).strip() for value in self.dependencies)
        gate_dependencies = tuple(
            str(value).strip() for value in self.gate_dependencies
        )
        if not stage_id or not title or not artifact_directory:
            raise ValueError(
                "stage_id, title, and artifact_directory must be non-empty"
            )
        if int(self.expected_records) < 0:
            raise ValueError("expected_records must be non-negative")
        if stage_id in dependencies:
            raise ValueError("a research stage cannot depend on itself")
        if len(set(dependencies)) != len(dependencies):
            raise ValueError("research stage dependencies must be unique")
        if not set(gate_dependencies) <= set(dependencies):
            raise ValueError("gate_dependencies must also be dependencies")
        object.__setattr__(self, "stage_id", stage_id)
        object.__setattr__(self, "title", title)
        object.__setattr__(self, "artifact_directory", artifact_directory)
        object.__setattr__(self, "dependencies", dependencies)
        object.__setattr__(self, "gate_dependencies", gate_dependencies)

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_id": self.stage_id,
            "title": self.title,
            "artifact_directory": self.artifact_directory,
            "expected_records": int(self.expected_records),
            "dependencies": list(self.dependencies),
            "gate_dependencies": list(self.gate_dependencies),
        }


@dataclass(frozen=True)
class ResearchStageResult:
    stage_id: str
    status: ResearchStageStatus
    record_count: int
    expected_records: int
    artifact_directory: str
    gate_status: str = "not_applicable"
    message: str = ""

    def __post_init__(self) -> None:
        if int(self.record_count) < 0 or int(self.expected_records) < 0:
            raise ValueError("record counts must be non-negative")
        if self.gate_status not in {
            "not_applicable",
            "passed",
            "failed",
        }:
            raise ValueError("unsupported gate_status")

    @property
    def complete(self) -> bool:
        return (
            self.status is ResearchStageStatus.COMPLETED
            and int(self.record_count) == int(self.expected_records)
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_id": self.stage_id,
            "status": self.status.value,
            "record_count": int(self.record_count),
            "expected_records": int(self.expected_records),
            "artifact_directory": self.artifact_directory,
            "gate_status": self.gate_status,
            "message": self.message,
        }


@dataclass(frozen=True)
class ResearchProgramPlan:
    stages: Tuple[ResearchStageSpec, ...]

    def __post_init__(self) -> None:
        if not self.stages:
            raise ValueError("research program must contain stages")
        stage_ids = tuple(stage.stage_id for stage in self.stages)
        if len(set(stage_ids)) != len(stage_ids):
            raise ValueError("research stage identifiers must be unique")
        positions = {stage_id: index for index, stage_id in enumerate(stage_ids)}
        for stage in self.stages:
            missing = set(stage.dependencies) - set(stage_ids)
            if missing:
                raise ValueError(
                    f"unknown dependencies for {stage.stage_id}: {sorted(missing)}"
                )
            if any(
                positions[dependency] >= positions[stage.stage_id]
                for dependency in stage.dependencies
            ):
                raise ValueError(
                    "research stages must be supplied in topological order"
                )

    def stage(self, stage_id: str) -> ResearchStageSpec:
        for stage in self.stages:
            if stage.stage_id == stage_id:
                return stage
        raise KeyError(stage_id)

    def stage_ids(self) -> Tuple[str, ...]:
        return tuple(stage.stage_id for stage in self.stages)

    def blocked_reason(
        self,
        stage_id: str,
        results: Mapping[str, ResearchStageResult],
    ) -> str | None:
        stage = self.stage(stage_id)
        for dependency in stage.dependencies:
            result = results.get(dependency)
            if result is None or not result.complete:
                return f"dependency_not_completed:{dependency}"
        for dependency in stage.gate_dependencies:
            if results[dependency].gate_status != "passed":
                return f"dependency_gate_failed:{dependency}"
        return None

    def to_dict(self) -> dict[str, object]:
        return {"stages": [stage.to_dict() for stage in self.stages]}


def deserialize_stage_result(payload: Mapping[str, object]) -> ResearchStageResult:
    return ResearchStageResult(
        stage_id=str(payload["stage_id"]),
        status=ResearchStageStatus(str(payload["status"])),
        record_count=int(payload.get("record_count", 0)),
        expected_records=int(payload.get("expected_records", 0)),
        artifact_directory=str(payload.get("artifact_directory", "")),
        gate_status=str(payload.get("gate_status", "not_applicable")),
        message=str(payload.get("message", "")),
    )


def selected_stage_closure(
    plan: ResearchProgramPlan,
    selected_stage_ids: Sequence[str],
) -> Tuple[str, ...]:
    selected = {str(value) for value in selected_stage_ids}
    unknown = selected - set(plan.stage_ids())
    if unknown:
        raise ValueError(f"unknown research stages: {sorted(unknown)}")
    closure = set(selected)
    changed = True
    while changed:
        changed = False
        for stage_id in tuple(closure):
            dependencies = set(plan.stage(stage_id).dependencies)
            if not dependencies <= closure:
                closure.update(dependencies)
                changed = True
    return tuple(stage_id for stage_id in plan.stage_ids() if stage_id in closure)
