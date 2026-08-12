from __future__ import annotations

import pytest

from sampling_zoo.core.experiment.research_program import (
    ResearchProgramPlan,
    ResearchStageResult,
    ResearchStageSpec,
    ResearchStageStatus,
    selected_stage_closure,
)


def _plan() -> ResearchProgramPlan:
    return ResearchProgramPlan(
        stages=(
            ResearchStageSpec("a", "A", "a", 2),
            ResearchStageSpec(
                "b",
                "B",
                "b",
                3,
                dependencies=("a",),
                gate_dependencies=("a",),
            ),
            ResearchStageSpec(
                "c",
                "C",
                "c",
                1,
                dependencies=("b",),
            ),
        )
    )


def test_research_program_requires_topological_order() -> None:
    with pytest.raises(ValueError, match="topological order"):
        ResearchProgramPlan(
            stages=(
                ResearchStageSpec("b", "B", "b", 1, dependencies=("a",)),
                ResearchStageSpec("a", "A", "a", 1),
            )
        )


def test_research_program_distinguishes_completion_and_gate() -> None:
    plan = _plan()
    failed_gate = ResearchStageResult(
        stage_id="a",
        status=ResearchStageStatus.COMPLETED,
        record_count=2,
        expected_records=2,
        artifact_directory="a",
        gate_status="failed",
    )
    assert plan.blocked_reason("b", {"a": failed_gate}) == (
        "dependency_gate_failed:a"
    )
    passed_gate = ResearchStageResult(
        stage_id="a",
        status=ResearchStageStatus.COMPLETED,
        record_count=2,
        expected_records=2,
        artifact_directory="a",
        gate_status="passed",
    )
    assert plan.blocked_reason("b", {"a": passed_gate}) is None


def test_selected_stage_closure_is_stable_and_includes_dependencies() -> None:
    assert selected_stage_closure(_plan(), ("c",)) == ("a", "b", "c")
