from __future__ import annotations

import json

import pytest

from examples.benchmark.rmt_bulk_spike_research_program import (
    PROGRAM_STAGE_CLASSIFICATION,
    PROGRAM_STAGE_CONFIRMATION,
    PROGRAM_STAGE_FULL_GRID,
    PROGRAM_STAGE_PILOT,
    PROGRAM_STAGE_REPORT,
    PROGRAM_STAGE_ROW_POLICY,
    PROGRAM_STAGE_SCALING,
    RMTBulkSpikeResearchProgramConfig,
    RMTBulkSpikeResearchProgramOrchestrator,
    build_rmt_bulk_spike_research_plan,
)
from sampling_zoo.core.experiment.research_program import (
    ResearchStageResult,
    ResearchStageStatus,
)


def test_research_program_plan_has_stable_seven_stage_protocol(tmp_path) -> None:
    config = RMTBulkSpikeResearchProgramConfig(output_root=tmp_path)
    plan = build_rmt_bulk_spike_research_plan(config)

    assert plan.stage_ids() == (
        PROGRAM_STAGE_CLASSIFICATION,
        PROGRAM_STAGE_ROW_POLICY,
        PROGRAM_STAGE_PILOT,
        PROGRAM_STAGE_CONFIRMATION,
        PROGRAM_STAGE_FULL_GRID,
        PROGRAM_STAGE_SCALING,
        PROGRAM_STAGE_REPORT,
    )
    assert plan.stage(PROGRAM_STAGE_CLASSIFICATION).expected_records == 480
    assert plan.stage(PROGRAM_STAGE_ROW_POLICY).expected_records == 480
    assert plan.stage(PROGRAM_STAGE_PILOT).expected_records == 320
    assert plan.stage(PROGRAM_STAGE_CONFIRMATION).expected_records == 640
    assert plan.stage(PROGRAM_STAGE_FULL_GRID).expected_records == 1_200
    assert plan.stage(PROGRAM_STAGE_SCALING).expected_records == 1_600
    assert plan.stage(PROGRAM_STAGE_REPORT).expected_records == 16


def test_research_program_persists_state_and_resumes_completed_stages(
    tmp_path,
) -> None:
    calls = []
    config = RMTBulkSpikeResearchProgramConfig(output_root=tmp_path)
    plan = build_rmt_bulk_spike_research_plan(config)

    def handler(stage):
        calls.append(stage.stage_id)
        return ResearchStageResult(
            stage_id=stage.stage_id,
            status=ResearchStageStatus.COMPLETED,
            record_count=stage.expected_records,
            expected_records=stage.expected_records,
            artifact_directory=str(tmp_path / stage.artifact_directory),
            gate_status=(
                "passed"
                if stage.stage_id
                in {
                    PROGRAM_STAGE_CLASSIFICATION,
                    PROGRAM_STAGE_PILOT,
                    PROGRAM_STAGE_CONFIRMATION,
                    PROGRAM_STAGE_FULL_GRID,
                }
                else "not_applicable"
            ),
        )

    handlers = {stage.stage_id: handler for stage in plan.stages}
    orchestrator = RMTBulkSpikeResearchProgramOrchestrator(
        config,
        handlers=handlers,
    )
    orchestrator.run()
    resumed = RMTBulkSpikeResearchProgramOrchestrator(
        config,
        handlers=handlers,
    )
    resumed.run()

    assert calls == list(plan.stage_ids())
    state = json.loads(
        (tmp_path / "research_program_state.json").read_text(encoding="utf-8")
    )
    assert state["status"] == "completed"
    assert len(state["stages"]) == 7


def test_research_program_rejects_incompatible_resume_config(tmp_path) -> None:
    initial = RMTBulkSpikeResearchProgramOrchestrator(
        RMTBulkSpikeResearchProgramConfig(output_root=tmp_path)
    )
    tmp_path.mkdir(parents=True, exist_ok=True)
    initial._write_plan()

    incompatible = RMTBulkSpikeResearchProgramOrchestrator(
        RMTBulkSpikeResearchProgramConfig(
            output_root=tmp_path,
            budgets=(0.01, 0.10, 0.20),
            scaling_budgets=(0.01, 0.10, 0.20),
        )
    )
    with pytest.raises(ValueError, match="resume config"):
        incompatible._write_plan()
