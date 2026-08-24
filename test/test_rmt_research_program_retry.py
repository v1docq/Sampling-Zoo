from __future__ import annotations

import json
from pathlib import Path

from examples.benchmark.rmt_bulk_spike_research_program import (
    PROGRAM_STAGE_CLASSIFICATION,
    PROGRAM_STAGE_CONFIRMATION,
    PROGRAM_STAGE_ROW_POLICY,
    RMTBulkSpikeResearchProgramConfig,
    RMTBulkSpikeResearchProgramOrchestrator,
)
from examples.benchmark.rmt_bulk_spike_topology_real_experiment import (
    PRACTICAL_EXPLORATION_GATE_PROFILE,
)
from sampling_zoo.core.experiment.research_program import (
    ResearchStageResult,
    ResearchStageSpec,
    ResearchStageStatus,
)


def _handler(
    output_root: Path,
    calls: list[str],
    *,
    classification_gate_status: str,
):
    def run(stage: ResearchStageSpec) -> ResearchStageResult:
        calls.append(stage.stage_id)
        stage_dir = output_root / stage.artifact_directory
        stage_dir.mkdir(parents=True, exist_ok=True)
        (stage_dir / "marker.txt").write_text(
            f"run={len(calls)}\n",
            encoding="utf-8",
        )
        gate_status = (
            classification_gate_status
            if stage.stage_id == PROGRAM_STAGE_CLASSIFICATION
            else "passed"
        )
        return ResearchStageResult(
            stage_id=stage.stage_id,
            status=ResearchStageStatus.COMPLETED,
            record_count=stage.expected_records,
            expected_records=stage.expected_records,
            artifact_directory=str(stage_dir),
            gate_status=gate_status,
        )

    return run


def _orchestrator(
    output_root: Path,
    calls: list[str],
    *,
    classification_gate_status: str,
    topology_gate_profile: str = "strict_research",
) -> RMTBulkSpikeResearchProgramOrchestrator:
    config = RMTBulkSpikeResearchProgramConfig(
        output_root=output_root,
        show_progress=False,
        topology_gate_profile=topology_gate_profile,
    )
    probe = RMTBulkSpikeResearchProgramOrchestrator(config)
    handler = _handler(
        output_root,
        calls,
        classification_gate_status=classification_gate_status,
    )
    return RMTBulkSpikeResearchProgramOrchestrator(
        config,
        handlers={stage_id: handler for stage_id in probe.plan.stage_ids()},
    )


def test_retry_archives_invalidated_stages_and_preserves_independent_result(
    tmp_path: Path,
) -> None:
    first_calls: list[str] = []
    first = _orchestrator(
        tmp_path,
        first_calls,
        classification_gate_status="failed",
    )
    first.run()
    plan_path = tmp_path / "research_program_plan.json"
    legacy_plan = json.loads(plan_path.read_text(encoding="utf-8"))
    legacy_plan["config"].pop("topology_gate_profile")
    plan_path.write_text(
        json.dumps(legacy_plan, indent=2),
        encoding="utf-8",
    )

    assert first_calls == [
        PROGRAM_STAGE_CLASSIFICATION,
        PROGRAM_STAGE_ROW_POLICY,
    ]

    second_calls: list[str] = []
    second = _orchestrator(
        tmp_path,
        second_calls,
        classification_gate_status="passed",
    )
    second.run(retry_stages=(PROGRAM_STAGE_CLASSIFICATION,))

    assert PROGRAM_STAGE_ROW_POLICY not in second_calls
    assert second_calls == [
        stage_id
        for stage_id in second.plan.stage_ids()
        if stage_id != PROGRAM_STAGE_ROW_POLICY
    ]
    retry_manifests = tuple(tmp_path.glob("research_program_retry_*.json"))
    assert len(retry_manifests) == 1
    manifest = json.loads(retry_manifests[0].read_text(encoding="utf-8"))
    assert manifest["requested_stage_ids"] == [PROGRAM_STAGE_CLASSIFICATION]
    assert PROGRAM_STAGE_ROW_POLICY in manifest["preserved_stage_ids"]
    assert manifest["archived_artifacts"][0]["stage_id"] == (
        PROGRAM_STAGE_CLASSIFICATION
    )
    archived_marker = (
        Path(manifest["archived_artifacts"][0]["destination"])
        / "marker.txt"
    )
    assert archived_marker.exists()

    state = json.loads(
        (tmp_path / "research_program_state.json").read_text(encoding="utf-8")
    )
    assert state["status"] == "completed"


def test_retry_records_explicit_gate_profile_protocol_amendment(
    tmp_path: Path,
) -> None:
    first_calls: list[str] = []
    first = _orchestrator(
        tmp_path,
        first_calls,
        classification_gate_status="passed",
    )
    first.run()
    plan_path = tmp_path / "research_program_plan.json"
    legacy_plan = json.loads(plan_path.read_text(encoding="utf-8"))
    legacy_plan["config"].pop("topology_gate_profile")
    plan_path.write_text(
        json.dumps(legacy_plan, indent=2),
        encoding="utf-8",
    )

    second_calls: list[str] = []
    second = _orchestrator(
        tmp_path,
        second_calls,
        classification_gate_status="passed",
        topology_gate_profile=PRACTICAL_EXPLORATION_GATE_PROFILE,
    )
    second.run(retry_stages=(PROGRAM_STAGE_CONFIRMATION,))

    assert second_calls[0] == PROGRAM_STAGE_CONFIRMATION
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    amendment = plan["protocol_amendments"][-1]
    assert amendment["field"] == "topology_gate_profile"
    assert amendment["from"] == "strict_research"
    assert amendment["to"] == PRACTICAL_EXPLORATION_GATE_PROFILE
    retry_manifests = sorted(tmp_path.glob("research_program_retry_*.json"))
    manifest = json.loads(retry_manifests[-1].read_text(encoding="utf-8"))
    assert manifest["protocol_amendment"] == amendment
