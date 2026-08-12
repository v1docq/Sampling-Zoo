import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from examples.benchmark.rmt_bulk_spike_research_smoke import (
    run_rmt_bulk_spike_research_smoke,
)


def test_research_smoke_runs_both_gated_stages(tmp_path: Path):
    classification_result = pd.DataFrame({"status": ["completed"] * 8})
    topology_result = pd.DataFrame({"status": ["completed"] * 8})

    def fake_topology_run(**kwargs):
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "bulk_spike_gate.json").write_text(
            json.dumps({
                "status": "failed",
                "checks": {
                    "full_dataset_references_completed": True,
                    "specialized_topologies_preserve_exact_budget": True,
                    "regression_tail_noninferiority": True,
                    "classification_probability_and_balance_noninferiority": True,
                },
            }),
            encoding="utf-8",
        )
        return topology_result

    with (
        patch(
            "examples.benchmark.rmt_bulk_spike_research_smoke."
            "run_rmt_classification_guarded_geometry_selector_experiment",
            return_value=classification_result,
        ) as classification_run,
        patch(
            "examples.benchmark.rmt_bulk_spike_research_smoke."
            "build_classification_geometry_gate",
            return_value={"status": "passed"},
        ) as classification_gate,
        patch(
            "examples.benchmark.rmt_bulk_spike_research_smoke."
            "run_rmt_bulk_spike_topology_real_experiment",
            side_effect=fake_topology_run,
        ) as topology_run,
    ):
        result = run_rmt_bulk_spike_research_smoke(
            output_root=tmp_path,
            show_progress=False,
        )

    assert result["status"] == "completed"
    assert result["stages"]["classification_geometry_guard"]["record_count"] == 8
    assert result["stages"]["bulk_spike_topology"]["record_count"] == 8
    assert result["stages"]["bulk_spike_topology"]["scientific_gate_status"] == "failed"
    classification_run.assert_called_once()
    classification_gate.assert_called_once_with(
        tmp_path / "01_classification_geometry_guard",
        expected_records=8,
        expected_selector_records=2,
    )
    topology_run.assert_called_once()

    persisted = json.loads((tmp_path / "smoke_meta.json").read_text(encoding="utf-8"))
    assert persisted["status"] == "completed"


def test_research_smoke_resumes_from_existing_replays(tmp_path: Path):
    classification_dir = tmp_path / "01_classification_geometry_guard"
    topology_dir = tmp_path / "02_bulk_spike_topology"
    classification_dir.mkdir(parents=True)
    topology_dir.mkdir(parents=True)
    pd.DataFrame({"status": ["completed"] * 8}).to_csv(
        classification_dir / "routing_geometry_replay.csv",
        index=False,
    )
    pd.DataFrame({"status": ["completed"] * 8}).to_csv(
        topology_dir / "routing_geometry_replay.csv",
        index=False,
    )
    (topology_dir / "bulk_spike_gate.json").write_text(
        json.dumps({
            "status": "failed",
            "checks": {
                "full_dataset_references_completed": True,
                "specialized_topologies_preserve_exact_budget": True,
                "regression_tail_noninferiority": True,
                "classification_probability_and_balance_noninferiority": True,
            },
        }),
        encoding="utf-8",
    )

    with (
        patch(
            "examples.benchmark.rmt_bulk_spike_research_smoke."
            "run_rmt_classification_guarded_geometry_selector_experiment",
        ) as classification_run,
        patch(
            "examples.benchmark.rmt_bulk_spike_research_smoke."
            "build_classification_geometry_gate",
            return_value={"status": "passed"},
        ),
        patch(
            "examples.benchmark.rmt_bulk_spike_research_smoke."
            "run_rmt_bulk_spike_topology_real_experiment",
        ) as topology_run,
    ):
        result = run_rmt_bulk_spike_research_smoke(
            output_root=tmp_path,
            show_progress=False,
        )

    assert result["status"] == "completed"
    classification_run.assert_not_called()
    topology_run.assert_not_called()


def test_research_smoke_persists_failed_stage(tmp_path: Path):
    classification_result = pd.DataFrame({"status": ["completed"] * 8})
    with (
        patch(
            "examples.benchmark.rmt_bulk_spike_research_smoke."
            "run_rmt_classification_guarded_geometry_selector_experiment",
            return_value=classification_result,
        ),
        patch(
            "examples.benchmark.rmt_bulk_spike_research_smoke."
            "build_classification_geometry_gate",
            return_value={"status": "failed"},
        ),
    ):
        try:
            run_rmt_bulk_spike_research_smoke(
                output_root=tmp_path,
                show_progress=False,
            )
        except RuntimeError as error:
            assert "classification geometry" in str(error)
        else:
            raise AssertionError("smoke run must fail when its gate fails")

    persisted = json.loads((tmp_path / "smoke_meta.json").read_text(encoding="utf-8"))
    assert persisted["status"] == "failed"
    assert persisted["error_type"] == "RuntimeError"
