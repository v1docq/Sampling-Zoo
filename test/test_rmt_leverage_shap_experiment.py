from __future__ import annotations

import json

import pandas as pd

from examples.benchmark.rmt_leverage_shap_experiment import (
    LeverageShapExperimentConfig,
    run_rmt_leverage_shap_experiment,
)


def test_tiny_leverage_shap_experiment_persists_factorial_artifacts(tmp_path) -> None:
    output = run_rmt_leverage_shap_experiment(
        LeverageShapExperimentConfig(
            budget_multipliers=(4.0,),
            seeds=(7,),
            output_dir=tmp_path,
            show_progress=False,
        )
    )

    meta = json.loads((output / "run_meta.json").read_text(encoding="utf-8"))
    raw = pd.read_csv(output / "leverage_shap_raw_runs.csv")
    assert meta["status"] == "completed"
    assert meta["record_count"] == 10
    assert len(raw) == 10
    assert set(raw["coalition_policy"]) == {"kernel_weight", "leverage"}
    assert raw["row_budget_exact"].all()
    assert (output / "leverage_shap_paired.csv").exists()
    assert (output / "leverage_shap_report.md").exists()
    assert (output / "shap_error_budget_curves.png").exists()
