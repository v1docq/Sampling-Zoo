from __future__ import annotations

import pandas as pd
import pytest

from examples.benchmark.rmt_scaling_law_analysis import (
    run_scaling_law_analysis,
)
from sampling_zoo.core.experiment.scaling_laws import (
    BudgetScalingLawFitter,
    ScalingLawFitSpec,
    ScalingLawObservation,
    primary_value_to_loss,
)


def test_budget_scaling_law_recovers_monotone_power_curve() -> None:
    observations = []
    for seed in range(5):
        for budget in (0.02, 0.05, 0.10, 0.20, 1.0):
            loss = 0.4 + 0.08 * budget ** (-0.6) + seed * 1e-4
            observations.append(
                ScalingLawObservation(
                    budget_ratio=budget,
                    primary_value=loss,
                    primary_metric="rmse",
                    replicate_id=str(seed),
                    is_full_reference=(budget == 1.0),
                )
            )
    result = BudgetScalingLawFitter(
        ScalingLawFitSpec(bootstrap_iterations=100)
    ).fit(observations)

    required = dict(result.required_budget_by_degradation)

    assert result.r_squared > 0.999
    assert result.exponent == pytest.approx(0.6, abs=0.02)
    assert result.exponent_confidence_interval is not None
    assert result.predict_loss(0.05) > result.predict_loss(0.20)
    assert required[0.01] is not None
    assert required[0.01] > required[0.05]


def test_roc_auc_is_converted_to_a_minimization_loss() -> None:
    assert primary_value_to_loss("roc_auc", 0.91) == pytest.approx(0.09)
    assert primary_value_to_loss("log_loss", 0.31) == pytest.approx(0.31)


def test_scaling_law_artifact_builder_persists_report_and_figure(tmp_path) -> None:
    rows = []
    for arm_name, offset in (
        ("B0_standard_A9", 0.03),
        ("B3_validation_selected", 0.02),
    ):
        for seed in (42, 43, 44):
            for budget in (0.05, 0.10, 0.20):
                rows.append(
                    {
                        "dataset": "synthetic",
                        "seed": seed,
                        "budget_ratio": budget,
                        "model": "lightgbm",
                        "arm_name": arm_name,
                        "status": "completed",
                        "test_primary_metric": "rmse",
                        "test_primary_value": (
                            0.5 + offset * budget ** (-0.5) + seed * 1e-5
                        ),
                        "full_reference_primary_value": 0.5 + offset,
                    }
                )
    runs_path = tmp_path / "runs.csv"
    output_dir = tmp_path / "scaling"
    pd.DataFrame(rows).to_csv(runs_path, index=False)

    result = run_scaling_law_analysis(
        runs_path,
        output_dir,
        bootstrap_iterations=0,
    )

    assert result.shape[0] == 2
    assert result["r_squared"].gt(0.99).all()
    assert (output_dir / "scaling_law_points.csv").exists()
    assert (output_dir / "scaling_law_fits.csv").exists()
    assert (output_dir / "scaling_law_fits.json").exists()
    assert (output_dir / "scaling_law_report.md").exists()
    assert (output_dir / "scaling_law_curves.png").exists()
