from __future__ import annotations

import pandas as pd
import pytest

from examples.benchmark.rmt_conditional_row_policy_experiment import (
    C1_POLICY,
    U0_POLICY,
    ConditionalRowPolicyExperimentConfig,
    ConditionalRowPolicyExperimentOrchestrator,
)


def test_default_phase_s1_grid_has_expected_official_size() -> None:
    config = ConditionalRowPolicyExperimentConfig(show_progress=False)

    assert config.leaf_count == 160
    assert config.expected_official_records == 480


def test_phase_s1_pairing_respects_metric_direction() -> None:
    raw = pd.DataFrame(
        [
            _row("reg", "regression", U0_POLICY.name, "rmse", 10.0),
            _row("reg", "regression", C1_POLICY.name, "rmse", 9.0),
            _row("binary", "classification", U0_POLICY.name, "roc_auc", 0.80),
            _row("binary", "classification", C1_POLICY.name, "roc_auc", 0.82),
        ]
    )

    paired = ConditionalRowPolicyExperimentOrchestrator._pair_with_uniform(raw)
    capped = paired[paired["row_policy_arm"] == C1_POLICY.name].set_index("dataset")

    assert capped.loc["reg", "primary_gain_vs_uniform"] == pytest.approx(0.10)
    assert capped.loc["binary", "primary_gain_vs_uniform"] == pytest.approx(0.02)


def test_phase_s1_practical_gate_requires_selection_and_win_rates() -> None:
    paired = pd.DataFrame(
        [
            {
                **_row("d0", "regression", C1_POLICY.name, "rmse", 9.0),
                "primary_gain_vs_uniform": gain,
            }
            for gain in (0.02, 0.01, 0.03, -0.01)
        ]
    )
    decisions = pd.DataFrame(
        [
            {
                "dataset": "d0",
                "seed": 42,
                "budget_ratio": 0.1,
                "model": "lightgbm",
                "selected_policy": C1_POLICY.name,
            }
        ]
    )
    paired["seed"] = 42
    paired["budget_ratio"] = 0.1
    paired["model"] = "lightgbm"

    gate = ConditionalRowPolicyExperimentOrchestrator._evaluate_gate(
        paired,
        decisions,
    )

    assert gate["status"] == "passed"
    assert gate["capped_selection_rate"] == pytest.approx(1.0)
    assert gate["capped_win_rate_when_selected"] == pytest.approx(0.75)


def _row(
    dataset: str,
    problem_type: str,
    arm: str,
    metric: str,
    value: float,
) -> dict:
    return {
        "dataset": dataset,
        "problem_type": problem_type,
        "seed": 42,
        "budget_ratio": 0.1,
        "model": "lightgbm",
        "row_policy_arm": arm,
        "primary_metric": metric,
        "primary_value": value,
        "tail_mae": float("nan"),
    }
