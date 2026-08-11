from __future__ import annotations

import pandas as pd
import pytest

from examples.benchmark.rmt_tail_guard_holdout import (
    DEFAULT_TAIL_GUARDED_SELECTOR_ARM_NAME,
    REFERENCE_ARM,
    TailGuardHoldoutGateSpec,
    evaluate_tail_guard_holdout,
)


def _replay(candidate_values: tuple[float, ...]) -> pd.DataFrame:
    rows = []
    for seed, candidate in enumerate(candidate_values, start=42):
        identity = {
            "task_name": "synthetic",
            "seed": seed,
            "budget_ratio": 0.1,
            "model": "ridge",
            "status": "completed",
            "test_primary_metric": "rmse",
        }
        rows.extend(
            [
                {
                    **identity,
                    "arm_name": REFERENCE_ARM,
                    "selected_arm_name": REFERENCE_ARM,
                    "test_primary_value": 10.0,
                    "test_tail_mean_absolute_error": 20.0,
                },
                {
                    **identity,
                    "arm_name": "A5_gmm_posterior",
                    "selected_arm_name": "A5_gmm_posterior",
                    "test_primary_value": candidate,
                    "test_tail_mean_absolute_error": 19.0,
                },
                {
                    **identity,
                    "arm_name": "A6_cosine_negative_control",
                    "selected_arm_name": "A6_cosine_negative_control",
                    "test_primary_value": candidate,
                    "test_tail_mean_absolute_error": 19.0,
                },
                {
                    **identity,
                    "arm_name": DEFAULT_TAIL_GUARDED_SELECTOR_ARM_NAME,
                    "selected_arm_name": "A5_gmm_posterior",
                    "test_primary_value": candidate,
                    "test_tail_mean_absolute_error": 19.0,
                },
            ]
        )
    return pd.DataFrame(rows)


def test_holdout_gate_passes_complete_noninferior_grid() -> None:
    replay = _replay((9.8, 9.9, 9.7))
    result, pairs, by_dataset = evaluate_tail_guard_holdout(
        replay,
        expected_records=12,
        spec=TailGuardHoldoutGateSpec(),
    )

    assert result.passed
    assert result.nonbaseline_win_rate == 1.0
    assert len(pairs) == 3
    assert by_dataset.iloc[0].median_gain_vs_a2 == pytest.approx(0.02)


def test_holdout_gate_reports_worst_case_primary_violation() -> None:
    replay = _replay((9.8, 10.2, 9.7))
    result, _, _ = evaluate_tail_guard_holdout(
        replay,
        expected_records=12,
        spec=TailGuardHoldoutGateSpec(max_primary_relative_degradation=0.01),
    )

    assert not result.passed
    assert "primary_worst_case_exceeds_limit" in result.violations
