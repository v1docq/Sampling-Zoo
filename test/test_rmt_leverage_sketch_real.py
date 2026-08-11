from __future__ import annotations

import pandas as pd

from examples.benchmark.rmt_leverage_sketch_real import (
    DEFAULT_REAL_ARMS,
    RealLeverageSketchArm,
    RealLeverageSketchConfig,
    RealLeverageSketchOrchestrator,
)


def test_real_phase_s_expected_record_count() -> None:
    config = RealLeverageSketchConfig(
        regression_tasks=("diamonds",),
        classification_tasks=("adult",),
        models=("lightgbm",),
        budget_ratios=(0.1, 0.2),
        seeds=(1, 2),
        show_progress=False,
    )

    assert config.expected_a9_records == 2 * 1 * 2 * 2 * len(DEFAULT_REAL_ARMS)
    assert all(
        arm.sampler_params()["class_allocation_policy"] == "proportional"
        for arm in config.arms
    )


def test_real_phase_s_pairing_respects_metric_direction(tmp_path) -> None:
    config = RealLeverageSketchConfig(
        regression_tasks=("diamonds",),
        classification_tasks=(),
        budget_ratios=(0.1,),
        seeds=(1,),
        output_dir=tmp_path,
        show_progress=False,
    )
    orchestrator = RealLeverageSketchOrchestrator(config)
    raw = pd.DataFrame(
        [
            {
                "dataset": "diamonds",
                "problem_type": "regression",
                "seed": 1,
                "budget_ratio": 0.1,
                "model": "lightgbm",
                "row_arm_id": arm,
                "primary_metric": "rmse",
                "primary_value": value,
                "tail_mae": value * 2,
                "ensemble_fit_seconds": 10.0,
                "gram_relative_error_mean": gram,
                "status": "completed",
            }
            for arm, value, gram in (
                ("R0_uniform", 10.0, 0.5),
                ("R3_robust_mixture_ipw", 9.0, 0.3),
            )
        ]
    )

    paired = orchestrator._pair_with_uniform(raw)
    candidate = paired[paired["row_arm_id"] == "R3_robust_mixture_ipw"].iloc[0]

    assert candidate["primary_gain_vs_uniform"] == 0.1
    assert candidate["tail_gain_vs_uniform"] == 0.1
    assert candidate["gram_error_delta_vs_uniform"] == -0.2


def test_real_phase_s_flattens_canonical_tail_metric() -> None:
    record = {
        "dataset": "diamonds",
        "problem_type": "regression",
        "seed": 1,
        "budget_ratio": 0.1,
        "model": "lightgbm",
        "status": "completed",
        "test": {
            "primary_metric": "rmse",
            "primary_value": 9.0,
            "metrics": {
                "rmse": 9.0,
                "mae": 5.0,
                "tail_mean_absolute_error": 12.5,
            },
        },
    }

    flattened = RealLeverageSketchOrchestrator._flatten_a9_record(
        record,
        RealLeverageSketchArm("R0_uniform", "uniform"),
    )

    assert flattened["tail_mae"] == 12.5


def test_real_phase_s_ranking_uses_raw_paired_runs() -> None:
    paired = pd.DataFrame(
        {
            "row_arm_id": ["R3", "R3", "R3", "R0", "R0", "R0"],
            "primary_gain_vs_uniform": [0.4, -0.1, -0.1, 0.0, 0.0, 0.0],
            "tail_gain_vs_uniform": [0.2, -0.2, 0.0, 0.0, 0.0, 0.0],
            "gram_relative_error_mean": [0.1, 0.2, 0.3, 0.4, 0.4, 0.4],
            "fit_time_ratio_vs_uniform": [1.1, 0.9, 1.0, 1.0, 1.0, 1.0],
        }
    )

    ranking = RealLeverageSketchOrchestrator._build_arm_ranking(paired)
    candidate = ranking[ranking["row_arm_id"] == "R3"].iloc[0]

    assert candidate["runs"] == 3
    assert candidate["median_gain"] == -0.1
    assert candidate["win_rate"] == 1 / 3
    assert candidate["worst_gain"] == -0.1
