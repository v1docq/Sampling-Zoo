from __future__ import annotations

import pandas as pd

from examples.benchmark.rmt_leverage_sketch_real import (
    DEFAULT_REAL_ARMS,
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
