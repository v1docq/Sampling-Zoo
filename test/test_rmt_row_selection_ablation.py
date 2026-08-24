from __future__ import annotations

import pandas as pd
import pytest

from examples.benchmark.rmt_row_selection_ablation import (
    DEFAULT_ROW_SELECTION_METHODS,
    RMTRowSelectionAblationConfig,
    make_rmt_row_selection_strategy_configs,
    normalize_row_selection_methods,
)
from examples.benchmark.rmt_report_tables import RMTReportTableBuilder


def test_row_ablation_grid_varies_only_row_selection_method() -> None:
    config = RMTRowSelectionAblationConfig(
        regression_tasks=("diamonds",),
        models=("ridge",),
        budget_ratios=(0.01, 0.05, 0.20),
        show_progress=False,
    )
    configs = make_rmt_row_selection_strategy_configs(config)
    rmt_configs = [value for key, value in configs.items() if key != "full_dataset"]

    assert len(rmt_configs) == 12
    assert {item["selection_method"] for item in rmt_configs} == set(
        DEFAULT_ROW_SELECTION_METHODS
    )
    assert {item["cluster_selection_metric"] for item in rmt_configs} == {
        "balanced_silhouette"
    }
    assert {item["leverage_cap_quantile"] for item in rmt_configs} == {0.95}

    comparable = []
    for item in rmt_configs[:4]:
        normalized = dict(item)
        normalized.pop("selection_method")
        comparable.append(normalized)
    assert all(item == comparable[0] for item in comparable[1:])


@pytest.mark.parametrize("values", [(), ("unknown",), ("hybrid", "HYBRID")])
def test_row_method_normalization_rejects_invalid_values(values) -> None:
    with pytest.raises(ValueError):
        normalize_row_selection_methods(values)


def test_row_selection_report_pairs_methods_and_checks_fingerprint() -> None:
    raw = pd.DataFrame(
        [
            {
                "dataset": "diamonds",
                "model": "lightgbm",
                "sampler": "rmt_contraction",
                "budget_ratio": 0.01,
                "row_selection_method": method,
                "partition_membership_fingerprint": "same-partitions",
                "rmse": rmse,
                "rmse_drop": rmse / 100.0,
                "fit_time": 2.0,
                "inference_time": 0.1,
                "total_train_rows": 100,
                "target_mean_abs_drift_avg": drift,
                "target_quantile_l1_drift_avg": drift * 2,
            }
            for method, rmse, drift in (
                ("hybrid", 10.0, 2.0),
                ("capped_leverage", 8.0, 1.0),
            )
        ]
    )

    comparison = RMTReportTableBuilder._build_row_selection_comparison(raw)
    capped = comparison[
        comparison["row_selection_method"] == "capped_leverage"
    ].iloc[0]

    assert capped["rmse_delta_vs_hybrid"] == pytest.approx(-2.0)
    assert capped["target_mean_abs_drift_avg_delta_vs_hybrid"] == pytest.approx(
        -1.0
    )
    assert bool(capped["partition_fingerprint_consistent"])
