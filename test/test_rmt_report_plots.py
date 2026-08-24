from __future__ import annotations

import pandas as pd

from examples.benchmark.rmt_report_plots import RMTReportPlotBuilder


def test_report_plot_builder_creates_dataset_model_figures(tmp_path) -> None:
    runs = pd.DataFrame(
        {
            "dataset": ["demo", "demo", "demo", "demo"],
            "model": ["lightgbm"] * 4,
            "scenario_family": [
                "lightgbm__rmt_experts",
                "lightgbm__rmt_experts",
                "lightgbm__random_concatenated",
                "lightgbm__random_concatenated",
            ],
            "budget_ratio": [0.01, 0.05, 0.01, 0.05],
            "rmse_drop": [0.04, 0.01, 0.06, 0.03],
            "fit_time": [1.0, 2.0, 0.5, 1.5],
            "inference_time": [0.1, 0.1, 0.05, 0.05],
            "tree_count_total": [10, 20, 8, 16],
            "max_tree_depth": [3, 4, 3, 4],
        }
    )

    paths = RMTReportPlotBuilder(dpi=40).build_plots(runs, tmp_path)

    assert {path.name for path in paths} == {
        "degradation__demo__lightgbm.png",
        "runtime_quality__demo__lightgbm.png",
        "complexity__demo__lightgbm.png",
    }
    assert all(path.exists() and path.stat().st_size > 0 for path in paths)


def test_report_plot_builder_ignores_incomplete_tables(tmp_path) -> None:
    paths = RMTReportPlotBuilder().build_plots(
        pd.DataFrame({"dataset": ["demo"]}),
        tmp_path,
    )

    assert paths == ()
