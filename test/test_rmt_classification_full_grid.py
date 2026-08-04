from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from examples.benchmark.rmt_classification_full_grid import (
    RMTClassificationFullGridConfig,
    RMTClassificationFullGridOrchestrator,
)
from examples.benchmark.rmt_classification_full_grid_plan import (
    ClassAllocationProfile,
    build_rmt_classification_full_grid,
)
from examples.benchmark.rmt_classification_report import (
    RMTClassificationReportPlotBuilder,
    RMTClassificationReportTableBuilder,
)


def test_lightgbm_classification_grid_is_explicit_and_non_cartesian() -> None:
    grid = build_rmt_classification_full_grid(
        budget_ratios=(0.01, 0.05, 0.10, 0.20),
        scenario_groups=("lightgbm_gate",),
    )

    assert len(grid.scenarios) == 41
    assert grid.model_names == ("lightgbm",)
    assert len({scenario.name for scenario in grid.scenarios}) == 41
    assert grid.scenarios[0].name == "lightgbm__full_dataset"


def test_classification_grid_pins_class_safe_rmt_profiles() -> None:
    grid = build_rmt_classification_full_grid(
        budget_ratios=(0.05,),
        scenario_groups=("lightgbm_gate",),
        min_samples_per_class=2,
    )
    by_name = {scenario.name: scenario for scenario in grid.scenarios}

    stratified = by_name[
        "lightgbm__rmt_stratified_capped_spectral__budget_05"
    ].strategy.materialize()
    assert stratified["selection_method"] == "capped_leverage"
    assert stratified["class_coverage_policy"] == "preserve_local_classes"
    assert stratified["class_allocation_policy"] == "proportional"
    assert stratified["min_samples_per_class"] == 2
    assert stratified["budget_feasibility_mode"] == "hard"
    assert stratified["router"] == "spectral"
    assert stratified["partition_model_mode"] == "independent"

    global_capped = by_name[
        "lightgbm__rmt_global_capped_concatenated__budget_05"
    ].strategy.materialize()
    assert global_capped["class_allocation_policy"] == (
        "minimum_then_global"
    )
    assert global_capped["partition_model_mode"] == "concatenated"
    assert "router" not in global_capped

    constrained = by_name[
        "lightgbm__rmt_stratified_capped_constrained_gating__budget_05"
    ].strategy.materialize()
    assert constrained["router"] == "constrained_gating"
    assert constrained["gating_device"] == "auto"


def test_tabpfn_group_contains_only_screening_scenarios() -> None:
    grid = build_rmt_classification_full_grid(
        budget_ratios=(0.10,),
        scenario_groups=("tabpfn_in_context",),
    )

    assert len(grid.scenarios) == 5
    assert grid.model_names == ("tabpfn_in_context",)
    names = {scenario.name for scenario in grid.scenarios}
    assert "tabpfn_in_context__full_dataset" in names
    assert not any("difficulty" in name for name in names)
    assert not any("global_capped" in name for name in names)


def test_classification_full_grid_config_derives_model_pool() -> None:
    config = RMTClassificationFullGridConfig(
        scenario_groups=("lightgbm_gate", "tabpfn_in_context"),
        budget_ratios=(0.10,),
        show_progress=False,
        synthetic_smoke=True,
    )
    orchestrator = RMTClassificationFullGridOrchestrator(config)

    assert config.models == ("lightgbm", "tabpfn_in_context")
    assert len(orchestrator._build_strategy_grid().scenarios) == 16
    assert len(
        orchestrator._build_experiment_plan().effective_config[
            "scenario_grid"
        ]["scenarios"]
    ) == 16


@pytest.mark.parametrize(
    "kwargs",
    [
        {"scenario_groups": ()},
        {"scenario_groups": ("unknown",)},
        {"budget_ratios": ()},
        {"budget_ratios": (0.0,)},
        {"min_samples_per_class": 0},
    ],
)
def test_classification_grid_rejects_invalid_axes(kwargs) -> None:
    complete = {
        "budget_ratios": (0.05,),
        "scenario_groups": ("lightgbm_gate",),
        **kwargs,
    }
    with pytest.raises(ValueError):
        build_rmt_classification_full_grid(**complete)


def test_classification_report_keeps_metric_directions_separate(
    tmp_path: Path,
) -> None:
    records = [
        _record("binary", "lightgbm__full_dataset", "full_dataset", 1.0, "roc_auc", 0.90),
        _record("binary", "lightgbm__random", "random", 0.10, "roc_auc", 0.85),
        _record("multi", "lightgbm__full_dataset", "full_dataset", 1.0, "log_loss", 0.40),
        _record("multi", "lightgbm__rmt", "rmt_contraction", 0.10, "log_loss", 0.45),
    ]
    builder = RMTClassificationReportTableBuilder()
    tables = builder.build_report_tables(records, tmp_path)
    raw = tables["raw"]

    binary = raw[(raw["dataset"] == "binary") & (raw["sampler"] == "random")].iloc[0]
    multiclass = raw[(raw["dataset"] == "multi") & (raw["sampler"] == "rmt_contraction")].iloc[0]
    assert binary["score_drop"] == pytest.approx(0.05)
    assert multiclass["score_drop"] == pytest.approx(0.05)
    assert binary["metric_direction"] == "higher_is_better"
    assert multiclass["metric_direction"] == "lower_is_better"
    assert np.isfinite(binary["brier_score"])
    assert np.isfinite(multiclass["expected_calibration_error"])

    minimal = tables["minimal_budget"]
    at_five_percent = minimal[
        minimal["allowed_absolute_drop"].eq(0.05)
    ]
    assert set(at_five_percent["minimal_budget_ratio"]) == {0.10}

    paths = RMTClassificationReportPlotBuilder().build_plots(
        raw,
        tmp_path / "plots",
    )
    assert paths
    assert all(path.exists() for path in paths)


def _record(
    dataset: str,
    scenario: str,
    sampler: str,
    budget: float,
    primary_metric: str,
    score: float,
) -> dict:
    metrics = {
        "accuracy": 0.75,
        "f1_macro": 0.70,
        "f1_weighted": 0.72,
        "roc_auc": score if primary_metric == "roc_auc" else float("nan"),
        "log_loss": score if primary_metric == "log_loss" else 0.40,
        "brier_score": 0.18,
        "expected_calibration_error": 0.06,
    }
    return {
        "dataset": dataset,
        "strategy_params": {
            "model": "lightgbm",
            "split_label": "split_1",
            "experiment_scenario": scenario,
            "scenario_group": "lightgbm_gate",
            "scenario_family": scenario,
            "strategy": sampler,
            "ensemble_method": "full_dataset" if sampler == "full_dataset" else "voting",
            "partition_model_mode": "concatenated",
            "budget_ratio": budget,
            "class_allocation_policy": (
                ClassAllocationProfile.STRATIFIED_CAPPED.sampler_policy
                if sampler == "rmt_contraction"
                else None
            ),
        },
        "model_metrics": metrics,
        "timings_sec": {"fit": 1.0, "inference": 0.1},
        "sample_stats": {
            "sample_size": 100,
            "model_fit_rows_total": 100,
            "class_coverage_rows_added": 0,
        },
        "extra": {
            "problem_type": "classification",
            "primary_metric": primary_metric,
            "n_classes": 2 if primary_metric == "roc_auc" else 3,
            "partition_diagnostics": {
                "class_balance_summary": {
                    "chunks_with_missing_classes": 0,
                    "single_class_chunks": 0,
                    "class_distribution_drift_avg": 0.05,
                }
            },
            "sampler_diagnostics": {
                "class_coverage_guaranteed": sampler == "rmt_contraction",
                "partition_budget_plan": {
                    "feasible": True,
                    "violations": [],
                },
            },
        },
    }
