from __future__ import annotations

import pandas as pd
import pytest

from examples.benchmark import rmt_regression_medium_datasets
from examples.benchmark.benchmark_logging import BenchmarkLogger
from examples.benchmark.rmt_partition_selection_ablation import (
    DEFAULT_MECHANISM_SMOKE_BUDGET_RATIOS,
    DEFAULT_MECHANISM_SMOKE_TASKS,
    DEFAULT_PARTITION_SELECTION_METRICS,
    RMTPartitionSelectionAblationConfig,
    RMTPartitionSelectionAblationOrchestrator,
    make_rmt_partition_selection_grid,
    make_rmt_partition_selection_strategy_configs,
    normalize_budget_ratios,
    normalize_partition_selection_metrics,
)
from examples.benchmark.rmt_report_tables import RMTReportTableBuilder
from sampling_zoo.core.experiment.morphisms import normalize_strategy_grid
from sampling_zoo.core.experiment.resume import leaf_run_key_from_components


def test_ablation_config_normalizes_stable_sequences() -> None:
    config = RMTPartitionSelectionAblationConfig(
        regression_tasks=["diamonds"],
        models="RIDGE",
        budget_ratios=[0.01, 0.1],
        cluster_selection_metrics=[
            "BALANCED_SILHOUETTE",
            "validation_proxy",
        ],
        show_progress=False,
    )

    assert config.regression_tasks == ("diamonds",)
    assert config.models == ("ridge",)
    assert config.budget_ratios == (0.01, 0.1)
    assert config.cluster_selection_metrics == (
        "balanced_silhouette",
        "validation_proxy",
    )
    assert config.to_regression_config().strategies == ("rmt_contraction",)
    assert config.to_regression_config().ensemble_methods == (
        "routed_weighted",
    )
    assert config.to_regression_config().router_modes == ("spectral",)
    assert config.model_n_jobs == 1


@pytest.mark.parametrize("model_n_jobs", [0, -1, True, 1.5, "2"])
def test_ablation_config_rejects_invalid_model_worker_limit(
    model_n_jobs,
) -> None:
    with pytest.raises(ValueError, match="model_n_jobs"):
        RMTPartitionSelectionAblationConfig(model_n_jobs=model_n_jobs)


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ((), "must contain"),
        (("unknown",), "Unsupported"),
        (("validation_proxy", "validation_proxy"), "duplicates"),
    ],
)
def test_partition_metric_normalization_rejects_invalid_values(
    values: tuple[str, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        normalize_partition_selection_metrics(values)


@pytest.mark.parametrize(
    "values",
    [(), (0.0,), (1.01,), (float("nan"),), (0.1, 0.10)],
)
def test_budget_normalization_rejects_ambiguous_grids(
    values: tuple[float, ...],
) -> None:
    with pytest.raises(ValueError):
        normalize_budget_ratios(values)


def test_partition_selection_grid_is_deterministic_and_budget_paired() -> None:
    first = make_rmt_partition_selection_grid(
        cluster_selection_metrics=(
            "balanced_silhouette",
            "validation_proxy",
        ),
        budget_ratios=(0.01, 0.03),
    )
    second = make_rmt_partition_selection_grid(
        cluster_selection_metrics=(
            "balanced_silhouette",
            "validation_proxy",
        ),
        budget_ratios=(0.01, 0.03),
    )

    assert first == second
    assert [point.config_name for point in first] == [
        (
            "rmt_contraction__view_gaussian__routed_weighted"
            "__router_spectral__selection_balanced_silhouette__budget_01"
        ),
        (
            "rmt_contraction__view_gaussian__routed_weighted"
            "__router_spectral__selection_validation_proxy__budget_01"
        ),
        (
            "rmt_contraction__view_gaussian__routed_weighted"
            "__router_spectral__selection_balanced_silhouette__budget_03"
        ),
        (
            "rmt_contraction__view_gaussian__routed_weighted"
            "__router_spectral__selection_validation_proxy__budget_03"
        ),
    ]


def test_materialized_ablation_configs_vary_only_selection_metric() -> None:
    config = RMTPartitionSelectionAblationConfig(
        regression_tasks=("diamonds",),
        models=("ridge",),
        budget_ratios=(0.1,),
        cluster_selection_metrics=(
            "balanced_silhouette",
            "validation_proxy",
        ),
        validation_proxy_fraction=0.25,
        validation_proxy_min_partition_rows=12,
        validation_proxy_smoothing=0.5,
        show_progress=False,
    )
    configs = make_rmt_partition_selection_strategy_configs(config)
    strategy_configs = [
        value
        for key, value in configs.items()
        if key != "full_dataset"
    ]

    assert len(strategy_configs) == 2
    assert {
        item["cluster_selection_metric"] for item in strategy_configs
    } == {"balanced_silhouette", "validation_proxy"}
    for item in strategy_configs:
        assert item["ensemble_method"] == "routed_weighted"
        assert item["router"] == "spectral"
        assert item["view_strategy"] == "gaussian"
        assert item["validation_proxy_fraction"] == 0.25
        assert item["validation_proxy_min_partition_rows"] == 12
        assert item["validation_proxy_smoothing"] == 0.5

    left = dict(strategy_configs[0])
    right = dict(strategy_configs[1])
    left.pop("cluster_selection_metric")
    right.pop("cluster_selection_metric")
    assert left == right


def test_ablation_strategy_names_keep_resume_leaf_runs_distinct() -> None:
    config = RMTPartitionSelectionAblationConfig(
        regression_tasks=("diamonds",),
        models=("ridge",),
        budget_ratios=(0.1,),
        show_progress=False,
    )
    strategy_grid = normalize_strategy_grid(
        make_rmt_partition_selection_strategy_configs(config)
    ).materialize()
    leaf_keys = {
        leaf_run_key_from_components(
            dataset="diamonds",
            split="holdout",
            model="ridge",
            strategy=name,
            strategy_config=strategy_config,
            seed=42,
        ).key
        for name, strategy_config in strategy_grid.items()
        if name != "full_dataset"
    }

    assert len(leaf_keys) == 4


def test_ablation_plan_and_metadata_capture_scientific_axes(
    tmp_path,
) -> None:
    config = RMTPartitionSelectionAblationConfig(
        regression_tasks=("diamonds",),
        models=("ridge",),
        budget_ratios=(0.1,),
        show_progress=False,
    )
    orchestrator = RMTPartitionSelectionAblationOrchestrator(config)
    plan = orchestrator._build_experiment_plan()
    logger = BenchmarkLogger(
        run_id="partition-selection-test",
        artifacts_root=tmp_path,
    )
    metadata = orchestrator._build_run_meta(logger, [], "running")

    assert plan.effective_config["experiment_kind"] == (
        "rmt_partition_selection_ablation"
    )
    assert plan.effective_config["cluster_selection_metrics"] == (
        DEFAULT_PARTITION_SELECTION_METRICS
    )
    assert plan.effective_config["model_n_jobs"] == 1
    assert metadata["experiment_kind"] == "rmt_partition_selection_ablation"
    assert metadata["cluster_selection_metrics"] == list(
        DEFAULT_PARTITION_SELECTION_METRICS
    )
    assert metadata["model_n_jobs"] == 1


def test_mechanism_smoke_scope_matches_first_experiment_gate() -> None:
    assert DEFAULT_MECHANISM_SMOKE_TASKS == (
        "Brazilian_houses",
        "diamonds",
        "pol",
    )
    assert DEFAULT_MECHANISM_SMOKE_BUDGET_RATIOS == (0.01, 0.05, 0.20)


def test_new_partition_selectors_materialize_budget_and_proxy_policies() -> None:
    config = RMTPartitionSelectionAblationConfig(
        regression_tasks=("diamonds",),
        models=("ridge",),
        budget_ratios=(0.05,),
        cluster_selection_metrics=(
            "budget_aware_validation_proxy",
            "downstream_proxy",
        ),
        show_progress=False,
    )

    configs = make_rmt_partition_selection_strategy_configs(config)
    budget_aware = next(
        value
        for key, value in configs.items()
        if "selection_budget_aware_validation_proxy" in key
    )
    downstream = next(
        value
        for key, value in configs.items()
        if "selection_downstream_proxy" in key
    )

    assert budget_aware["budget_feasibility_mode"] == "hard"
    assert budget_aware["include_single_partition_candidate"] is True
    assert budget_aware["min_sampled_rows_per_partition"] == 32
    assert downstream["budget_feasibility_mode"] == "hard"
    assert downstream["cluster_ensemble_method"] == "best_score"
    assert downstream["downstream_proxy_shortlist_size"] == 3
    assert downstream["downstream_proxy_n_estimators"] == 32


def test_ablation_orchestrator_applies_model_worker_limit(monkeypatch) -> None:
    calls = []
    marker = object()

    def fake_make_model_pool(**kwargs):
        calls.append(kwargs)
        return {"ridge": marker}

    monkeypatch.setattr(
        "examples.benchmark.rmt_partition_selection_ablation.make_model_pool",
        fake_make_model_pool,
    )
    orchestrator = RMTPartitionSelectionAblationOrchestrator(
        RMTPartitionSelectionAblationConfig(
            regression_tasks=("diamonds",),
            models=("ridge",),
            model_n_jobs=2,
            show_progress=False,
        )
    )

    assert orchestrator._make_model_pool() == {"ridge": marker}
    assert calls == [
        {
            "seed": 42,
            "model_names": ("ridge",),
            "problem_type": "regression",
            "n_jobs": 2,
        }
    ]


def test_base_orchestrator_keeps_legacy_model_pool_defaults(monkeypatch) -> None:
    calls = []

    def fake_make_model_pool(**kwargs):
        calls.append(kwargs)
        return {}

    monkeypatch.setattr(
        rmt_regression_medium_datasets,
        "make_model_pool",
        fake_make_model_pool,
    )
    orchestrator = rmt_regression_medium_datasets.RMTRegressionExperimentOrchestrator(
        rmt_regression_medium_datasets.RMTRegressionExperimentConfig(
            regression_tasks=("diamonds",),
            models=("ridge",),
            show_progress=False,
        )
    )

    assert orchestrator._make_model_pool() == {}
    assert calls == [
        {
            "seed": 42,
            "model_names": ("ridge",),
            "problem_type": "regression",
        }
    ]


def test_partition_selection_comparison_reports_paired_deltas() -> None:
    rows = []
    for metric, rmse, fit_time, selected_n_partitions in (
        ("balanced_silhouette", 10.0, 2.0, 4.0),
        ("validation_proxy", 8.0, 2.5, 3.0),
    ):
        rows.append(
            {
                "dataset": "synthetic",
                "model": "ridge",
                "sampler": "rmt_contraction",
                "ensemble_method": "routed_weighted",
                "router": "spectral",
                "view_strategy": "gaussian",
                "partition_selection_method": "auto",
                "cluster_ensemble_method": "coassociation",
                "budget_ratio": 0.1,
                "cluster_selection_metric": metric,
                "rmse": rmse,
                "rmse_drop": rmse / 10.0,
                "fit_time": fit_time,
                "inference_time": 0.1,
                "total_train_rows": 100.0,
                "selected_n_partitions": selected_n_partitions,
                "chunk_size_imbalance_ratio": 1.5,
                "validation_mean_max_routing_proba": 0.8,
                "validation_mean_routing_entropy": 0.2,
                "validation_proxy_relative_gain": 0.3,
            }
        )

    comparison = RMTReportTableBuilder._build_partition_selection_comparison(
        pd.DataFrame(rows)
    )

    assert len(comparison) == 1
    assert comparison.loc[0, "rmse_balanced_silhouette"] == 10.0
    assert comparison.loc[0, "rmse_validation_proxy"] == 8.0
    assert comparison.loc[
        0,
        "rmse_delta_validation_proxy_minus_balanced_silhouette",
    ] == -2.0
    assert comparison.loc[
        0,
        "selected_n_partitions_delta_validation_proxy_minus_balanced_silhouette",
    ] == -1.0
    assert comparison.loc[0, "paired_run_count"] == 1
    assert bool(comparison.loc[0, "pair_complete"]) is True


def test_partition_selection_comparison_marks_incomplete_pairs() -> None:
    row = {
        "dataset": "synthetic",
        "model": "ridge",
        "sampler": "rmt_contraction",
        "ensemble_method": "routed_weighted",
        "router": "spectral",
        "view_strategy": "gaussian",
        "partition_selection_method": "auto",
        "cluster_ensemble_method": "coassociation",
        "budget_ratio": 0.1,
        "cluster_selection_metric": "validation_proxy",
        "rmse": 8.0,
        "rmse_drop": 0.8,
        "fit_time": 2.5,
        "inference_time": 0.1,
        "total_train_rows": 100.0,
        "selected_n_partitions": 3.0,
        "chunk_size_imbalance_ratio": 1.5,
        "validation_mean_max_routing_proba": 0.8,
        "validation_mean_routing_entropy": 0.2,
        "validation_proxy_relative_gain": 0.3,
    }

    comparison = RMTReportTableBuilder._build_partition_selection_comparison(
        pd.DataFrame([row])
    )

    assert len(comparison) == 1
    assert bool(comparison.loc[0, "pair_complete"]) is False
    assert comparison.loc[0, "paired_run_count"] == 0
    assert pd.isna(
        comparison.loc[
            0,
            "rmse_delta_validation_proxy_minus_balanced_silhouette",
        ]
    )
