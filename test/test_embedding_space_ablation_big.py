from __future__ import annotations

from dataclasses import replace

from examples.benchmark.embedding_space_ablation_big_datasets import (
    DEFAULT_BIG_DATASET_PLANS,
    DEFAULT_BIG_TASKS,
    EmbeddingSpaceAblationBigConfig,
    EmbeddingSpaceAblationBigOrchestrator,
    _select_task_plans,
)


def test_default_big_plan_runs_both_clusterers_and_tracks_historical_winner() -> None:
    plans = {plan.task: plan for plan in DEFAULT_BIG_DATASET_PLANS}

    assert set(plans) == set(DEFAULT_BIG_TASKS)
    assert all(
        plan.cluster_algorithms == ("kmeans", "gmm")
        for plan in plans.values()
    )
    assert plans["Yolanda"].historical_winner == "gmm"
    assert plans["Buzzinsocialmedia_Twitter"].historical_winner == "gmm"
    assert plans["black_friday"].historical_winner == "kmeans"
    assert max(max(plan.budgets) for plan in plans.values()) <= 0.5
    assert plans["Airlines_DepDelay_10M"].budgets == (0.01, 0.05, 0.1)
    assert plans["nyc-taxi-green-dec-2016"].budgets == (0.1, 0.3)


def test_big_task_grid_is_matched_for_each_representation() -> None:
    plan = next(
        plan for plan in DEFAULT_BIG_DATASET_PLANS if plan.task == "Yolanda"
    )
    config = EmbeddingSpaceAblationBigConfig(
        regression_tasks=(plan.task,),
        models=("lightgbm",),
        budget_ratios=plan.budgets,
        controlled_cluster_algorithms=plan.cluster_algorithms,
        task_plans=(plan,),
        show_progress=False,
    )
    grid = EmbeddingSpaceAblationBigOrchestrator(
        config
    )._strategy_grid_for_plan(plan)

    assert len(grid.strategies) == 12
    assert {
        tuple(point.config["cluster_algorithms"])
        for point in grid.strategies
    } == {("kmeans", "gmm")}
    assert {
        point.config["strategy"] for point in grid.strategies
    } == {"rmt_contraction", "raw_feature_clustering"}
    assert {point.config["budget_ratio"] for point in grid.strategies} == {
        0.05,
        0.1,
        0.2,
    }


def test_big_plan_override_is_explicit_and_task_scoped() -> None:
    plans = _select_task_plans(
        ("black_friday",),
        (0.02,),
        ("gmm",),
    )

    assert len(plans) == 1
    assert plans[0].task == "black_friday"
    assert plans[0].budgets == (0.02,)
    assert plans[0].cluster_algorithms == ("gmm",)
    assert plans[0].historical_winner == "gmm"


def test_big_protocol_expected_leaf_count_tracks_task_plans() -> None:
    plans = tuple(
        replace(plan, budgets=(0.1,))
        for plan in DEFAULT_BIG_DATASET_PLANS[:2]
    )
    config = EmbeddingSpaceAblationBigConfig(
        regression_tasks=tuple(plan.task for plan in plans),
        models=("tabpfn", "tabicl"),
        budget_ratios=(0.1,),
        controlled_cluster_algorithms=("gmm", "kmeans"),
        task_plans=plans,
        show_progress=False,
    )

    protocol = EmbeddingSpaceAblationBigOrchestrator(
        config
    )._comparison_protocol()

    assert protocol["expected_leaf_runs"] == 16
