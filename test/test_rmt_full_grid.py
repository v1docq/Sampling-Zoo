from __future__ import annotations

import pytest

from examples.benchmark.rmt_full_grid_plan import (
    DEFAULT_FULL_GRID_SCENARIO_GROUPS,
    build_rmt_regression_full_grid,
)
from examples.benchmark.rmt_regression_full_grid import (
    RMTFullGridExperimentConfig,
    RMTFullGridExperimentOrchestrator,
)


def test_full_grid_is_explicit_and_has_expected_scenario_count() -> None:
    grid = build_rmt_regression_full_grid(
        budget_ratios=(0.01, 0.05, 0.10, 0.20),
        scenario_groups=DEFAULT_FULL_GRID_SCENARIO_GROUPS,
    )

    assert len(grid.scenarios) == 57
    assert grid.model_names == (
        "lightgbm",
        "tabpfn_in_context",
        "tabpfn_finetuned",
    )
    assert grid.scenarios[0].name == "lightgbm__full_dataset"
    assert len({scenario.name for scenario in grid.scenarios}) == 57


def test_full_grid_pins_winning_rmt_profile_and_training_modes() -> None:
    grid = build_rmt_regression_full_grid(
        budget_ratios=(0.05,),
        scenario_groups=("lightgbm_controls",),
    )
    by_name = {scenario.name: scenario for scenario in grid.scenarios}

    experts = by_name["lightgbm__rmt_experts__budget_05"]
    experts_config = experts.strategy.materialize()
    assert experts_config["selection_method"] == "capped_leverage"
    assert experts_config["leverage_cap_quantile"] == 0.95
    assert experts_config["embedding_mode"] == "sv_scaled"
    assert experts_config["view_strategy"] == "gaussian"
    assert experts_config["cluster_selection_metric"] == (
        "balanced_silhouette"
    )
    assert experts_config["router"] == "spectral"
    assert experts_config["ensemble_method"] == "routed_weighted"
    assert experts_config["partition_model_mode"] == "independent"

    concatenated = by_name["lightgbm__rmt_concatenated__budget_05"]
    concatenated_config = concatenated.strategy.materialize()
    assert concatenated_config["partition_model_mode"] == "concatenated"
    assert concatenated_config["ensemble_method"] == "voting"
    assert "router" not in concatenated_config


def test_tabpfn_groups_do_not_expand_to_unrequested_lightgbm_scenarios() -> None:
    grid = build_rmt_regression_full_grid(
        budget_ratios=(0.10,),
        scenario_groups=("tabpfn_finetuned",),
    )

    assert len(grid.scenarios) == 4
    assert grid.model_names == ("tabpfn_finetuned",)
    assert {scenario.strategy.strategy for scenario in grid.scenarios} == {
        "rmt_contraction",
        "random",
        "difficulty",
    }


def test_full_grid_config_derives_model_pool_from_scenario_groups() -> None:
    config = RMTFullGridExperimentConfig(
        scenario_groups=("lightgbm_controls",),
        budget_ratios=(0.05,),
        show_progress=False,
        synthetic_smoke=True,
    )
    orchestrator = RMTFullGridExperimentOrchestrator(config)
    plan = orchestrator._build_experiment_plan()

    assert config.models == ("lightgbm",)
    assert len(orchestrator._build_strategy_grid().scenarios) == 7
    assert len(plan.effective_config["scenario_grid"]["scenarios"]) == 7


@pytest.mark.parametrize(
    "kwargs",
    [
        {"scenario_groups": ()},
        {"scenario_groups": ("unknown",)},
        {"budget_ratios": ()},
        {"budget_ratios": (0.0,)},
    ],
)
def test_full_grid_rejects_invalid_axes(kwargs) -> None:
    complete_kwargs = {
        "budget_ratios": (0.05,),
        "scenario_groups": ("lightgbm_controls",),
        **kwargs,
    }
    with pytest.raises(ValueError):
        build_rmt_regression_full_grid(**complete_kwargs)
