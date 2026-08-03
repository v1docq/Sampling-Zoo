"""Pure construction of the final RMT regression experiment matrix."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Sequence

from examples.benchmark.benchmark_sampling_strategies import (
    make_chunking_strategy_configs,
)
from examples.benchmark.rmt_experiment_utils import budget_ratio_tag
from sampling_zoo.core.experiment.contracts import (
    ModelStrategyScenarioGridContract,
    ModelStrategyScenarioSpec,
    PartitionModelMode,
    StrategySpec,
)


class RMTFullGridScenarioGroup(str, Enum):
    LIGHTGBM_CONTROLS = "lightgbm_controls"
    TABPFN_IN_CONTEXT = "tabpfn_in_context"
    TABPFN_FINETUNED = "tabpfn_finetuned"


DEFAULT_FULL_GRID_SCENARIO_GROUPS: tuple[str, ...] = tuple(
    group.value for group in RMTFullGridScenarioGroup
)


@dataclass(frozen=True)
class _ScenarioFamily:
    key: str
    group: str
    model_name: str
    strategy: str
    ensemble_method: str
    partition_model_mode: str


def build_rmt_regression_full_grid(
    *,
    budget_ratios: Sequence[float],
    scenario_groups: Sequence[str] = DEFAULT_FULL_GRID_SCENARIO_GROUPS,
    n_partitions: int = 5,
    seed: int = 42,
) -> ModelStrategyScenarioGridContract:
    groups = _normalize_scenario_groups(scenario_groups)
    budgets = _normalize_budget_ratios(budget_ratios)
    scenarios: list[ModelStrategyScenarioSpec] = []

    if RMTFullGridScenarioGroup.LIGHTGBM_CONTROLS.value in groups:
        scenarios.append(_full_dataset_lightgbm_scenario())

    for family in _scenario_families():
        if family.group not in groups:
            continue
        for budget_ratio in budgets:
            scenarios.append(
                _budget_scenario(
                    family=family,
                    budget_ratio=budget_ratio,
                    n_partitions=n_partitions,
                    seed=seed,
                )
            )
    return ModelStrategyScenarioGridContract(scenarios=tuple(scenarios))


def model_names_for_scenario_groups(
    scenario_groups: Sequence[str],
) -> tuple[str, ...]:
    groups = _normalize_scenario_groups(scenario_groups)
    model_by_group = {
        RMTFullGridScenarioGroup.LIGHTGBM_CONTROLS.value: "lightgbm",
        RMTFullGridScenarioGroup.TABPFN_IN_CONTEXT.value: (
            "tabpfn_in_context"
        ),
        RMTFullGridScenarioGroup.TABPFN_FINETUNED.value: (
            "tabpfn_finetuned"
        ),
    }
    return tuple(model_by_group[group] for group in groups)


def _scenario_families() -> tuple[_ScenarioFamily, ...]:
    independent = PartitionModelMode.INDEPENDENT.value
    concatenated = PartitionModelMode.CONCATENATED.value
    return (
        _ScenarioFamily(
            "rmt_experts",
            "lightgbm_controls",
            "lightgbm",
            "rmt_contraction",
            "routed_weighted",
            independent,
        ),
        _ScenarioFamily(
            "rmt_concatenated",
            "lightgbm_controls",
            "lightgbm",
            "rmt_contraction",
            "voting",
            concatenated,
        ),
        _ScenarioFamily(
            "random_experts",
            "lightgbm_controls",
            "lightgbm",
            "random",
            "voting",
            independent,
        ),
        _ScenarioFamily(
            "random_concatenated",
            "lightgbm_controls",
            "lightgbm",
            "random",
            "voting",
            concatenated,
        ),
        _ScenarioFamily(
            "difficulty_experts",
            "lightgbm_controls",
            "lightgbm",
            "difficulty",
            "voting",
            independent,
        ),
        _ScenarioFamily(
            "difficulty_concatenated",
            "lightgbm_controls",
            "lightgbm",
            "difficulty",
            "voting",
            concatenated,
        ),
        *_tabpfn_families("tabpfn_in_context"),
        *_tabpfn_families("tabpfn_finetuned"),
    )


def _tabpfn_families(model_name: str) -> tuple[_ScenarioFamily, ...]:
    group = model_name
    return (
        _ScenarioFamily(
            "rmt_experts",
            group,
            model_name,
            "rmt_contraction",
            "routed_weighted",
            PartitionModelMode.INDEPENDENT.value,
        ),
        _ScenarioFamily(
            "rmt_concatenated",
            group,
            model_name,
            "rmt_contraction",
            "voting",
            PartitionModelMode.CONCATENATED.value,
        ),
        _ScenarioFamily(
            "random_concatenated",
            group,
            model_name,
            "random",
            "voting",
            PartitionModelMode.CONCATENATED.value,
        ),
        _ScenarioFamily(
            "difficulty_concatenated",
            group,
            model_name,
            "difficulty",
            "voting",
            PartitionModelMode.CONCATENATED.value,
        ),
    )


def _full_dataset_lightgbm_scenario() -> ModelStrategyScenarioSpec:
    name = "lightgbm__full_dataset"
    config = {
        "strategy": "full_dataset",
        "force_direct_model": True,
        "ensemble_method": "full_dataset",
        "budget_ratio": 1.0,
        "experiment_scenario": name,
        "scenario_family": "lightgbm__full_dataset",
        "scenario_group": "lightgbm_controls",
    }
    return ModelStrategyScenarioSpec(
        name=name,
        model_name="lightgbm",
        group="lightgbm_controls",
        strategy=StrategySpec(name=name, config=config, config_name=name),
    )


def _budget_scenario(
    *,
    family: _ScenarioFamily,
    budget_ratio: float,
    n_partitions: int,
    seed: int,
) -> ModelStrategyScenarioSpec:
    name = (
        f"{family.model_name}__{family.key}__"
        f"budget_{budget_ratio_tag(budget_ratio)}"
    )
    config = make_chunking_strategy_configs(
        problem_type="regression",
        strategy_names=(family.strategy,),
        n_partitions=n_partitions,
        seed=seed,
        ensemble_method=family.ensemble_method,
        budget_ratio=budget_ratio,
        force_chunking=True,
    )[family.strategy]
    config.update(
        {
            "partition_model_mode": family.partition_model_mode,
            "experiment_scenario": name,
            "scenario_family": f"{family.model_name}__{family.key}",
            "scenario_group": family.group,
        }
    )
    if family.strategy == "rmt_contraction":
        config.update(_winning_rmt_profile())
        if family.partition_model_mode == PartitionModelMode.INDEPENDENT.value:
            config["router"] = "spectral"

    return ModelStrategyScenarioSpec(
        name=name,
        model_name=family.model_name,
        group=family.group,
        strategy=StrategySpec(name=name, config=config, config_name=name),
    )


def _winning_rmt_profile() -> dict[str, Any]:
    return {
        "n_views": "auto",
        "n_views_policy": "auto",
        "view_strategy": "gaussian",
        "embedding_mode": "sv_scaled",
        "partition_selection_method": "auto",
        "cluster_selection_metric": "balanced_silhouette",
        "cluster_ensemble_method": "coassociation",
        "selection_method": "capped_leverage",
        "leverage_cap_quantile": 0.95,
    }


def _normalize_scenario_groups(values: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(dict.fromkeys(str(value).strip() for value in values))
    supported = {group.value for group in RMTFullGridScenarioGroup}
    unknown = sorted(set(normalized) - supported)
    if not normalized:
        raise ValueError("scenario_groups must not be empty")
    if unknown:
        raise ValueError(f"Unsupported scenario group(s): {unknown}")
    return normalized


def _normalize_budget_ratios(values: Sequence[float]) -> tuple[float, ...]:
    budgets = tuple(dict.fromkeys(float(value) for value in values))
    if not budgets:
        raise ValueError("budget_ratios must not be empty")
    invalid = [value for value in budgets if not 0 < value <= 1]
    if invalid:
        raise ValueError(f"budget_ratios must be in (0, 1]: {invalid}")
    return budgets
