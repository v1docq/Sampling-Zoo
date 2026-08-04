"""Pure construction of the staged RMT classification experiment matrix."""

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


class RMTClassificationScenarioGroup(str, Enum):
    LIGHTGBM_GATE = "lightgbm_gate"
    TABPFN_IN_CONTEXT = "tabpfn_in_context"


class ClassAllocationProfile(str, Enum):
    GLOBAL_CAPPED = "global_capped"
    STRATIFIED_CAPPED = "stratified_capped"

    @property
    def sampler_policy(self) -> str:
        if self is ClassAllocationProfile.STRATIFIED_CAPPED:
            return "proportional"
        return "minimum_then_global"


DEFAULT_CLASSIFICATION_SCENARIO_GROUPS: tuple[str, ...] = (
    RMTClassificationScenarioGroup.LIGHTGBM_GATE.value,
)


@dataclass(frozen=True)
class _ClassificationScenarioFamily:
    key: str
    group: str
    model_name: str
    strategy: str
    ensemble_method: str
    partition_model_mode: str
    allocation_profile: ClassAllocationProfile | None = None
    router: str | None = None


def build_rmt_classification_full_grid(
    *,
    budget_ratios: Sequence[float],
    scenario_groups: Sequence[str] = DEFAULT_CLASSIFICATION_SCENARIO_GROUPS,
    n_partitions: int = 5,
    seed: int = 42,
    min_samples_per_class: int = 1,
) -> ModelStrategyScenarioGridContract:
    """Build an explicit model/strategy grid without a Cartesian product."""

    groups = _normalize_scenario_groups(scenario_groups)
    budgets = _normalize_budget_ratios(budget_ratios)
    if int(min_samples_per_class) < 1:
        raise ValueError("min_samples_per_class must be positive")

    scenarios: list[ModelStrategyScenarioSpec] = []
    for group in groups:
        model_name = _model_name_for_group(group)
        scenarios.append(_full_dataset_scenario(model_name, group))
        for family in _scenario_families(group, model_name):
            for budget_ratio in budgets:
                scenarios.append(
                    _budget_scenario(
                        family=family,
                        budget_ratio=budget_ratio,
                        n_partitions=n_partitions,
                        seed=seed,
                        min_samples_per_class=min_samples_per_class,
                    )
                )
    return ModelStrategyScenarioGridContract(scenarios=tuple(scenarios))


def model_names_for_classification_groups(
    scenario_groups: Sequence[str],
) -> tuple[str, ...]:
    groups = _normalize_scenario_groups(scenario_groups)
    return tuple(_model_name_for_group(group) for group in groups)


def _scenario_families(
    group: str,
    model_name: str,
) -> tuple[_ClassificationScenarioFamily, ...]:
    concatenated = PartitionModelMode.CONCATENATED.value
    independent = PartitionModelMode.INDEPENDENT.value
    random_control = _ClassificationScenarioFamily(
        key="random_concatenated",
        group=group,
        model_name=model_name,
        strategy="random",
        ensemble_method="voting",
        partition_model_mode=concatenated,
    )
    if group == RMTClassificationScenarioGroup.TABPFN_IN_CONTEXT.value:
        allocation = ClassAllocationProfile.STRATIFIED_CAPPED
        return (
            random_control,
            _rmt_family(
                group,
                model_name,
                allocation,
                "concatenated",
                "voting",
                concatenated,
            ),
            _rmt_family(
                group,
                model_name,
                allocation,
                "spectral",
                "routed_weighted",
                independent,
                router="spectral",
            ),
            _rmt_family(
                group,
                model_name,
                allocation,
                "constrained_gating",
                "routed_weighted",
                independent,
                router="constrained_gating",
            ),
        )

    families: list[_ClassificationScenarioFamily] = [
        random_control,
        _ClassificationScenarioFamily(
            key="difficulty_concatenated",
            group=group,
            model_name=model_name,
            strategy="difficulty",
            ensemble_method="voting",
            partition_model_mode=concatenated,
        ),
    ]
    for allocation in ClassAllocationProfile:
        families.extend(
            [
                _rmt_family(
                    group,
                    model_name,
                    allocation,
                    "concatenated",
                    "voting",
                    concatenated,
                ),
                _rmt_family(
                    group,
                    model_name,
                    allocation,
                    "voting",
                    "voting",
                    independent,
                ),
                _rmt_family(
                    group,
                    model_name,
                    allocation,
                    "spectral",
                    "routed_weighted",
                    independent,
                    router="spectral",
                ),
                _rmt_family(
                    group,
                    model_name,
                    allocation,
                    "constrained_gating",
                    "routed_weighted",
                    independent,
                    router="constrained_gating",
                ),
            ]
        )
    return tuple(families)


def _rmt_family(
    group: str,
    model_name: str,
    allocation: ClassAllocationProfile,
    topology: str,
    ensemble_method: str,
    partition_model_mode: str,
    *,
    router: str | None = None,
) -> _ClassificationScenarioFamily:
    return _ClassificationScenarioFamily(
        key=f"rmt_{allocation.value}_{topology}",
        group=group,
        model_name=model_name,
        strategy="rmt_contraction",
        ensemble_method=ensemble_method,
        partition_model_mode=partition_model_mode,
        allocation_profile=allocation,
        router=router,
    )


def _full_dataset_scenario(
    model_name: str,
    group: str,
) -> ModelStrategyScenarioSpec:
    name = f"{model_name}__full_dataset"
    config = {
        "strategy": "full_dataset",
        "force_direct_model": True,
        "ensemble_method": "full_dataset",
        "budget_ratio": 1.0,
        "experiment_scenario": name,
        "scenario_family": f"{model_name}__full_dataset",
        "scenario_group": group,
    }
    return ModelStrategyScenarioSpec(
        name=name,
        model_name=model_name,
        group=group,
        strategy=StrategySpec(name=name, config=config, config_name=name),
    )


def _budget_scenario(
    *,
    family: _ClassificationScenarioFamily,
    budget_ratio: float,
    n_partitions: int,
    seed: int,
    min_samples_per_class: int,
) -> ModelStrategyScenarioSpec:
    name = (
        f"{family.model_name}__{family.key}__"
        f"budget_{budget_ratio_tag(budget_ratio)}"
    )
    config = make_chunking_strategy_configs(
        problem_type="classification",
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
        config.update(
            _classification_rmt_profile(
                allocation_profile=family.allocation_profile,
                min_samples_per_class=min_samples_per_class,
            )
        )
        if family.router is not None:
            config["router"] = family.router
        if family.router == "constrained_gating":
            config.update(_constrained_gating_profile())
    return ModelStrategyScenarioSpec(
        name=name,
        model_name=family.model_name,
        group=family.group,
        strategy=StrategySpec(name=name, config=config, config_name=name),
    )


def _classification_rmt_profile(
    *,
    allocation_profile: ClassAllocationProfile | None,
    min_samples_per_class: int,
) -> dict[str, Any]:
    if allocation_profile is None:
        raise ValueError("RMT classification family requires allocation_profile")
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
        "cluster_target_type": "classification",
        "class_coverage_policy": "preserve_local_classes",
        "class_allocation_policy": allocation_profile.sampler_policy,
        "min_samples_per_class": int(min_samples_per_class),
        "budget_feasibility_mode": "hard",
        "min_sampled_rows_per_partition": 3,
    }


def _constrained_gating_profile() -> dict[str, Any]:
    return {
        "gating_hidden_dim": 64,
        "gating_epochs": 200,
        "gating_lr": 1e-3,
        "gating_kl_weight": 0.10,
        "gating_balance_weight": 0.01,
        "gating_weight_decay": 1e-4,
        "gating_batch_size": 2048,
        "gating_device": "auto",
    }


def _model_name_for_group(group: str) -> str:
    if group == RMTClassificationScenarioGroup.LIGHTGBM_GATE.value:
        return "lightgbm"
    if group == RMTClassificationScenarioGroup.TABPFN_IN_CONTEXT.value:
        return "tabpfn_in_context"
    raise ValueError(f"Unsupported classification scenario group: {group}")


def _normalize_scenario_groups(values: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(dict.fromkeys(str(value).strip() for value in values))
    supported = {group.value for group in RMTClassificationScenarioGroup}
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
