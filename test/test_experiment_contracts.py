from __future__ import annotations

import pytest

from sampling_zoo.core.experiment.errors import EmptyExperimentInputError, InvalidExperimentConfigError
from sampling_zoo.core.experiment.contracts import (
    ModelStrategyScenarioGridContract,
    ModelStrategyScenarioSpec,
    PartitionSizeDiagnosticsContract,
    RuntimeDiagnosticsContract,
    StrategySpec,
)
from sampling_zoo.core.experiment.morphisms import (
    build_standard_rmt_experiment_plan,
    materialize_strategy_grid,
    normalize_strategy_grid,
)
from sampling_zoo.core.experiment.stages import ExperimentStageId


def test_strategy_grid_normalization_is_deterministic_and_idempotent() -> None:
    raw_configs = {
        "rmt__budget_10": {
            "strategy": "rmt_contraction",
            "ensemble_method": "routed_weighted",
            "budget_ratio": 0.1,
            "router": "constrained_gating",
            "view_strategy": "gaussian",
        },
        "random__budget_10": {
            "strategy": "random",
            "ensemble_method": "voting",
            "budget_ratio": 0.1,
        },
    }

    grid = normalize_strategy_grid(raw_configs)
    rematerialized = materialize_strategy_grid(grid)
    renormalized = normalize_strategy_grid(rematerialized)

    assert rematerialized == raw_configs
    assert renormalized.to_dict() == grid.to_dict()
    assert grid.strategies[0].router == "constrained_gating"
    assert grid.strategies[0].budget_ratio == 0.1


def test_strategy_grid_rejects_invalid_payloads() -> None:
    with pytest.raises(EmptyExperimentInputError):
        normalize_strategy_grid({})

    with pytest.raises(InvalidExperimentConfigError):
        normalize_strategy_grid({"broken": {"ensemble_method": "voting"}})


def test_model_strategy_scenario_grid_binds_models_without_cartesian_product() -> None:
    grid = ModelStrategyScenarioGridContract(
        scenarios=(
            ModelStrategyScenarioSpec(
                name="ridge__random",
                model_name="ridge",
                group="controls",
                strategy=StrategySpec(
                    name="random",
                    config={"strategy": "random", "budget_ratio": 0.1},
                ),
            ),
            ModelStrategyScenarioSpec(
                name="lightgbm__rmt",
                model_name="lightgbm",
                group="rmt",
                strategy=StrategySpec(
                    name="rmt",
                    config={
                        "strategy": "rmt_contraction",
                        "budget_ratio": 0.1,
                    },
                ),
            ),
        )
    )

    assert grid.model_names == ("ridge", "lightgbm")
    assert [item["name"] for item in grid.to_dict()["scenarios"]] == [
        "ridge__random",
        "lightgbm__rmt",
    ]

    with pytest.raises(ValueError, match="Duplicate scenario"):
        ModelStrategyScenarioGridContract(
            scenarios=(grid.scenarios[0], grid.scenarios[0])
        )


def test_standard_rmt_experiment_stage_plan_order_is_stable() -> None:
    plan = build_standard_rmt_experiment_plan({"models": ["lightgbm"], "seed": 42})

    assert plan.stage_ids() == (
        ExperimentStageId.PREPARE_RUNTIME,
        ExperimentStageId.CREATE_LOGGER,
        ExperimentStageId.CREATE_RUNNER,
        ExperimentStageId.LOAD_DATASETS,
        ExperimentStageId.BUILD_STRATEGY_GRID,
        ExperimentStageId.RUN_DATASETS,
        ExperimentStageId.BUILD_REPORTS,
        ExperimentStageId.WRITE_METADATA,
        ExperimentStageId.FINALIZE,
    )
    assert plan.effective_config["seed"] == 42


def test_partition_size_diagnostics_require_consistent_counts() -> None:
    contract = PartitionSizeDiagnosticsContract(
        pre_budget_sizes={"chunk_0": 70, "chunk_1": 30},
        post_budget_sizes={"chunk_0": 14, "chunk_1": 6},
        requested_budget_size=20,
        selected_rows=20,
        unique_selected_rows=20,
        duplicate_rows=0,
    )

    assert contract.to_dict()["post_budget_sizes"] == {
        "chunk_0": 14,
        "chunk_1": 6,
    }

    with pytest.raises(ValueError, match="must sum to selected_rows"):
        PartitionSizeDiagnosticsContract(
            pre_budget_sizes={"chunk_0": 10},
            post_budget_sizes={"chunk_0": 5},
            selected_rows=5,
            unique_selected_rows=4,
            duplicate_rows=0,
        )


def test_runtime_diagnostics_require_finite_non_negative_values() -> None:
    contract = RuntimeDiagnosticsContract(
        stage_seconds={"preprocessing": 0.25, "training": 0.75},
        total_seconds=1.1,
        cold_start=False,
    )

    assert contract.to_dict()["stage_seconds"]["training"] == 0.75

    with pytest.raises(ValueError, match="finite and non-negative"):
        RuntimeDiagnosticsContract(
            stage_seconds={"preprocessing": float("nan")},
            total_seconds=1.0,
        )
