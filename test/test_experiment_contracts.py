from __future__ import annotations

import pytest

from sampling_zoo.core.experiment.errors import EmptyExperimentInputError, InvalidExperimentConfigError
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
