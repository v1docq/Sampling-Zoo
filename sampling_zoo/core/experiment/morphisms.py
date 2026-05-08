"""Pure transformations between raw benchmark objects and experiment contracts."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import (
    ChunkModelContract,
    DatasetContract,
    EvaluationContract,
    FoldContract,
    ModelSpec,
    PartitionContract,
    RoutingContract,
    StrategyGridContract,
    StrategySpec,
)
from .errors import EmptyExperimentInputError, InvalidExperimentConfigError
from .stages import ExperimentPlan, ExperimentStageId, StageRequest


def normalize_strategy_grid(strategy_configs: Mapping[str, Mapping[str, Any]]) -> StrategyGridContract:
    if not strategy_configs:
        raise EmptyExperimentInputError(
            scope="strategy_grid",
            message="Strategy config mapping must not be empty.",
            code="empty_strategy_grid",
        )
    specs = []
    for config_name, raw_config in strategy_configs.items():
        if not isinstance(raw_config, Mapping):
            raise InvalidExperimentConfigError(
                scope=f"strategy_grid.{config_name}",
                message="Strategy config must be a mapping.",
                code="invalid_strategy_config",
            )
        strategy = raw_config.get("strategy")
        if not isinstance(strategy, str) or not strategy.strip():
            raise InvalidExperimentConfigError(
                scope=f"strategy_grid.{config_name}",
                message="Strategy config requires a non-empty 'strategy' value.",
                code="missing_strategy_name",
            )
        specs.append(StrategySpec(name=str(config_name), config=dict(raw_config), config_name=str(config_name)))
    return StrategyGridContract(strategies=tuple(specs))


def materialize_strategy_grid(strategy_grid: StrategyGridContract | Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    if isinstance(strategy_grid, StrategyGridContract):
        return strategy_grid.materialize()
    return normalize_strategy_grid(strategy_grid).materialize()


def dataset_to_contract(dataset: Any) -> DatasetContract:
    n_rows = _safe_len(getattr(dataset, "X", None))
    n_features = getattr(getattr(dataset, "X", None), "shape", (None, None))[1]
    return DatasetContract(
        name=str(getattr(dataset, "name", "dataset")),
        problem_type=str(getattr(dataset, "problem_type", "")),
        source_path=getattr(dataset, "source_path", None),
        task_id=getattr(dataset, "task_id", None),
        task_name=getattr(dataset, "task_name", None),
        suite_id=getattr(dataset, "suite_id", None),
        n_rows=n_rows,
        n_features=None if n_features is None else int(n_features),
    )


def model_pool_to_contracts(model_pool: Mapping[str, Any]) -> tuple[ModelSpec, ...]:
    return tuple(ModelSpec(name=str(name)) for name in model_pool)


def fold_to_contract(fold: Any) -> FoldContract:
    return FoldContract(
        fold_idx=getattr(fold, "fold_idx", None) if str(getattr(fold, "split_label", "")).startswith("fold_") else None,
        split_label=str(getattr(fold, "split_label", "split")),
        n_train=_safe_len(getattr(fold, "X_train", None)),
        n_val=_safe_len(getattr(fold, "X_val", None)),
        n_test=_safe_len(getattr(fold, "X_test", None)),
    )


def partitions_to_contract(partitions: Mapping[str, Any], diagnostics: Mapping[str, Any] | None = None) -> PartitionContract:
    sizes = {str(name): int(_partition_size(partition)) for name, partition in partitions.items()}
    return PartitionContract(names=tuple(sizes.keys()), sizes=sizes, diagnostics=dict(diagnostics or {}))


def chunk_models_to_contracts(models: Sequence[Mapping[str, Any]]) -> tuple[ChunkModelContract, ...]:
    return tuple(
        ChunkModelContract(
            name=str(model_info.get("name", f"chunk_{idx}")),
            data_size=int(model_info.get("data_size", 0) or 0),
            metrics=dict(model_info.get("metrics", {}) or {}),
        )
        for idx, model_info in enumerate(models)
    )


def routing_to_contract(router: Any) -> RoutingContract:
    diagnostics = dict(getattr(router, "diagnostics_", {}) or {})
    return RoutingContract(
        mode=str(getattr(router, "router_mode", diagnostics.get("router_mode", "spectral"))),
        status=diagnostics.get("status"),
        diagnostics=diagnostics,
    )


def evaluation_to_contract(
    *,
    metrics: Mapping[str, Any],
    timings: Mapping[str, float] | None = None,
    sample_stats: Mapping[str, Any] | None = None,
) -> EvaluationContract:
    return EvaluationContract(metrics=dict(metrics), timings=dict(timings or {}), sample_stats=dict(sample_stats or {}))


def build_standard_rmt_experiment_plan(effective_config: Mapping[str, Any]) -> ExperimentPlan:
    return ExperimentPlan(
        stage_requests=(
            StageRequest(ExperimentStageId.PREPARE_RUNTIME),
            StageRequest(ExperimentStageId.CREATE_LOGGER),
            StageRequest(ExperimentStageId.CREATE_RUNNER),
            StageRequest(ExperimentStageId.LOAD_DATASETS),
            StageRequest(ExperimentStageId.BUILD_STRATEGY_GRID),
            StageRequest(ExperimentStageId.RUN_DATASETS),
            StageRequest(ExperimentStageId.BUILD_REPORTS),
            StageRequest(ExperimentStageId.WRITE_METADATA),
            StageRequest(ExperimentStageId.FINALIZE),
        ),
        effective_config=dict(effective_config),
    )


def _safe_len(value: Any) -> int | None:
    try:
        return int(len(value))
    except Exception:
        return None


def _partition_size(partition: Any) -> int:
    if isinstance(partition, Mapping) and "feature" in partition:
        return int(len(partition["feature"]))
    return int(len(partition))
