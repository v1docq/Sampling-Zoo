"""Immutable contracts exchanged between Sampling Zoo experiment stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


class _StrEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class ProblemType(_StrEnum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class EnsembleMethod(_StrEnum):
    VOTING = "voting"
    WEIGHTED = "weighted"
    ROUTED_WEIGHTED = "routed_weighted"
    FULL_DATASET = "full_dataset"


class RoutingMode(_StrEnum):
    SPECTRAL = "spectral"
    LEARNED_HEAD = "learned_head"
    CONSTRAINED_GATING = "constrained_gating"


class RoutingRefinementMode(_StrEnum):
    NONE = "none"
    EM_RETRAINING = "em_retraining"


class PartitionModelMode(_StrEnum):
    INDEPENDENT = "independent"
    CONCATENATED = "concatenated"


@dataclass(frozen=True)
class DatasetContract:
    name: str
    problem_type: str
    source_path: str | None = None
    task_id: int | None = None
    task_name: str | None = None
    suite_id: int | None = None
    n_rows: int | None = None
    n_features: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return _drop_none(self.__dict__)


@dataclass(frozen=True)
class StrategySpec:
    name: str
    config: Mapping[str, Any] = field(default_factory=dict)
    config_name: str | None = None

    @property
    def strategy(self) -> str:
        return str(self.config.get("strategy", self.name))

    @property
    def ensemble_method(self) -> str:
        return str(self.config.get("ensemble_method", "voting"))

    @property
    def budget_ratio(self) -> float | None:
        raw = self.config.get("budget_ratio")
        return None if raw is None else float(raw)

    @property
    def router(self) -> str | None:
        raw = self.config.get("router")
        return None if raw is None else str(raw)

    @property
    def view_strategy(self) -> str | None:
        raw = self.config.get("view_strategy")
        return None if raw is None else str(raw)

    def materialize(self) -> dict[str, Any]:
        return dict(self.config)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "config_name": self.config_name or self.name,
            "strategy": self.strategy,
            "ensemble_method": self.ensemble_method,
            "budget_ratio": self.budget_ratio,
            "router": self.router,
            "view_strategy": self.view_strategy,
            "config": dict(self.config),
        }


@dataclass(frozen=True)
class StrategyGridContract:
    strategies: tuple[StrategySpec, ...]

    def __post_init__(self) -> None:
        names = [spec.config_name or spec.name for spec in self.strategies]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"Duplicate strategy config names: {duplicates}")

    def materialize(self) -> dict[str, dict[str, Any]]:
        return {
            spec.config_name or spec.name: spec.materialize()
            for spec in self.strategies
        }

    def to_dict(self) -> dict[str, Any]:
        return {"strategies": [spec.to_dict() for spec in self.strategies]}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    params: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "params": dict(self.params)}


@dataclass(frozen=True)
class FoldContract:
    fold_idx: int | None
    split_label: str
    n_train: int
    n_val: int
    n_test: int

    def to_dict(self) -> dict[str, Any]:
        return _drop_none(self.__dict__)


@dataclass(frozen=True)
class PartitionContract:
    names: tuple[str, ...]
    sizes: Mapping[str, int]
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    @property
    def n_partitions(self) -> int:
        return len(self.names)

    def to_dict(self) -> dict[str, Any]:
        return {
            "names": list(self.names),
            "sizes": dict(self.sizes),
            "n_partitions": self.n_partitions,
            "diagnostics": dict(self.diagnostics),
        }


@dataclass(frozen=True)
class PartitionSizeDiagnosticsContract:
    pre_budget_sizes: Mapping[str, int]
    post_budget_sizes: Mapping[str, int]
    requested_budget_size: int | None = None
    selected_rows: int | None = None
    unique_selected_rows: int | None = None
    duplicate_rows: int | None = None

    def __post_init__(self) -> None:
        sizes = [
            *map(int, self.pre_budget_sizes.values()),
            *map(int, self.post_budget_sizes.values()),
        ]
        optional_counts = (
            self.requested_budget_size,
            self.selected_rows,
            self.unique_selected_rows,
            self.duplicate_rows,
        )
        sizes.extend(int(value) for value in optional_counts if value is not None)
        if any(value < 0 for value in sizes):
            raise ValueError("partition size diagnostics must be non-negative")
        if (
            self.selected_rows is not None
            and self.unique_selected_rows is not None
            and self.unique_selected_rows > self.selected_rows
        ):
            raise ValueError("unique_selected_rows cannot exceed selected_rows")
        if (
            self.selected_rows is not None
            and self.duplicate_rows is not None
            and self.unique_selected_rows is not None
            and self.unique_selected_rows + self.duplicate_rows != self.selected_rows
        ):
            raise ValueError(
                "unique_selected_rows and duplicate_rows must sum to selected_rows"
            )

    def to_dict(self) -> dict[str, Any]:
        return _drop_none(
            {
                "pre_budget_sizes": {
                    str(name): int(size)
                    for name, size in self.pre_budget_sizes.items()
                },
                "post_budget_sizes": {
                    str(name): int(size)
                    for name, size in self.post_budget_sizes.items()
                },
                "requested_budget_size": self.requested_budget_size,
                "selected_rows": self.selected_rows,
                "unique_selected_rows": self.unique_selected_rows,
                "duplicate_rows": self.duplicate_rows,
            }
        )


@dataclass(frozen=True)
class RuntimeDiagnosticsContract:
    stage_seconds: Mapping[str, float]
    total_seconds: float
    cold_start: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = [float(self.total_seconds), *map(float, self.stage_seconds.values())]
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("runtime diagnostics must be finite and non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_seconds": {
                str(name): float(value)
                for name, value in self.stage_seconds.items()
            },
            "total_seconds": float(self.total_seconds),
            "cold_start": bool(self.cold_start),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ChunkModelContract:
    name: str
    data_size: int
    metrics: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "data_size": int(self.data_size), "metrics": dict(self.metrics)}


@dataclass(frozen=True)
class RoutingContract:
    mode: str
    status: str | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _drop_none({"mode": self.mode, "status": self.status, "diagnostics": dict(self.diagnostics)})


@dataclass(frozen=True)
class EvaluationContract:
    metrics: Mapping[str, Any]
    timings: Mapping[str, float] = field(default_factory=dict)
    sample_stats: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metrics": dict(self.metrics),
            "timings": dict(self.timings),
            "sample_stats": dict(self.sample_stats),
        }


@dataclass(frozen=True)
class RunRecordContract:
    dataset: DatasetContract
    strategy: StrategySpec
    model: ModelSpec
    fold: FoldContract
    evaluation: EvaluationContract
    extra: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset.to_dict(),
            "strategy": self.strategy.to_dict(),
            "model": self.model.to_dict(),
            "fold": self.fold.to_dict(),
            "evaluation": self.evaluation.to_dict(),
            "extra": dict(self.extra),
        }


@dataclass(frozen=True)
class ArtifactContract:
    root: Path
    paths: Mapping[str, Path] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "paths": {key: str(value) for key, value in self.paths.items()},
        }


@dataclass(frozen=True)
class PartitionTrainingRequest:
    problem: str
    ensemble_method: str
    validation_metric: str
    n_partitions: int
    routing_refinement: str = RoutingRefinementMode.NONE.value
    partition_model_mode: str = PartitionModelMode.INDEPENDENT.value
    n_training_partitions: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class PartitionTrainingResult:
    request: PartitionTrainingRequest
    partitions: PartitionContract
    chunk_models: tuple[ChunkModelContract, ...]
    routing: RoutingContract
    validation_diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request": self.request.to_dict(),
            "partitions": self.partitions.to_dict(),
            "chunk_models": [model.to_dict() for model in self.chunk_models],
            "routing": self.routing.to_dict(),
            "validation_diagnostics": dict(self.validation_diagnostics),
        }


def _drop_none(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if value is not None}
