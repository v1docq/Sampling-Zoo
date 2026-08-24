"""Stage plan primitives for Sampling Zoo experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping

from .errors import ExperimentContractError, InvalidExperimentConfigError


class ExperimentStageId(str, Enum):
    PREPARE_RUNTIME = "prepare_runtime"
    CREATE_LOGGER = "create_logger"
    CREATE_RUNNER = "create_runner"
    LOAD_DATASETS = "load_datasets"
    BUILD_STRATEGY_GRID = "build_strategy_grid"
    RUN_DATASETS = "run_datasets"
    BUILD_REPORTS = "build_reports"
    WRITE_METADATA = "write_metadata"
    FINALIZE = "finalize"


@dataclass(frozen=True)
class StageRequest:
    stage_id: ExperimentStageId
    payload: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {"stage_id": self.stage_id.value, "payload": self.payload}


@dataclass(frozen=True)
class StageResult:
    stage_id: ExperimentStageId
    status: str
    payload: Any = None
    error: ExperimentContractError | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_id": self.stage_id.value,
            "status": self.status,
            "payload": self.payload,
            "error": None if self.error is None else self.error.to_dict(),
        }


@dataclass(frozen=True)
class ExperimentPlan:
    stage_requests: tuple[StageRequest, ...]
    effective_config: Mapping[str, Any] = field(default_factory=dict)

    def stage_ids(self) -> tuple[ExperimentStageId, ...]:
        return tuple(request.stage_id for request in self.stage_requests)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_requests": [request.to_dict() for request in self.stage_requests],
            "effective_config": dict(self.effective_config),
        }


class StageRegistry:
    """Small callable registry for effectful experiment stage execution."""

    def __init__(self, handlers: Mapping[ExperimentStageId, Callable[[Any], Any]] | None = None) -> None:
        self._handlers = dict(handlers or {})

    def register(self, stage_id: ExperimentStageId, handler: Callable[[Any], Any]) -> "StageRegistry":
        self._handlers[stage_id] = handler
        return self

    def execute(self, request: StageRequest) -> StageResult:
        handler = self._handlers.get(request.stage_id)
        if handler is None:
            return StageResult(
                stage_id=request.stage_id,
                status="failed",
                error=InvalidExperimentConfigError(
                    scope="experiment.stage_registry",
                    message=f"No handler registered for stage '{request.stage_id.value}'",
                    code="missing_stage_handler",
                ),
            )
        try:
            return StageResult(stage_id=request.stage_id, status="completed", payload=handler(request.payload))
        except ExperimentContractError as exc:
            return StageResult(stage_id=request.stage_id, status="failed", error=exc)

    def execute_plan(self, plan: ExperimentPlan) -> tuple[StageResult, ...]:
        return tuple(self.execute(request) for request in plan.stage_requests)
