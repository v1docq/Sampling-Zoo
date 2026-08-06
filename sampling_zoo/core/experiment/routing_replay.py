"""Offline evaluation of routing geometry over cached expert predictions."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import mean_absolute_error

from sampling_zoo.core.metrics.eval_metrics import (
    calculate_metrics,
    get_metric_comparator,
    metric_drop,
)
from sampling_zoo.core.sampling_strategies.spectral.backend.matrix_backend import MatrixRMTBackend
from sampling_zoo.core.sampling_strategies.spectral.routing_contracts import (
    PartitionGeometrySpec,
    RoutingDistanceContract,
    RoutingWeightContract,
)


def _readonly(value: Any) -> np.ndarray:
    array = np.array(value, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class RoutingReplayRequest:
    """Cached validation/test state needed to replay routing without model fitting."""

    arm_name: str
    geometry_spec: PartitionGeometrySpec
    distances: RoutingDistanceContract
    target: np.ndarray
    expert_outputs: np.ndarray
    problem_type: str
    classes: tuple[Any, ...] = ()
    expert_priors: Optional[np.ndarray] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        target = _readonly(self.target).reshape(-1)
        outputs = _readonly(self.expert_outputs)
        if self.problem_type not in {"regression", "classification"}:
            raise ValueError("problem_type must be regression or classification")
        expected_ndim = 2 if self.problem_type == "regression" else 3
        if outputs.ndim != expected_ndim:
            raise ValueError(
                f"{self.problem_type} expert_outputs must be {expected_ndim}D, "
                f"got shape {outputs.shape}"
            )
        if outputs.shape[0] != target.shape[0]:
            raise ValueError("expert_outputs and target must contain the same rows")
        if outputs.shape[1] != len(self.distances.partition_names):
            raise ValueError("expert_outputs must align with routing partitions")
        if not np.all(np.isfinite(outputs)):
            raise ValueError("expert_outputs must be finite")
        classes = tuple(self.classes)
        if self.problem_type == "classification":
            if len(classes) != outputs.shape[2]:
                raise ValueError("classes must align with classification probability columns")
            if np.any(outputs < 0):
                raise ValueError("classification expert probabilities must be finite and non-negative")
            if not np.allclose(np.sum(outputs, axis=2), 1.0, atol=1e-6):
                raise ValueError("classification expert probabilities must sum to one")

        priors = None
        if self.expert_priors is not None:
            priors = _readonly(self.expert_priors).astype(float).reshape(-1)
            if priors.shape != (outputs.shape[1],):
                raise ValueError("expert_priors must align with routing partitions")
            if np.any(priors < 0) or not np.all(np.isfinite(priors)) or priors.sum() <= 0:
                raise ValueError("expert_priors must be finite, non-negative, and have positive mass")

        object.__setattr__(self, "target", target)
        object.__setattr__(self, "expert_outputs", outputs)
        object.__setattr__(self, "classes", classes)
        object.__setattr__(self, "expert_priors", priors)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True)
class RoutingReplayResult:
    arm_name: str
    temperature: float
    primary_metric: str
    primary_value: float
    metrics: Mapping[str, float]
    routing: RoutingWeightContract
    blended_output: np.ndarray
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))
        object.__setattr__(self, "blended_output", _readonly(self.blended_output))
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))

    def summary(self) -> dict[str, Any]:
        return {
            "arm_name": self.arm_name,
            "temperature": float(self.temperature),
            "primary_metric": self.primary_metric,
            "primary_value": float(self.primary_value),
            "metrics": dict(self.metrics),
            "routing": dict(self.routing.diagnostics),
            "diagnostics": dict(self.diagnostics),
        }


@dataclass(frozen=True)
class RoutingTemperatureSelection:
    best: RoutingReplayResult
    candidates: tuple[RoutingReplayResult, ...]


@dataclass(frozen=True)
class RoutingGeometrySelectionPolicy:
    """Validated policy for selecting one calibrated geometry on validation rows."""

    candidate_arm_names: tuple[str, ...]
    fallback_arm_name: str
    minimum_improvement: float = 0.0

    def __post_init__(self) -> None:
        names = tuple(str(name).strip() for name in self.candidate_arm_names)
        fallback = str(self.fallback_arm_name).strip()
        if not names or any(not name for name in names):
            raise ValueError("candidate_arm_names must be non-empty")
        if len(set(names)) != len(names):
            raise ValueError("candidate_arm_names must be unique")
        if fallback not in names:
            raise ValueError("fallback_arm_name must be one of candidate_arm_names")
        if float(self.minimum_improvement) < 0:
            raise ValueError("minimum_improvement must be non-negative")
        object.__setattr__(self, "candidate_arm_names", names)
        object.__setattr__(self, "fallback_arm_name", fallback)
        object.__setattr__(self, "minimum_improvement", float(self.minimum_improvement))

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_arm_names": list(self.candidate_arm_names),
            "fallback_arm_name": self.fallback_arm_name,
            "minimum_improvement": self.minimum_improvement,
        }


@dataclass(frozen=True)
class RoutingGeometrySelection:
    """Immutable validation decision and the evidence used to make it."""

    selected: RoutingReplayResult
    candidates: tuple[RoutingReplayResult, ...]
    policy: RoutingGeometrySelectionPolicy
    fallback_used: bool
    reason: str
    improvement_vs_fallback: float
    candidate_scores: Mapping[str, float]

    def __post_init__(self) -> None:
        scores = {str(name): float(value) for name, value in self.candidate_scores.items()}
        object.__setattr__(self, "candidate_scores", MappingProxyType(scores))

    def summary(self) -> dict[str, Any]:
        return {
            "selected_arm_name": self.selected.arm_name,
            "primary_metric": self.selected.primary_metric,
            "selected_score": float(self.selected.primary_value),
            "fallback_arm_name": self.policy.fallback_arm_name,
            "fallback_used": bool(self.fallback_used),
            "reason": self.reason,
            "improvement_vs_fallback": float(self.improvement_vs_fallback),
            "candidate_scores": dict(self.candidate_scores),
            "policy": self.policy.to_dict(),
        }


class ValidationRoutingGeometrySelector:
    """Select a calibrated routing geometry using validation metrics only."""

    def select(
        self,
        results: Sequence[RoutingReplayResult],
        policy: RoutingGeometrySelectionPolicy,
    ) -> RoutingGeometrySelection:
        by_name = self._index_results(results)
        missing = [name for name in policy.candidate_arm_names if name not in by_name]
        if missing:
            raise ValueError(f"Missing validation results for routing arms: {missing}")

        candidates = tuple(by_name[name] for name in policy.candidate_arm_names)
        metrics = {result.primary_metric for result in candidates}
        if len(metrics) != 1:
            raise ValueError("Routing geometry candidates must use the same primary metric")
        primary_metric = candidates[0].primary_metric
        finite = [result for result in candidates if np.isfinite(result.primary_value)]
        if not finite:
            raise ValueError("No finite validation metric is available for geometry selection")

        fallback = by_name[policy.fallback_arm_name]
        comparator = get_metric_comparator(primary_metric)
        best = fallback if np.isfinite(fallback.primary_value) else finite[0]
        for candidate in finite:
            if comparator(candidate.primary_value, best.primary_value):
                best = candidate

        fallback_finite = np.isfinite(fallback.primary_value)
        improvement = (
            -metric_drop(primary_metric, best.primary_value, fallback.primary_value)
            if fallback_finite
            else float("inf")
        )
        if (
            best.arm_name != fallback.arm_name
            and fallback_finite
            and improvement <= policy.minimum_improvement
        ):
            best = fallback
            improvement = 0.0
            reason = "fallback_minimum_improvement"
        elif best.arm_name == fallback.arm_name:
            improvement = 0.0
            reason = "fallback_best_or_tied"
        elif not fallback_finite:
            reason = "fallback_non_finite"
        else:
            reason = "validation_metric_improvement"

        return RoutingGeometrySelection(
            selected=best,
            candidates=candidates,
            policy=policy,
            fallback_used=best.arm_name == fallback.arm_name,
            reason=reason,
            improvement_vs_fallback=float(improvement),
            candidate_scores={
                result.arm_name: float(result.primary_value) for result in candidates
            },
        )

    @staticmethod
    def _index_results(
        results: Sequence[RoutingReplayResult],
    ) -> dict[str, RoutingReplayResult]:
        indexed: dict[str, RoutingReplayResult] = {}
        for result in results:
            if result.arm_name in indexed:
                raise ValueError(f"Duplicate validation result for arm {result.arm_name!r}")
            indexed[result.arm_name] = result
        return indexed


class RoutingReplayEvaluator:
    """Pure evaluator and temperature selector for cached chunk-model outputs."""

    def __init__(self, backend: Optional[MatrixRMTBackend] = None) -> None:
        self.backend = backend or MatrixRMTBackend()

    def evaluate(
        self,
        request: RoutingReplayRequest,
        *,
        temperature: Optional[float] = None,
    ) -> RoutingReplayResult:
        selected_temperature = (
            float(request.geometry_spec.temperature)
            if temperature is None
            else float(temperature)
        )
        weights = self.backend.normalize_routing_values(
            request.distances.values,
            kind=str(request.distances.kind),
            temperature=selected_temperature,
        )
        weights = self._apply_uniform_shrinkage(
            weights,
            float(request.geometry_spec.uniform_shrinkage),
        )
        weights = self._apply_expert_priors(weights, request.expert_priors)
        routing = RoutingWeightContract(
            partition_names=request.distances.partition_names,
            weights=weights,
            temperature=selected_temperature,
            diagnostics=self._routing_diagnostics(weights),
        )
        blended = self._blend(weights, request.expert_outputs, request.problem_type)
        metrics, primary_metric = self._evaluate_output(request, blended)
        diagnostics = self._expert_alignment_diagnostics(request, weights, blended)
        return RoutingReplayResult(
            arm_name=request.arm_name,
            temperature=selected_temperature,
            primary_metric=primary_metric,
            primary_value=float(metrics[primary_metric]),
            metrics=metrics,
            routing=routing,
            blended_output=blended,
            diagnostics=diagnostics,
        )

    def calibrate_temperature(
        self,
        request: RoutingReplayRequest,
        candidates: Sequence[float],
    ) -> RoutingTemperatureSelection:
        temperatures = tuple(float(value) for value in candidates)
        if not temperatures or any(value <= 0 for value in temperatures):
            raise ValueError("Temperature candidates must be positive and non-empty")
        results = tuple(self.evaluate(request, temperature=value) for value in temperatures)
        finite_results = [result for result in results if np.isfinite(result.primary_value)]
        if not finite_results:
            raise ValueError("No finite primary metric was produced during temperature calibration")
        comparator = get_metric_comparator(results[0].primary_metric)
        best = finite_results[0]
        for candidate in finite_results[1:]:
            if comparator(candidate.primary_value, best.primary_value):
                best = candidate
        return RoutingTemperatureSelection(best=best, candidates=results)

    @staticmethod
    def _apply_uniform_shrinkage(weights: np.ndarray, shrinkage: float) -> np.ndarray:
        if not 0.0 <= shrinkage <= 1.0:
            raise ValueError("uniform_shrinkage must be in [0, 1]")
        if shrinkage <= 0:
            return weights
        result = (1.0 - shrinkage) * weights + shrinkage / weights.shape[1]
        return result / np.sum(result, axis=1, keepdims=True)

    @staticmethod
    def _apply_expert_priors(
        weights: np.ndarray,
        priors: Optional[np.ndarray],
    ) -> np.ndarray:
        if priors is None:
            return weights
        combined = weights * np.asarray(priors, dtype=float)[None, :]
        row_sums = np.sum(combined, axis=1, keepdims=True)
        return np.divide(
            combined,
            row_sums,
            out=np.full_like(combined, 1.0 / combined.shape[1]),
            where=row_sums > 0,
        )

    @staticmethod
    def _blend(weights: np.ndarray, outputs: np.ndarray, problem_type: str) -> np.ndarray:
        if problem_type == "regression":
            return np.sum(weights * outputs, axis=1)
        probabilities = np.sum(weights[:, :, None] * outputs, axis=1)
        probabilities = np.clip(probabilities, 1e-15, 1.0)
        return probabilities / np.sum(probabilities, axis=1, keepdims=True)

    @staticmethod
    def _evaluate_output(
        request: RoutingReplayRequest,
        blended: np.ndarray,
    ) -> tuple[dict[str, float], str]:
        if request.problem_type == "regression":
            metrics = calculate_metrics(
                y_true=request.target,
                y_labels=blended,
                y_proba=None,
                problem_type="regression",
            )
            metrics["mae"] = float(mean_absolute_error(request.target, blended))
            return {name: float(value) for name, value in metrics.items()}, "rmse"

        classes = np.asarray(request.classes)
        labels = classes[np.argmax(blended, axis=1)]
        metrics = calculate_metrics(
            y_true=request.target,
            y_labels=labels,
            y_proba=blended,
            problem_type="classification",
            classes=classes,
        )
        metrics["ece"] = float(metrics["expected_calibration_error"])
        primary = "roc_auc" if len(classes) == 2 else "log_loss"
        return {name: float(value) for name, value in metrics.items()}, primary

    @staticmethod
    def _expert_alignment_diagnostics(
        request: RoutingReplayRequest,
        weights: np.ndarray,
        blended: np.ndarray,
    ) -> dict[str, Any]:
        if request.problem_type == "regression":
            expert_losses = (request.expert_outputs - request.target[:, None]) ** 2
            blended_losses = (blended - request.target) ** 2
        else:
            classes = np.asarray(request.classes)
            class_to_index = {label: idx for idx, label in enumerate(classes.tolist())}
            target_index = np.asarray([class_to_index[label] for label in request.target], dtype=int)
            row_index = np.arange(request.target.shape[0])
            expert_true_proba = request.expert_outputs[
                row_index[:, None],
                np.arange(request.expert_outputs.shape[1])[None, :],
                target_index[:, None],
            ]
            expert_losses = -np.log(np.maximum(expert_true_proba, 1e-15))
            blended_losses = -np.log(np.maximum(blended[row_index, target_index], 1e-15))
        oracle_losses = np.min(expert_losses, axis=1)
        correlation = RoutingReplayEvaluator._rank_correlation(
            weights.reshape(-1),
            (-expert_losses).reshape(-1),
        )
        return {
            "oracle_expert_regret": float(np.mean(blended_losses - oracle_losses)),
            "mean_oracle_loss": float(np.mean(oracle_losses)),
            "mean_blended_loss": float(np.mean(blended_losses)),
            "weight_negative_loss_rank_correlation": correlation,
        }

    @staticmethod
    def _routing_diagnostics(weights: np.ndarray) -> dict[str, Any]:
        entropy = -np.sum(weights * np.log(np.maximum(weights, 1e-12)), axis=1)
        ordered_weights = np.sort(weights, axis=1)
        top1_margin = (
            ordered_weights[:, -1] - ordered_weights[:, -2]
            if weights.shape[1] > 1
            else ordered_weights[:, -1]
        )
        assignments = np.argmax(weights, axis=1)
        counts = np.bincount(assignments, minlength=weights.shape[1])
        soft_mass = np.sum(weights, axis=0)
        hard_imbalance = (
            float("inf")
            if np.any(counts == 0)
            else float(np.max(counts) / np.min(counts))
        )
        soft_imbalance = float(np.max(soft_mass) / max(np.min(soft_mass), 1e-12))
        return {
            "mean_max_probability": float(np.mean(np.max(weights, axis=1))),
            "mean_top1_margin": float(np.mean(top1_margin)),
            "median_top1_margin": float(np.median(top1_margin)),
            "mean_entropy": float(np.mean(entropy)),
            "effective_expert_count": float(np.exp(np.mean(entropy))),
            "hard_assignment_counts": counts.astype(int).tolist(),
            "soft_assignment_mass": soft_mass.tolist(),
            "dead_expert_count": int(np.sum(soft_mass <= 1e-8)),
            "hard_assignment_dead_expert_count": int(np.sum(counts == 0)),
            "hard_assignment_imbalance": hard_imbalance,
            "soft_assignment_imbalance": soft_imbalance,
        }

    @staticmethod
    def _rank_correlation(left: np.ndarray, right: np.ndarray) -> Optional[float]:
        if left.size < 2 or np.all(left == left[0]) or np.all(right == right[0]):
            return None
        correlation = float(np.corrcoef(rankdata(left), rankdata(right))[0, 1])
        return correlation if np.isfinite(correlation) else None
