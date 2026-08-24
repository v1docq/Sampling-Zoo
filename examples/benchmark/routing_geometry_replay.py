"""Replay Phase A routing arms over one fitted RMT chunk ensemble."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold, StratifiedKFold
from tqdm.auto import tqdm

from sampling_zoo.core.experiment.routing_geometry_selection import (
    CrossFittedRoutingGeometrySelector,
    RoutingGeometryGuardMetricScore,
    RoutingGeometryFoldScore,
    RoutingGeometrySelectionDecision,
    RoutingGeometryTailRiskScore,
)
from sampling_zoo.core.experiment.routing_replay import (
    RoutingGeometrySelection,
    RoutingGeometrySelectionPolicy,
    RoutingReplayEvaluator,
    RoutingReplayRequest,
    RoutingReplayResult,
    ValidationRoutingGeometrySelector,
)
from sampling_zoo.core.metrics.eval_metrics import HIGHER_IS_BETTER, LOWER_IS_BETTER
from sampling_zoo.core.sampling_strategies.spectral.routing_contracts import (
    PartitionGeometrySpec,
    RoutingDistanceContract,
)


DEFAULT_ROUTING_TEMPERATURES: tuple[float, ...] = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0)
DEFAULT_VALIDATION_SELECTOR_ARM_NAMES: tuple[str, ...] = (
    "A2_median_scaled_euclidean",
    "A5_gmm_posterior",
    "A6_cosine_negative_control",
)


@dataclass(frozen=True)
class RoutingGeometryArm:
    name: str
    spec: PartitionGeometrySpec
    temperature_candidates: tuple[float, ...] = DEFAULT_ROUTING_TEMPERATURES
    use_validation_priors: bool = True


@dataclass(frozen=True)
class CrossFittedRoutingReplaySelection:
    """Cross-fitted decision plus full-validation calibration for outer testing."""

    decision: RoutingGeometrySelectionDecision
    selected_arm: RoutingGeometryArm
    validation_results: tuple[RoutingReplayResult, ...]
    fold_scores: tuple[RoutingGeometryFoldScore, ...]
    expert_priors: Optional[tuple[float, ...]]

    @property
    def validation_result(self) -> RoutingReplayResult:
        return next(
            result
            for result in self.validation_results
            if result.arm_name == self.selected_arm.name
        )


def default_routing_geometry_arms() -> tuple[RoutingGeometryArm, ...]:
    """Return the agreed Phase A screening grid in stable execution order."""

    return (
        RoutingGeometryArm(
            name="A0_uniform_voting",
            spec=PartitionGeometrySpec(uniform_shrinkage=1.0),
            temperature_candidates=(1.0,),
            use_validation_priors=False,
        ),
        RoutingGeometryArm(
            name="A1_source_centroid_squared_euclidean",
            spec=PartitionGeometrySpec(),
            temperature_candidates=(1.0,),
        ),
        RoutingGeometryArm(
            name="A2_median_scaled_euclidean",
            spec=PartitionGeometrySpec(metric="median_scaled_euclidean"),
        ),
        RoutingGeometryArm(
            name="A3_diag_shrinkage_mahalanobis",
            spec=PartitionGeometrySpec(metric="diag_shrinkage_mahalanobis"),
        ),
        RoutingGeometryArm(
            name="A4_full_shrinkage_mahalanobis",
            spec=PartitionGeometrySpec(metric="full_shrinkage_mahalanobis"),
        ),
        RoutingGeometryArm(
            name="A5_gmm_posterior",
            spec=PartitionGeometrySpec(
                metric="full_shrinkage_mahalanobis",
                kernel="gmm_posterior",
            ),
        ),
        RoutingGeometryArm(
            name="A6_cosine_negative_control",
            spec=PartitionGeometrySpec(metric="cosine"),
        ),
    )


def default_validation_selector_arms() -> tuple[RoutingGeometryArm, ...]:
    """Return the selector candidates identified by the Phase A screen."""

    by_name = {arm.name: arm for arm in default_routing_geometry_arms()}
    return tuple(by_name[name] for name in DEFAULT_VALIDATION_SELECTOR_ARM_NAMES)


def default_validation_selector_policy(
    *,
    minimum_improvement: float = 0.0,
) -> RoutingGeometrySelectionPolicy:
    return RoutingGeometrySelectionPolicy(
        candidate_arm_names=DEFAULT_VALIDATION_SELECTOR_ARM_NAMES,
        fallback_arm_name="A2_median_scaled_euclidean",
        minimum_improvement=minimum_improvement,
    )


class FittedEnsembleRoutingReplay:
    """Adapter from a fitted SamplingEnsemble to pure routing replay contracts."""

    def __init__(
        self,
        evaluator: Optional[RoutingReplayEvaluator] = None,
        *,
        show_progress: bool = True,
    ) -> None:
        self.evaluator = evaluator or RoutingReplayEvaluator()
        self.show_progress = bool(show_progress)

    def run_validation(
        self,
        *,
        ensemble: Any,
        X_val: pd.DataFrame,
        y_val: Any,
        arms: Optional[Sequence[RoutingGeometryArm]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> list[RoutingReplayResult]:
        context = self._prepare_context(
            ensemble=ensemble,
            features=X_val,
            stage="validation",
        )
        selected_arms = tuple(arms or default_routing_geometry_arms())
        results = []
        for arm in self._iter_arms(selected_arms):
            request = self._build_request(
                ensemble=ensemble,
                target=y_val,
                arm=arm,
                context=context,
                metadata=metadata,
            )
            selection = self.evaluator.calibrate_temperature(
                request,
                arm.temperature_candidates,
            )
            results.append(selection.best)
        return results

    def run_evaluation(
        self,
        *,
        ensemble: Any,
        features: pd.DataFrame,
        target: Any,
        arms: Sequence[RoutingGeometryArm],
        temperatures: Mapping[str, float],
        expert_priors: Optional[Sequence[float]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> list[RoutingReplayResult]:
        """Evaluate validation-selected arms on held-out rows without recalibration."""

        context = self._prepare_context(
            ensemble=ensemble,
            features=features,
            stage="inference",
        )
        results = []
        for arm in self._iter_arms(tuple(arms)):
            if arm.name not in temperatures:
                raise ValueError(f"Missing selected temperature for arm {arm.name!r}")
            request = self._build_request(
                ensemble=ensemble,
                target=target,
                arm=arm,
                context=context,
                metadata=metadata,
                expert_priors=expert_priors,
            )
            results.append(
                self.evaluator.evaluate(
                    request,
                    temperature=float(temperatures[arm.name]),
                )
            )
        return results

    def select_geometry_cross_fitted(
        self,
        *,
        ensemble: Any,
        X_val: pd.DataFrame,
        y_val: Any,
        arms: Sequence[RoutingGeometryArm],
        selector: CrossFittedRoutingGeometrySelector,
        n_folds: int = 5,
        random_state: int = 42,
        regression_stratification_bins: Optional[int] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> CrossFittedRoutingReplaySelection:
        """Select geometry on inner folds without consulting outer test rows."""

        selected_arms = tuple(arms)
        arm_by_name = self._index_arms(selected_arms)
        expected_names = (
            selector.spec.reference_arm,
            *selector.spec.candidate_arms,
        )
        if tuple(arm_by_name) != expected_names:
            raise ValueError(
                "Cross-fitted arms must match selector reference and candidates in stable order"
            )

        context = self._prepare_context(
            ensemble=ensemble,
            features=X_val,
            stage="validation",
        )
        requests = {
            arm.name: self._build_request(
                ensemble=ensemble,
                target=y_val,
                arm=arm,
                context=context,
                metadata=metadata,
            )
            for arm in selected_arms
        }
        splits = self._selection_splits(
            np.asarray(y_val),
            problem_type=str(ensemble.problem),
            n_folds=int(n_folds),
            random_state=int(random_state),
            regression_stratification_bins=regression_stratification_bins,
        )
        fold_scores = []
        reference_request = requests[selector.spec.reference_arm]
        for fold_index, (calibration_indices, evaluation_indices) in enumerate(splits):
            fold_priors = self._fit_cross_fitted_priors(
                reference_request,
                calibration_indices,
            )
            for arm in selected_arms:
                request = requests[arm.name]
                calibration_request = self._slice_request(request, calibration_indices)
                evaluation_request = self._slice_request(request, evaluation_indices)
                if arm.use_validation_priors:
                    calibration_request = self._with_expert_priors(
                        calibration_request,
                        fold_priors,
                    )
                    evaluation_request = self._with_expert_priors(
                        evaluation_request,
                        fold_priors,
                    )
                temperature = self.evaluator.calibrate_temperature(
                    calibration_request,
                    arm.temperature_candidates,
                ).best.temperature
                result = self.evaluator.evaluate(
                    evaluation_request,
                    temperature=temperature,
                )
                fold_scores.append(
                    self._fold_score(
                        result,
                        fold_id=str(fold_index),
                        evaluation_rows=len(evaluation_indices),
                    )
                )

        decision = selector.select(fold_scores)
        full_indices = np.arange(reference_request.target.shape[0])
        full_priors = self._fit_cross_fitted_priors(reference_request, full_indices)
        validation_results = []
        for arm in selected_arms:
            request = requests[arm.name]
            if arm.use_validation_priors:
                request = self._with_expert_priors(request, full_priors)
            validation_results.append(
                self.evaluator.calibrate_temperature(
                    request,
                    arm.temperature_candidates,
                ).best
            )

        return CrossFittedRoutingReplaySelection(
            decision=decision,
            selected_arm=arm_by_name[decision.selected_arm],
            validation_results=tuple(validation_results),
            fold_scores=tuple(fold_scores),
            expert_priors=tuple(float(value) for value in full_priors),
        )

    @staticmethod
    def select_validation_geometry(
        results: Sequence[RoutingReplayResult],
        *,
        policy: Optional[RoutingGeometrySelectionPolicy] = None,
        selector: Optional[ValidationRoutingGeometrySelector] = None,
    ) -> RoutingGeometrySelection:
        """Select one geometry from held-out validation results."""

        return (selector or ValidationRoutingGeometrySelector()).select(
            results,
            policy or default_validation_selector_policy(),
        )

    def _prepare_context(
        self,
        *,
        ensemble: Any,
        features: pd.DataFrame,
        stage: str,
    ) -> dict[str, Any]:
        partitioner = getattr(ensemble, "partitioner", None)
        if partitioner is None or not hasattr(partitioner, "transform_embedding"):
            raise ValueError("Routing replay requires a fitted RMT partitioner")

        model_names, expert_outputs = ensemble.export_expert_outputs(
            features,
            stage=stage,
        )
        embedding = partitioner.transform_embedding(features)
        validation_priors = ensemble.validation_prior_weights()
        raw_classes = getattr(ensemble, "classes_", None)
        classes = () if raw_classes is None else tuple(np.asarray(raw_classes).tolist())
        return {
            "partitioner": partitioner,
            "model_names": model_names,
            "expert_outputs": expert_outputs,
            "embedding": embedding,
            "validation_priors": validation_priors,
            "classes": classes,
        }

    def _build_request(
        self,
        *,
        ensemble: Any,
        target: Any,
        arm: RoutingGeometryArm,
        context: Mapping[str, Any],
        metadata: Optional[dict[str, Any]],
        expert_priors: Optional[Sequence[float]] = None,
    ) -> RoutingReplayRequest:
        partitioner = context["partitioner"]
        geometry = partitioner.build_partition_geometry(arm.spec)
        distances, _weights = partitioner.route_embedding(
            context["embedding"],
            geometry=geometry,
        )
        aligned_distances = self._align_distances(
            distances,
            context["model_names"],
        )
        resolved_priors = None
        if arm.use_validation_priors:
            resolved_priors = (
                context["validation_priors"]
                if expert_priors is None
                else np.asarray(expert_priors, dtype=float)
            )
        return RoutingReplayRequest(
            arm_name=arm.name,
            geometry_spec=arm.spec,
            distances=aligned_distances,
            target=np.asarray(target),
            expert_outputs=context["expert_outputs"],
            problem_type=str(ensemble.problem),
            classes=context["classes"],
            expert_priors=resolved_priors,
            metadata=metadata or {},
        )

    @staticmethod
    def _selection_splits(
        target: np.ndarray,
        *,
        problem_type: str,
        n_folds: int,
        random_state: int,
        regression_stratification_bins: Optional[int] = None,
    ) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
        if n_folds < 3:
            raise ValueError("n_folds must be at least 3")
        if target.shape[0] < n_folds:
            raise ValueError("Validation rows must be at least n_folds")
        indices = np.arange(target.shape[0])
        if problem_type == "classification":
            _, counts = np.unique(target, return_counts=True)
            effective_folds = min(n_folds, int(np.min(counts)))
            if effective_folds < 3:
                raise ValueError(
                    "Classification cross-fitting requires at least three rows per class"
                )
            splitter = StratifiedKFold(
                n_splits=effective_folds,
                shuffle=True,
                random_state=random_state,
            )
            split_iterator = splitter.split(indices, target)
        elif regression_stratification_bins is None:
            splitter = KFold(
                n_splits=n_folds,
                shuffle=True,
                random_state=random_state,
            )
            split_iterator = splitter.split(indices)
        else:
            labels = FittedEnsembleRoutingReplay._regression_stratification_labels(
                target,
                n_folds=n_folds,
                requested_bins=int(regression_stratification_bins),
            )
            splitter = StratifiedKFold(
                n_splits=n_folds,
                shuffle=True,
                random_state=random_state,
            )
            split_iterator = splitter.split(indices, labels)
        return tuple(
            (np.asarray(calibration), np.asarray(evaluation))
            for calibration, evaluation in split_iterator
        )

    @staticmethod
    def _regression_stratification_labels(
        target: np.ndarray,
        *,
        n_folds: int,
        requested_bins: int,
    ) -> np.ndarray:
        """Assign balanced target-rank bins for regression cross-fitting."""

        values = np.asarray(target, dtype=float).reshape(-1)
        if not np.all(np.isfinite(values)):
            raise ValueError("Regression stratification target must be finite")
        if requested_bins < 2:
            raise ValueError("regression_stratification_bins must be at least 2")
        effective_bins = min(int(requested_bins), values.size // int(n_folds))
        if effective_bins < 2:
            raise ValueError(
                "Regression stratification requires at least two bins with n_folds rows"
            )
        order = np.argsort(values, kind="stable")
        labels = np.empty(values.size, dtype=int)
        labels[order] = np.minimum(
            np.arange(values.size, dtype=int) * effective_bins // values.size,
            effective_bins - 1,
        )
        return labels

    @staticmethod
    def _slice_request(
        request: RoutingReplayRequest,
        indices: np.ndarray,
    ) -> RoutingReplayRequest:
        distances = RoutingDistanceContract(
            partition_names=request.distances.partition_names,
            values=request.distances.values[indices],
            kind=request.distances.kind,
            diagnostics=dict(request.distances.diagnostics),
        )
        return RoutingReplayRequest(
            arm_name=request.arm_name,
            geometry_spec=request.geometry_spec,
            distances=distances,
            target=request.target[indices],
            expert_outputs=request.expert_outputs[indices],
            problem_type=request.problem_type,
            classes=request.classes,
            expert_priors=request.expert_priors,
            metadata=dict(request.metadata),
        )

    @staticmethod
    def _with_expert_priors(
        request: RoutingReplayRequest,
        priors: Sequence[float],
    ) -> RoutingReplayRequest:
        return RoutingReplayRequest(
            arm_name=request.arm_name,
            geometry_spec=request.geometry_spec,
            distances=request.distances,
            target=request.target,
            expert_outputs=request.expert_outputs,
            problem_type=request.problem_type,
            classes=request.classes,
            expert_priors=np.asarray(priors, dtype=float),
            metadata=dict(request.metadata),
        )

    @staticmethod
    def _fit_cross_fitted_priors(
        request: RoutingReplayRequest,
        calibration_indices: np.ndarray,
    ) -> np.ndarray:
        target = request.target[calibration_indices]
        outputs = request.expert_outputs[calibration_indices]
        if request.problem_type == "regression":
            rmse = np.sqrt(np.mean((outputs - target[:, None]) ** 2, axis=0))
            quality = 1.0 / np.maximum(rmse, np.finfo(float).eps)
        else:
            classes = np.asarray(request.classes)
            quality = np.asarray(
                [
                    f1_score(
                        target,
                        classes[np.argmax(outputs[:, expert_index, :], axis=1)],
                        labels=classes,
                        average="weighted",
                        zero_division=0,
                    )
                    for expert_index in range(outputs.shape[1])
                ],
                dtype=float,
            )
            quality = np.maximum(quality, np.finfo(float).eps)
        return quality / np.sum(quality)

    @staticmethod
    def _fold_score(
        result: RoutingReplayResult,
        *,
        fold_id: str,
        evaluation_rows: int,
    ) -> RoutingGeometryFoldScore:
        metric_name = result.primary_metric.lower()
        if metric_name in LOWER_IS_BETTER:
            direction = "lower"
        elif metric_name in HIGHER_IS_BETTER:
            direction = "higher"
        else:
            raise ValueError(f"Unknown metric direction for {result.primary_metric!r}")
        has_mae = "mae" in result.metrics
        has_tail = "tail_mean_absolute_error" in result.metrics
        tail_quantile = result.diagnostics.get("tail_absolute_error_quantile")
        classification_guards = (
            tuple(
                RoutingGeometryGuardMetricScore(
                    metric=metric,
                    value=float(result.metrics[metric]),
                    direction=direction,
                )
                for metric, direction in (
                    ("brier_score", "lower"),
                    ("expected_calibration_error", "lower"),
                    ("f1_macro", "higher"),
                    ("worst_class_recall", "higher"),
                )
                if metric in result.metrics and np.isfinite(result.metrics[metric])
            )
            if metric_name in {"roc_auc", "log_loss"}
            else ()
        )
        return RoutingGeometryFoldScore(
            arm_name=result.arm_name,
            fold_id=fold_id,
            primary_metric=result.primary_metric,
            primary_value=result.primary_value,
            primary_direction=direction,
            robust_metric="mae" if has_mae else None,
            robust_value=float(result.metrics["mae"]) if has_mae else None,
            robust_direction="lower" if has_mae else None,
            tail_risk=(
                RoutingGeometryTailRiskScore(
                    metric="tail_mean_absolute_error",
                    value=float(result.metrics["tail_mean_absolute_error"]),
                    direction="lower",
                    quantile=float(tail_quantile),
                )
                if has_tail and tail_quantile is not None
                else None
            ),
            guard_metrics=classification_guards,
            evaluation_rows=evaluation_rows,
            selected_temperature=result.temperature,
        )

    @staticmethod
    def _index_arms(
        arms: Sequence[RoutingGeometryArm],
    ) -> dict[str, RoutingGeometryArm]:
        indexed = {}
        for arm in arms:
            if arm.name in indexed:
                raise ValueError(f"Duplicate routing arm {arm.name!r}")
            indexed[arm.name] = arm
        return indexed

    def _iter_arms(self, arms: Sequence[RoutingGeometryArm]):
        return tqdm(
            arms,
            desc="Replay routing geometry",
            disable=not self.show_progress,
            leave=False,
        )

    @staticmethod
    def records(
        results: Iterable[RoutingReplayResult],
        *,
        metadata: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        common = dict(metadata or {})
        return [{**common, **result.summary()} for result in results]

    @staticmethod
    def _align_distances(
        distances: RoutingDistanceContract,
        model_names: Sequence[str],
    ) -> RoutingDistanceContract:
        column_by_name = {
            name: idx for idx, name in enumerate(distances.partition_names)
        }
        missing = [name for name in model_names if name not in column_by_name]
        if missing:
            raise ValueError(f"Expert names are missing from partition geometry: {missing}")
        columns = [column_by_name[name] for name in model_names]
        return RoutingDistanceContract(
            partition_names=tuple(model_names),
            values=distances.values[:, columns],
            kind=distances.kind,
            diagnostics={
                **dict(distances.diagnostics),
                "aligned_from_partition_count": len(distances.partition_names),
            },
        )
