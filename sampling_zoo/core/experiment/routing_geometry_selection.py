"""Cross-fitted selection of routing geometry from paired fold metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple
import zlib

import numpy as np


_METRIC_DIRECTIONS = {"lower", "higher"}
_GAIN_SCALES = {"absolute", "relative"}


@dataclass(frozen=True)
class ClassificationMetricGuardSpec:
    """One probability or class-balance metric used as a selection guard."""

    metric: str
    direction: str
    gain_scale: str
    noninferiority_margin: float

    def __post_init__(self) -> None:
        if not self.metric:
            raise ValueError("classification guard metric must be non-empty")
        if self.direction not in _METRIC_DIRECTIONS:
            raise ValueError("classification guard direction must be lower or higher")
        if self.gain_scale not in _GAIN_SCALES:
            raise ValueError("classification guard gain_scale must be absolute or relative")
        if float(self.noninferiority_margin) < 0.0:
            raise ValueError("classification guard margin must be non-negative")
        object.__setattr__(self, "metric", self.metric.strip().lower())
        object.__setattr__(
            self,
            "noninferiority_margin",
            float(self.noninferiority_margin),
        )


@dataclass(frozen=True)
class ClassificationRoutingGuardSpec:
    """AMLB-compatible primary and probability-quality routing guards."""

    roc_auc_absolute_margin: float = 0.005
    log_loss_relative_margin: float = 0.01
    primary_selection_margin: Optional[float] = None
    metric_guards: Tuple[ClassificationMetricGuardSpec, ...] = (
        ClassificationMetricGuardSpec(
            metric="brier_score",
            direction="lower",
            gain_scale="relative",
            noninferiority_margin=0.01,
        ),
        ClassificationMetricGuardSpec(
            metric="expected_calibration_error",
            direction="lower",
            gain_scale="absolute",
            noninferiority_margin=0.01,
        ),
        ClassificationMetricGuardSpec(
            metric="f1_macro",
            direction="higher",
            gain_scale="absolute",
            noninferiority_margin=0.01,
        ),
        ClassificationMetricGuardSpec(
            metric="worst_class_recall",
            direction="higher",
            gain_scale="absolute",
            noninferiority_margin=0.01,
        ),
    )
    require_all_metrics: bool = True

    def __post_init__(self) -> None:
        if float(self.roc_auc_absolute_margin) < 0.0:
            raise ValueError("roc_auc_absolute_margin must be non-negative")
        if float(self.log_loss_relative_margin) < 0.0:
            raise ValueError("log_loss_relative_margin must be non-negative")
        if (
            self.primary_selection_margin is not None
            and float(self.primary_selection_margin) < 0.0
        ):
            raise ValueError("primary_selection_margin must be non-negative")
        if self.primary_selection_margin is not None:
            object.__setattr__(
                self,
                "primary_selection_margin",
                float(self.primary_selection_margin),
            )
        guards = tuple(self.metric_guards)
        names = tuple(item.metric for item in guards)
        if len(set(names)) != len(names):
            raise ValueError("classification metric guards must be unique")
        object.__setattr__(self, "metric_guards", guards)

    def primary_guard(self, metric: str) -> ClassificationMetricGuardSpec:
        normalized = str(metric).strip().lower()
        strict_margin = self.primary_selection_margin
        if normalized == "roc_auc":
            return ClassificationMetricGuardSpec(
                metric="roc_auc",
                direction="higher",
                gain_scale="absolute",
                noninferiority_margin=float(
                    self.roc_auc_absolute_margin
                    if strict_margin is None
                    else strict_margin
                ),
            )
        if normalized == "log_loss":
            return ClassificationMetricGuardSpec(
                metric="log_loss",
                direction="lower",
                gain_scale="relative",
                noninferiority_margin=float(
                    self.log_loss_relative_margin
                    if strict_margin is None
                    else strict_margin
                ),
            )
        raise ValueError(
            "Classification routing guard supports roc_auc or log_loss primary metrics"
        )


@dataclass(frozen=True)
class RoutingGeometryGuardMetricScore:
    """One held-out classification guard metric."""

    metric: str
    value: float
    direction: str

    def __post_init__(self) -> None:
        if not self.metric:
            raise ValueError("guard metric must be non-empty")
        if self.direction not in _METRIC_DIRECTIONS:
            raise ValueError("guard metric direction must be lower or higher")
        if not np.isfinite(float(self.value)):
            raise ValueError("guard metric value must be finite")
        object.__setattr__(self, "metric", self.metric.strip().lower())
        object.__setattr__(self, "value", float(self.value))


@dataclass(frozen=True)
class ClassificationRoutingGuardEvidence:
    """Paired fold evidence for one classification non-inferiority guard."""

    metric: str
    direction: str
    gain_scale: str
    noninferiority_margin: float
    paired_fold_count: int
    mean_gain: Optional[float]
    median_gain: Optional[float]
    confidence_interval: Optional[Tuple[float, float]]
    eligible: bool
    rejection_reasons: Tuple[str, ...] = ()

    def summary(self) -> dict[str, object]:
        return {
            "metric": self.metric,
            "direction": self.direction,
            "gain_scale": self.gain_scale,
            "noninferiority_margin": self.noninferiority_margin,
            "paired_fold_count": self.paired_fold_count,
            "mean_gain": self.mean_gain,
            "median_gain": self.median_gain,
            "confidence_interval": self.confidence_interval,
            "eligible": self.eligible,
            "rejection_reasons": list(self.rejection_reasons),
        }


@dataclass(frozen=True)
class RoutingGeometryTailRiskScore:
    """One upper-tail risk metric attached to a held-out regression fold."""

    metric: str
    value: float
    direction: str
    quantile: float

    def __post_init__(self) -> None:
        if not self.metric:
            raise ValueError("tail risk metric must be non-empty")
        if self.direction not in _METRIC_DIRECTIONS:
            raise ValueError("tail risk direction must be lower or higher")
        if not np.isfinite(float(self.value)):
            raise ValueError("tail risk value must be finite")
        if not 0.0 <= float(self.quantile) < 1.0:
            raise ValueError("tail risk quantile must be in [0, 1)")
        object.__setattr__(self, "value", float(self.value))
        object.__setattr__(self, "quantile", float(self.quantile))


@dataclass(frozen=True)
class RoutingGeometryFoldScore:
    """Metrics for one geometry evaluated on one held-out inner fold."""

    arm_name: str
    fold_id: str
    primary_metric: str
    primary_value: float
    primary_direction: str
    evaluation_rows: int
    selected_temperature: float
    robust_metric: Optional[str] = None
    robust_value: Optional[float] = None
    robust_direction: Optional[str] = None
    tail_risk: Optional[RoutingGeometryTailRiskScore] = None
    guard_metrics: Tuple[RoutingGeometryGuardMetricScore, ...] = ()

    def __post_init__(self) -> None:
        if not self.arm_name:
            raise ValueError("arm_name must be non-empty")
        if not self.fold_id:
            raise ValueError("fold_id must be non-empty")
        if not self.primary_metric:
            raise ValueError("primary_metric must be non-empty")
        if self.primary_direction not in _METRIC_DIRECTIONS:
            raise ValueError("primary_direction must be lower or higher")
        if not np.isfinite(self.primary_value):
            raise ValueError("primary_value must be finite")
        if int(self.evaluation_rows) < 1:
            raise ValueError("evaluation_rows must be positive")
        if not np.isfinite(self.selected_temperature) or self.selected_temperature <= 0:
            raise ValueError("selected_temperature must be finite and positive")

        robust_fields = (
            self.robust_metric,
            self.robust_value,
            self.robust_direction,
        )
        if any(value is not None for value in robust_fields):
            if any(value is None for value in robust_fields):
                raise ValueError("robust metric, value, and direction must be provided together")
            if self.robust_direction not in _METRIC_DIRECTIONS:
                raise ValueError("robust_direction must be lower or higher")
            if not np.isfinite(float(self.robust_value)):
                raise ValueError("robust_value must be finite")

        guards = tuple(self.guard_metrics)
        guard_names = tuple(item.metric for item in guards)
        if len(set(guard_names)) != len(guard_names):
            raise ValueError("guard metrics must be unique within a fold score")

        object.__setattr__(self, "evaluation_rows", int(self.evaluation_rows))
        object.__setattr__(self, "selected_temperature", float(self.selected_temperature))
        object.__setattr__(self, "guard_metrics", guards)

    @property
    def guard_metric_map(self) -> dict[str, RoutingGeometryGuardMetricScore]:
        return {item.metric: item for item in self.guard_metrics}

    def summary(self) -> dict[str, object]:
        return {
            "arm_name": self.arm_name,
            "fold_id": self.fold_id,
            "primary_metric": self.primary_metric,
            "primary_value": float(self.primary_value),
            "primary_direction": self.primary_direction,
            "robust_metric": self.robust_metric,
            "robust_value": (
                None if self.robust_value is None else float(self.robust_value)
            ),
            "robust_direction": self.robust_direction,
            "tail_metric": (
                None if self.tail_risk is None else self.tail_risk.metric
            ),
            "tail_value": (
                None if self.tail_risk is None else float(self.tail_risk.value)
            ),
            "tail_direction": (
                None if self.tail_risk is None else self.tail_risk.direction
            ),
            "tail_quantile": (
                None if self.tail_risk is None else float(self.tail_risk.quantile)
            ),
            "evaluation_rows": self.evaluation_rows,
            "selected_temperature": self.selected_temperature,
            "guard_metrics": [item.__dict__ for item in self.guard_metrics],
        }


@dataclass(frozen=True)
class CrossFittedRoutingGeometrySelectionSpec:
    """Predeclared gates for choosing an alternative to the A2 reference."""

    reference_arm: str = "A2_median_scaled_euclidean"
    candidate_arms: Tuple[str, ...] = (
        "A5_gmm_posterior",
        "A6_cosine_negative_control",
    )
    min_folds: int = 3
    confidence_level: float = 0.95
    bootstrap_iterations: int = 2_000
    min_positive_fold_fraction: float = 2.0 / 3.0
    min_mean_relative_gain: float = 0.0
    min_median_relative_gain: float = 0.0
    noninferiority_margin: float = 0.005
    require_robust_metric: bool = False
    tail_guard_primary_metrics: Tuple[str, ...] = ()
    tail_noninferiority_margin: Optional[float] = None
    classification_guard: Optional[ClassificationRoutingGuardSpec] = None
    random_state: int = 42

    def __post_init__(self) -> None:
        candidates = tuple(self.candidate_arms)
        if not self.reference_arm:
            raise ValueError("reference_arm must be non-empty")
        if not candidates:
            raise ValueError("candidate_arms must be non-empty")
        if self.reference_arm in candidates:
            raise ValueError("reference_arm cannot also be a candidate")
        if len(set(candidates)) != len(candidates):
            raise ValueError("candidate_arms must be unique")
        if int(self.min_folds) < 3:
            raise ValueError("min_folds must be at least 3")
        if not 0.0 < float(self.confidence_level) < 1.0:
            raise ValueError("confidence_level must be in (0, 1)")
        if int(self.bootstrap_iterations) < 100:
            raise ValueError("bootstrap_iterations must be at least 100")
        if not 0.0 < float(self.min_positive_fold_fraction) <= 1.0:
            raise ValueError("min_positive_fold_fraction must be in (0, 1]")
        if float(self.noninferiority_margin) < 0.0:
            raise ValueError("noninferiority_margin must be non-negative")
        tail_metrics = tuple(
            str(metric).strip().lower() for metric in self.tail_guard_primary_metrics
        )
        if any(not metric for metric in tail_metrics):
            raise ValueError("tail_guard_primary_metrics must contain non-empty names")
        if len(set(tail_metrics)) != len(tail_metrics):
            raise ValueError("tail_guard_primary_metrics must be unique")
        if (
            self.tail_noninferiority_margin is not None
            and float(self.tail_noninferiority_margin) < 0.0
        ):
            raise ValueError("tail_noninferiority_margin must be non-negative")
        object.__setattr__(self, "candidate_arms", candidates)
        object.__setattr__(self, "tail_guard_primary_metrics", tail_metrics)


@dataclass(frozen=True)
class RoutingGeometryCandidateEvidence:
    """Paired evidence for one candidate relative to the reference geometry."""

    arm_name: str
    paired_fold_count: int
    primary_metric: Optional[str]
    robust_metric: Optional[str]
    mean_relative_gain: Optional[float]
    median_relative_gain: Optional[float]
    confidence_interval: Optional[Tuple[float, float]]
    positive_fold_fraction: Optional[float]
    robust_mean_relative_gain: Optional[float]
    robust_median_relative_gain: Optional[float]
    robust_confidence_interval: Optional[Tuple[float, float]]
    robust_positive_fold_fraction: Optional[float]
    tail_metric: Optional[str]
    tail_quantile: Optional[float]
    tail_mean_relative_gain: Optional[float]
    tail_median_relative_gain: Optional[float]
    tail_confidence_interval: Optional[Tuple[float, float]]
    tail_positive_fold_fraction: Optional[float]
    eligible: bool
    rejection_reasons: Tuple[str, ...] = ()
    classification_guard_evidence: Tuple[
        ClassificationRoutingGuardEvidence, ...
    ] = ()

    def summary(self) -> dict[str, object]:
        return {
            "arm_name": self.arm_name,
            "paired_fold_count": self.paired_fold_count,
            "primary_metric": self.primary_metric,
            "robust_metric": self.robust_metric,
            "mean_relative_gain": self.mean_relative_gain,
            "median_relative_gain": self.median_relative_gain,
            "confidence_interval": self.confidence_interval,
            "positive_fold_fraction": self.positive_fold_fraction,
            "robust_mean_relative_gain": self.robust_mean_relative_gain,
            "robust_median_relative_gain": self.robust_median_relative_gain,
            "robust_confidence_interval": self.robust_confidence_interval,
            "robust_positive_fold_fraction": self.robust_positive_fold_fraction,
            "tail_metric": self.tail_metric,
            "tail_quantile": self.tail_quantile,
            "tail_mean_relative_gain": self.tail_mean_relative_gain,
            "tail_median_relative_gain": self.tail_median_relative_gain,
            "tail_confidence_interval": self.tail_confidence_interval,
            "tail_positive_fold_fraction": self.tail_positive_fold_fraction,
            "eligible": self.eligible,
            "rejection_reasons": list(self.rejection_reasons),
            "classification_guard_evidence": [
                item.summary() for item in self.classification_guard_evidence
            ],
        }


@dataclass(frozen=True)
class RoutingGeometrySelectionDecision:
    """Explicit selected/fallback result that never consults the outer test split."""

    status: str
    selected_arm: str
    reference_arm: str
    reason: str
    evidence: Tuple[RoutingGeometryCandidateEvidence, ...]

    def __post_init__(self) -> None:
        if self.status not in {"selected", "fallback_to_a2"}:
            raise ValueError("status must be selected or fallback_to_a2")
        if self.status == "fallback_to_a2" and self.selected_arm != self.reference_arm:
            raise ValueError("fallback_to_a2 must select the reference arm")

    def summary(self) -> dict[str, object]:
        return {
            "status": self.status,
            "selected_arm": self.selected_arm,
            "reference_arm": self.reference_arm,
            "reason": self.reason,
            "evidence": [item.summary() for item in self.evidence],
        }


class CrossFittedRoutingGeometrySelector:
    """Select a routing geometry using paired, cross-fitted metric evidence."""

    def __init__(
        self,
        spec: Optional[CrossFittedRoutingGeometrySelectionSpec] = None,
    ) -> None:
        self.spec = spec or CrossFittedRoutingGeometrySelectionSpec()

    def select(
        self,
        scores: Sequence[RoutingGeometryFoldScore],
    ) -> RoutingGeometrySelectionDecision:
        indexed = self._index_scores(scores)
        reference = indexed.get(self.spec.reference_arm, {})
        if len(reference) < self.spec.min_folds:
            raise ValueError(
                "Reference geometry must have at least "
                f"{self.spec.min_folds} complete fold scores"
            )

        evidence = tuple(
            self._candidate_evidence(
                arm_name=arm_name,
                reference=reference,
                candidate=indexed.get(arm_name, {}),
            )
            for arm_name in self.spec.candidate_arms
        )
        eligible = [item for item in evidence if item.eligible]
        if not eligible:
            return RoutingGeometrySelectionDecision(
                status="fallback_to_a2",
                selected_arm=self.spec.reference_arm,
                reference_arm=self.spec.reference_arm,
                reason="no_candidate_passed_cross_fitted_gates",
                evidence=evidence,
            )

        selected = sorted(
            eligible,
            key=lambda item: (
                -self._confidence_lower(item.confidence_interval),
                -float(item.median_relative_gain),
                -float(item.mean_relative_gain),
                item.arm_name,
            ),
        )[0]
        return RoutingGeometrySelectionDecision(
            status="selected",
            selected_arm=selected.arm_name,
            reference_arm=self.spec.reference_arm,
            reason="candidate_passed_cross_fitted_gates",
            evidence=evidence,
        )

    @staticmethod
    def _index_scores(
        scores: Sequence[RoutingGeometryFoldScore],
    ) -> dict[str, dict[str, RoutingGeometryFoldScore]]:
        indexed: dict[str, dict[str, RoutingGeometryFoldScore]] = {}
        for score in scores:
            by_fold = indexed.setdefault(score.arm_name, {})
            if score.fold_id in by_fold:
                raise ValueError(
                    f"Duplicate fold score for {score.arm_name!r}, fold {score.fold_id!r}"
                )
            by_fold[score.fold_id] = score
        return indexed

    def _candidate_evidence(
        self,
        *,
        arm_name: str,
        reference: dict[str, RoutingGeometryFoldScore],
        candidate: dict[str, RoutingGeometryFoldScore],
    ) -> RoutingGeometryCandidateEvidence:
        reference_folds = set(reference)
        candidate_folds = set(candidate)
        paired_folds = sorted(reference_folds & candidate_folds)
        reasons = []
        if reference_folds != candidate_folds:
            reasons.append("incomplete_fold_pairs")
        if len(paired_folds) < self.spec.min_folds:
            reasons.append("insufficient_paired_folds")
        if not paired_folds:
            return self._empty_evidence(arm_name, reasons)

        primary_gains = []
        robust_gains = []
        tail_gains = []
        primary_metric = None
        robust_metric = None
        tail_metric = None
        tail_quantile = None
        for fold_id in paired_folds:
            reference_score = reference[fold_id]
            candidate_score = candidate[fold_id]
            self._validate_metric_pair(reference_score, candidate_score)
            primary_metric = reference_score.primary_metric
            primary_gains.append(
                self._relative_gain(
                    reference_score.primary_value,
                    candidate_score.primary_value,
                    reference_score.primary_direction,
                )
            )
            if reference_score.robust_metric is not None:
                robust_metric = reference_score.robust_metric
                robust_gains.append(
                    self._relative_gain(
                        float(reference_score.robust_value),
                        float(candidate_score.robust_value),
                        str(reference_score.robust_direction),
                    )
                )
            if reference_score.tail_risk is not None:
                tail_metric = reference_score.tail_risk.metric
                tail_quantile = reference_score.tail_risk.quantile
                tail_gains.append(
                    self._relative_gain(
                        reference_score.tail_risk.value,
                        candidate_score.tail_risk.value,
                        reference_score.tail_risk.direction,
                    )
                )

        primary = np.asarray(primary_gains, dtype=float)
        primary_ci = self._bootstrap_interval(primary, arm_name, offset=0)
        mean_gain = float(np.mean(primary))
        median_gain = float(np.median(primary))
        positive_fraction = float(np.mean(primary > 0.0))

        if mean_gain < self.spec.min_mean_relative_gain:
            reasons.append("mean_gain_below_threshold")
        if median_gain < self.spec.min_median_relative_gain:
            reasons.append("median_gain_below_threshold")
        if positive_fraction < self.spec.min_positive_fold_fraction:
            reasons.append("unstable_primary_gain_sign")
        if primary_ci[0] < -self.spec.noninferiority_margin:
            reasons.append("primary_confidence_interval_exceeds_harm_margin")

        robust_mean = None
        robust_median = None
        robust_ci = None
        robust_positive_fraction = None
        if robust_gains:
            robust = np.asarray(robust_gains, dtype=float)
            robust_mean = float(np.mean(robust))
            robust_median = float(np.median(robust))
            robust_ci = self._bootstrap_interval(robust, arm_name, offset=1)
            robust_positive_fraction = float(np.mean(robust > 0.0))
            if robust_mean < -self.spec.noninferiority_margin:
                reasons.append("robust_mean_exceeds_harm_margin")
            if robust_median < -self.spec.noninferiority_margin:
                reasons.append("robust_median_exceeds_harm_margin")
            if robust_ci[0] < -self.spec.noninferiority_margin:
                reasons.append("robust_confidence_interval_exceeds_harm_margin")
        elif self.spec.require_robust_metric:
            reasons.append("robust_metric_required")

        tail_mean = None
        tail_median = None
        tail_ci = None
        tail_positive_fraction = None
        tail_required = (
            primary_metric is not None
            and primary_metric.lower() in self.spec.tail_guard_primary_metrics
        )
        if tail_gains:
            tail = np.asarray(tail_gains, dtype=float)
            tail_mean = float(np.mean(tail))
            tail_median = float(np.median(tail))
            tail_ci = self._bootstrap_interval(tail, arm_name, offset=2)
            tail_positive_fraction = float(np.mean(tail > 0.0))
            tail_margin = (
                self.spec.noninferiority_margin
                if self.spec.tail_noninferiority_margin is None
                else float(self.spec.tail_noninferiority_margin)
            )
            if tail_required:
                if tail_mean < -tail_margin:
                    reasons.append("tail_mean_exceeds_harm_margin")
                if tail_median < -tail_margin:
                    reasons.append("tail_median_exceeds_harm_margin")
                if tail_ci[0] < -tail_margin:
                    reasons.append("tail_confidence_interval_exceeds_harm_margin")
        elif tail_required:
            reasons.append("tail_metric_required")

        classification_guard_evidence = self._classification_guard_evidence(
            arm_name=arm_name,
            paired_folds=paired_folds,
            reference=reference,
            candidate=candidate,
            primary_metric=primary_metric,
        )
        for guard in classification_guard_evidence:
            reasons.extend(
                f"classification_guard:{guard.metric}:{reason}"
                for reason in guard.rejection_reasons
            )

        return RoutingGeometryCandidateEvidence(
            arm_name=arm_name,
            paired_fold_count=len(paired_folds),
            primary_metric=primary_metric,
            robust_metric=robust_metric,
            mean_relative_gain=mean_gain,
            median_relative_gain=median_gain,
            confidence_interval=primary_ci,
            positive_fold_fraction=positive_fraction,
            robust_mean_relative_gain=robust_mean,
            robust_median_relative_gain=robust_median,
            robust_confidence_interval=robust_ci,
            robust_positive_fold_fraction=robust_positive_fraction,
            tail_metric=tail_metric,
            tail_quantile=tail_quantile,
            tail_mean_relative_gain=tail_mean,
            tail_median_relative_gain=tail_median,
            tail_confidence_interval=tail_ci,
            tail_positive_fold_fraction=tail_positive_fraction,
            eligible=not reasons,
            rejection_reasons=tuple(reasons),
            classification_guard_evidence=classification_guard_evidence,
        )

    def _classification_guard_evidence(
        self,
        *,
        arm_name: str,
        paired_folds: Sequence[str],
        reference: dict[str, RoutingGeometryFoldScore],
        candidate: dict[str, RoutingGeometryFoldScore],
        primary_metric: Optional[str],
    ) -> Tuple[ClassificationRoutingGuardEvidence, ...]:
        spec = self.spec.classification_guard
        if spec is None or primary_metric not in {"roc_auc", "log_loss"}:
            return ()

        metric_specs = (spec.primary_guard(primary_metric), *spec.metric_guards)
        evidence = []
        for offset, metric_spec in enumerate(metric_specs, start=100):
            gains = []
            missing = False
            for fold_id in paired_folds:
                reference_score = reference[fold_id]
                candidate_score = candidate[fold_id]
                if metric_spec.metric == primary_metric:
                    reference_value = reference_score.primary_value
                    candidate_value = candidate_score.primary_value
                else:
                    reference_metric = reference_score.guard_metric_map.get(
                        metric_spec.metric
                    )
                    candidate_metric = candidate_score.guard_metric_map.get(
                        metric_spec.metric
                    )
                    if reference_metric is None or candidate_metric is None:
                        missing = True
                        continue
                    reference_value = reference_metric.value
                    candidate_value = candidate_metric.value
                gains.append(
                    self._scaled_gain(
                        reference_value,
                        candidate_value,
                        metric_spec.direction,
                        metric_spec.gain_scale,
                    )
                )

            rejection_reasons = []
            if missing and spec.require_all_metrics:
                rejection_reasons.append("required_metric_missing")
            if len(gains) != len(paired_folds):
                rejection_reasons.append("incomplete_fold_pairs")
            if not gains:
                evidence.append(
                    ClassificationRoutingGuardEvidence(
                        metric=metric_spec.metric,
                        direction=metric_spec.direction,
                        gain_scale=metric_spec.gain_scale,
                        noninferiority_margin=metric_spec.noninferiority_margin,
                        paired_fold_count=0,
                        mean_gain=None,
                        median_gain=None,
                        confidence_interval=None,
                        eligible=False,
                        rejection_reasons=tuple(dict.fromkeys(rejection_reasons)),
                    )
                )
                continue

            gain_array = np.asarray(gains, dtype=float)
            mean_gain = float(np.mean(gain_array))
            median_gain = float(np.median(gain_array))
            confidence_interval = self._bootstrap_interval(
                gain_array,
                arm_name,
                offset=offset,
            )
            margin = float(metric_spec.noninferiority_margin)
            if mean_gain < -margin:
                rejection_reasons.append("mean_exceeds_harm_margin")
            if median_gain < -margin:
                rejection_reasons.append("median_exceeds_harm_margin")
            if confidence_interval[0] < -margin:
                rejection_reasons.append("confidence_interval_exceeds_harm_margin")
            evidence.append(
                ClassificationRoutingGuardEvidence(
                    metric=metric_spec.metric,
                    direction=metric_spec.direction,
                    gain_scale=metric_spec.gain_scale,
                    noninferiority_margin=margin,
                    paired_fold_count=len(gains),
                    mean_gain=mean_gain,
                    median_gain=median_gain,
                    confidence_interval=confidence_interval,
                    eligible=not rejection_reasons,
                    rejection_reasons=tuple(dict.fromkeys(rejection_reasons)),
                )
            )
        return tuple(evidence)

    @staticmethod
    def _empty_evidence(
        arm_name: str,
        reasons: Sequence[str],
    ) -> RoutingGeometryCandidateEvidence:
        return RoutingGeometryCandidateEvidence(
            arm_name=arm_name,
            paired_fold_count=0,
            primary_metric=None,
            robust_metric=None,
            mean_relative_gain=None,
            median_relative_gain=None,
            confidence_interval=None,
            positive_fold_fraction=None,
            robust_mean_relative_gain=None,
            robust_median_relative_gain=None,
            robust_confidence_interval=None,
            robust_positive_fold_fraction=None,
            tail_metric=None,
            tail_quantile=None,
            tail_mean_relative_gain=None,
            tail_median_relative_gain=None,
            tail_confidence_interval=None,
            tail_positive_fold_fraction=None,
            eligible=False,
            rejection_reasons=tuple(reasons),
        )

    @staticmethod
    def _validate_metric_pair(
        reference: RoutingGeometryFoldScore,
        candidate: RoutingGeometryFoldScore,
    ) -> None:
        if (
            reference.primary_metric != candidate.primary_metric
            or reference.primary_direction != candidate.primary_direction
        ):
            raise ValueError("Paired primary metrics and directions must match")
        reference_has_robust = reference.robust_metric is not None
        candidate_has_robust = candidate.robust_metric is not None
        if reference_has_robust != candidate_has_robust:
            raise ValueError("Paired scores must agree on robust metric availability")
        if reference_has_robust and (
            reference.robust_metric != candidate.robust_metric
            or reference.robust_direction != candidate.robust_direction
        ):
            raise ValueError("Paired robust metrics and directions must match")
        reference_has_tail = reference.tail_risk is not None
        candidate_has_tail = candidate.tail_risk is not None
        if reference_has_tail != candidate_has_tail:
            raise ValueError("Paired scores must agree on tail risk availability")
        if reference_has_tail and (
            reference.tail_risk.metric != candidate.tail_risk.metric
            or reference.tail_risk.direction != candidate.tail_risk.direction
            or not np.isclose(
                reference.tail_risk.quantile,
                candidate.tail_risk.quantile,
            )
        ):
            raise ValueError(
                "Paired tail risk metrics, directions, and quantiles must match"
            )
        reference_guards = reference.guard_metric_map
        candidate_guards = candidate.guard_metric_map
        if set(reference_guards) != set(candidate_guards):
            raise ValueError("Paired scores must agree on guard metric availability")
        for metric, reference_guard in reference_guards.items():
            if reference_guard.direction != candidate_guards[metric].direction:
                raise ValueError("Paired guard metric directions must match")

    @staticmethod
    def _relative_gain(reference: float, candidate: float, direction: str) -> float:
        scale = max(abs(float(reference)), np.finfo(float).eps)
        difference = (
            float(reference) - float(candidate)
            if direction == "lower"
            else float(candidate) - float(reference)
        )
        return difference / scale

    @staticmethod
    def _scaled_gain(
        reference: float,
        candidate: float,
        direction: str,
        scale: str,
    ) -> float:
        difference = (
            float(reference) - float(candidate)
            if direction == "lower"
            else float(candidate) - float(reference)
        )
        if scale == "absolute":
            return difference
        denominator = max(abs(float(reference)), np.finfo(float).eps)
        return difference / denominator

    def _bootstrap_interval(
        self,
        gains: np.ndarray,
        arm_name: str,
        *,
        offset: int,
    ) -> Tuple[float, float]:
        stable_arm_seed = zlib.crc32(arm_name.encode("utf-8"))
        seed = np.random.SeedSequence(
            [int(self.spec.random_state), int(stable_arm_seed), int(offset)]
        )
        rng = np.random.default_rng(seed)
        indices = rng.integers(
            0,
            gains.size,
            size=(int(self.spec.bootstrap_iterations), gains.size),
        )
        bootstrap_means = np.mean(gains[indices], axis=1)
        alpha = (1.0 - float(self.spec.confidence_level)) / 2.0
        lower, upper = np.quantile(bootstrap_means, (alpha, 1.0 - alpha))
        return float(lower), float(upper)

    @staticmethod
    def _confidence_lower(interval: Optional[Tuple[float, float]]) -> float:
        return float("-inf") if interval is None else float(interval[0])
