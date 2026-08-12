"""Cross-fitted conditional selection of an RMT row-sampling policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple
import zlib

import numpy as np


_DIRECTIONS = {"lower", "higher"}


@dataclass(frozen=True)
class RowSamplingPolicySpec:
    """Named row policy materialized at the sampler boundary."""

    name: str
    selection_method: str

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise ValueError("row policy name must be non-empty")
        if not str(self.selection_method).strip():
            raise ValueError("selection_method must be non-empty")


@dataclass(frozen=True)
class RowPolicyFoldScore:
    """Held-out evidence for one row policy and one inner fold."""

    policy_name: str
    fold_id: str
    primary_metric: str
    primary_value: float
    primary_direction: str
    evaluation_rows: int
    tail_metric: Optional[str] = None
    tail_value: Optional[float] = None
    tail_direction: Optional[str] = None
    class_coverage_preserved: Optional[bool] = None

    def __post_init__(self) -> None:
        if not self.policy_name or not self.fold_id or not self.primary_metric:
            raise ValueError("policy_name, fold_id, and primary_metric must be non-empty")
        if self.primary_direction not in _DIRECTIONS:
            raise ValueError("primary_direction must be lower or higher")
        if not np.isfinite(float(self.primary_value)):
            raise ValueError("primary_value must be finite")
        if int(self.evaluation_rows) < 1:
            raise ValueError("evaluation_rows must be positive")
        tail_fields = (self.tail_metric, self.tail_value, self.tail_direction)
        if any(value is not None for value in tail_fields):
            if any(value is None for value in tail_fields):
                raise ValueError("tail metric, value, and direction must be provided together")
            if self.tail_direction not in _DIRECTIONS:
                raise ValueError("tail_direction must be lower or higher")
            if not np.isfinite(float(self.tail_value)):
                raise ValueError("tail_value must be finite")
        object.__setattr__(self, "evaluation_rows", int(self.evaluation_rows))


@dataclass(frozen=True)
class RowSamplingPolicyRequest:
    """Complete immutable input for one conditional policy decision."""

    reference_policy: RowSamplingPolicySpec
    candidate_policy: RowSamplingPolicySpec
    fold_scores: Tuple[RowPolicyFoldScore, ...]
    primary_gain_scale: str = "relative"

    def __post_init__(self) -> None:
        if self.reference_policy.name == self.candidate_policy.name:
            raise ValueError("reference and candidate row policies must differ")
        if self.primary_gain_scale not in {"absolute", "relative"}:
            raise ValueError("primary_gain_scale must be absolute or relative")
        object.__setattr__(self, "fold_scores", tuple(self.fold_scores))


@dataclass(frozen=True)
class RowSamplingPolicyEvidence:
    """Paired non-inferiority evidence for capped leverage versus uniform."""

    paired_fold_count: int
    primary_metric: Optional[str]
    mean_gain: Optional[float]
    median_gain: Optional[float]
    confidence_interval: Optional[Tuple[float, float]]
    positive_fold_fraction: Optional[float]
    tail_metric: Optional[str]
    tail_mean_gain: Optional[float]
    tail_median_gain: Optional[float]
    tail_confidence_interval: Optional[Tuple[float, float]]
    class_coverage_preserved: Optional[bool]
    eligible: bool
    rejection_reasons: Tuple[str, ...] = ()

    def summary(self) -> dict[str, object]:
        return {
            "paired_fold_count": self.paired_fold_count,
            "primary_metric": self.primary_metric,
            "mean_gain": self.mean_gain,
            "median_gain": self.median_gain,
            "confidence_interval": self.confidence_interval,
            "positive_fold_fraction": self.positive_fold_fraction,
            "tail_metric": self.tail_metric,
            "tail_mean_gain": self.tail_mean_gain,
            "tail_median_gain": self.tail_median_gain,
            "tail_confidence_interval": self.tail_confidence_interval,
            "class_coverage_preserved": self.class_coverage_preserved,
            "eligible": self.eligible,
            "rejection_reasons": list(self.rejection_reasons),
        }


@dataclass(frozen=True)
class RowSamplingPolicyResult:
    """Selected policy or an explicit fallback to uniform."""

    status: str
    selected_policy: RowSamplingPolicySpec
    reference_policy: RowSamplingPolicySpec
    candidate_policy: RowSamplingPolicySpec
    reason: str
    evidence: RowSamplingPolicyEvidence

    def __post_init__(self) -> None:
        if self.status not in {"selected", "fallback_to_reference"}:
            raise ValueError("status must be selected or fallback_to_reference")
        if (
            self.status == "fallback_to_reference"
            and self.selected_policy != self.reference_policy
        ):
            raise ValueError("fallback_to_reference must select the reference policy")

    def summary(self) -> dict[str, object]:
        return {
            "status": self.status,
            "selected_policy": self.selected_policy.name,
            "selected_method": self.selected_policy.selection_method,
            "reference_policy": self.reference_policy.name,
            "candidate_policy": self.candidate_policy.name,
            "reason": self.reason,
            "evidence": self.evidence.summary(),
        }


@dataclass(frozen=True)
class CrossFittedRowPolicySelectionSpec:
    """Frozen Phase S.1 gates for conditionally enabling capped leverage."""

    min_folds: int = 3
    confidence_level: float = 0.95
    bootstrap_iterations: int = 2_000
    primary_noninferiority_margin: float = 0.005
    min_positive_fold_fraction: float = 2.0 / 3.0
    tail_noninferiority_margin: float = 0.01
    require_class_coverage: bool = True
    random_state: int = 42

    def __post_init__(self) -> None:
        if int(self.min_folds) < 3:
            raise ValueError("min_folds must be at least 3")
        if not 0.0 < float(self.confidence_level) < 1.0:
            raise ValueError("confidence_level must be in (0, 1)")
        if int(self.bootstrap_iterations) < 100:
            raise ValueError("bootstrap_iterations must be at least 100")
        if float(self.primary_noninferiority_margin) < 0.0:
            raise ValueError("primary_noninferiority_margin must be non-negative")
        if not 0.0 < float(self.min_positive_fold_fraction) <= 1.0:
            raise ValueError("min_positive_fold_fraction must be in (0, 1]")
        if float(self.tail_noninferiority_margin) < 0.0:
            raise ValueError("tail_noninferiority_margin must be non-negative")


class CrossFittedRowPolicySelector:
    """Choose capped leverage only when every predeclared guard passes."""

    def __init__(
        self,
        spec: Optional[CrossFittedRowPolicySelectionSpec] = None,
    ) -> None:
        self.spec = spec or CrossFittedRowPolicySelectionSpec()

    def select(self, request: RowSamplingPolicyRequest) -> RowSamplingPolicyResult:
        evidence = self._build_evidence(request)
        if not evidence.eligible:
            return RowSamplingPolicyResult(
                status="fallback_to_reference",
                selected_policy=request.reference_policy,
                reference_policy=request.reference_policy,
                candidate_policy=request.candidate_policy,
                reason="candidate_failed_cross_fitted_row_policy_gates",
                evidence=evidence,
            )
        return RowSamplingPolicyResult(
            status="selected",
            selected_policy=request.candidate_policy,
            reference_policy=request.reference_policy,
            candidate_policy=request.candidate_policy,
            reason="candidate_passed_cross_fitted_row_policy_gates",
            evidence=evidence,
        )

    def _build_evidence(
        self,
        request: RowSamplingPolicyRequest,
    ) -> RowSamplingPolicyEvidence:
        indexed = self._index_scores(request.fold_scores)
        reference = indexed.get(request.reference_policy.name, {})
        candidate = indexed.get(request.candidate_policy.name, {})
        paired_folds = sorted(set(reference) & set(candidate))
        reasons = []
        if set(reference) != set(candidate):
            reasons.append("incomplete_fold_pairs")
        if len(paired_folds) < self.spec.min_folds:
            reasons.append("insufficient_paired_folds")
        if not paired_folds:
            return RowSamplingPolicyEvidence(
                paired_fold_count=0,
                primary_metric=None,
                mean_gain=None,
                median_gain=None,
                confidence_interval=None,
                positive_fold_fraction=None,
                tail_metric=None,
                tail_mean_gain=None,
                tail_median_gain=None,
                tail_confidence_interval=None,
                class_coverage_preserved=None,
                eligible=False,
                rejection_reasons=tuple(reasons),
            )

        primary_gains = []
        tail_gains = []
        coverage_values = []
        primary_metric = None
        tail_metric = None
        for fold_id in paired_folds:
            reference_score = reference[fold_id]
            candidate_score = candidate[fold_id]
            self._validate_pair(reference_score, candidate_score)
            primary_metric = reference_score.primary_metric
            primary_gains.append(
                self._gain(
                    reference_score.primary_value,
                    candidate_score.primary_value,
                    reference_score.primary_direction,
                    request.primary_gain_scale,
                )
            )
            if reference_score.tail_metric is not None:
                tail_metric = reference_score.tail_metric
                tail_gains.append(
                    self._gain(
                        float(reference_score.tail_value),
                        float(candidate_score.tail_value),
                        str(reference_score.tail_direction),
                        "relative",
                    )
                )
            if candidate_score.class_coverage_preserved is not None:
                coverage_values.append(candidate_score.class_coverage_preserved)

        primary = np.asarray(primary_gains, dtype=float)
        primary_interval = self._bootstrap_interval(primary, request, offset=0)
        mean_gain = float(np.mean(primary))
        median_gain = float(np.median(primary))
        positive_fraction = float(np.mean(primary > 0.0))
        margin = float(self.spec.primary_noninferiority_margin)
        if primary_interval[0] < -margin:
            reasons.append("primary_confidence_interval_exceeds_harm_margin")
        if positive_fraction < float(self.spec.min_positive_fold_fraction):
            reasons.append("insufficient_positive_fold_fraction")

        tail_mean = None
        tail_median = None
        tail_interval = None
        if tail_gains:
            tail = np.asarray(tail_gains, dtype=float)
            tail_mean = float(np.mean(tail))
            tail_median = float(np.median(tail))
            tail_interval = self._bootstrap_interval(tail, request, offset=1)
            tail_margin = float(self.spec.tail_noninferiority_margin)
            if tail_mean < -tail_margin:
                reasons.append("tail_mean_exceeds_harm_margin")
            if tail_median < -tail_margin:
                reasons.append("tail_median_exceeds_harm_margin")
            if tail_interval[0] < -tail_margin:
                reasons.append("tail_confidence_interval_exceeds_harm_margin")

        coverage_preserved = (
            bool(all(coverage_values)) if coverage_values else None
        )
        if self.spec.require_class_coverage and coverage_values and not coverage_preserved:
            reasons.append("class_coverage_not_preserved")

        return RowSamplingPolicyEvidence(
            paired_fold_count=len(paired_folds),
            primary_metric=primary_metric,
            mean_gain=mean_gain,
            median_gain=median_gain,
            confidence_interval=primary_interval,
            positive_fold_fraction=positive_fraction,
            tail_metric=tail_metric,
            tail_mean_gain=tail_mean,
            tail_median_gain=tail_median,
            tail_confidence_interval=tail_interval,
            class_coverage_preserved=coverage_preserved,
            eligible=not reasons,
            rejection_reasons=tuple(reasons),
        )

    @staticmethod
    def _index_scores(
        scores: Sequence[RowPolicyFoldScore],
    ) -> dict[str, dict[str, RowPolicyFoldScore]]:
        indexed: dict[str, dict[str, RowPolicyFoldScore]] = {}
        for score in scores:
            by_fold = indexed.setdefault(score.policy_name, {})
            if score.fold_id in by_fold:
                raise ValueError(
                    f"Duplicate row policy fold score for {score.policy_name!r}, "
                    f"fold {score.fold_id!r}"
                )
            by_fold[score.fold_id] = score
        return indexed

    @staticmethod
    def _validate_pair(
        reference: RowPolicyFoldScore,
        candidate: RowPolicyFoldScore,
    ) -> None:
        if (
            reference.primary_metric != candidate.primary_metric
            or reference.primary_direction != candidate.primary_direction
        ):
            raise ValueError("Paired primary row-policy metrics must match")
        reference_has_tail = reference.tail_metric is not None
        candidate_has_tail = candidate.tail_metric is not None
        if reference_has_tail != candidate_has_tail:
            raise ValueError("Paired scores must agree on tail metric availability")
        if reference_has_tail and (
            reference.tail_metric != candidate.tail_metric
            or reference.tail_direction != candidate.tail_direction
        ):
            raise ValueError("Paired tail row-policy metrics must match")

    @staticmethod
    def _gain(
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
        return difference / max(abs(float(reference)), np.finfo(float).eps)

    def _bootstrap_interval(
        self,
        gains: np.ndarray,
        request: RowSamplingPolicyRequest,
        *,
        offset: int,
    ) -> Tuple[float, float]:
        stable_seed = zlib.crc32(
            (
                request.reference_policy.name
                + "::"
                + request.candidate_policy.name
            ).encode("utf-8")
        )
        rng = np.random.default_rng(
            np.random.SeedSequence(
                [int(self.spec.random_state), int(stable_seed), int(offset)]
            )
        )
        indices = rng.integers(
            0,
            gains.size,
            size=(int(self.spec.bootstrap_iterations), gains.size),
        )
        bootstrap_means = np.mean(gains[indices], axis=1)
        alpha = (1.0 - float(self.spec.confidence_level)) / 2.0
        lower, upper = np.quantile(bootstrap_means, (alpha, 1.0 - alpha))
        return float(lower), float(upper)
