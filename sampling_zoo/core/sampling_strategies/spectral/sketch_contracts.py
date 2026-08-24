"""Typed contracts for exact-budget spectral row sketches."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional

import numpy as np


class RowSamplingPolicy(str, Enum):
    """Supported row-selection policies for one spectral partition."""

    ALL = "all"
    UNIFORM = "uniform"
    LEVERAGE = "leverage"
    CAPPED_LEVERAGE = "capped_leverage"
    SATURATED_LEVERAGE = "saturated_leverage"
    ROBUST_LEVERAGE_MIXTURE = "robust_leverage_mixture"
    SATURATED_RIDGE_LEVERAGE = "saturated_ridge_leverage"
    MAXVOL = "maxvol"
    HYBRID = "hybrid"


class TrainingReweighting(str, Enum):
    """How selected rows are weighted during downstream model fitting."""

    NONE = "none"
    INVERSE_PROBABILITY = "inverse_probability"


def _readonly_vector(values: Any, *, dtype: Any) -> np.ndarray:
    array = np.asarray(values, dtype=dtype).reshape(-1).copy()
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class LeverageScoreContract:
    """Raw and policy-adjusted row importance scores."""

    policy: RowSamplingPolicy
    raw_scores: np.ndarray
    effective_scores: np.ndarray
    cap_value: Optional[float] = None
    ridge_lambda: Optional[float] = None

    def __post_init__(self) -> None:
        raw = _readonly_vector(self.raw_scores, dtype=float)
        effective = _readonly_vector(self.effective_scores, dtype=float)
        if raw.size != effective.size:
            raise ValueError("raw_scores and effective_scores must have equal size")
        if raw.size and (
            not np.all(np.isfinite(raw))
            or not np.all(np.isfinite(effective))
            or np.any(raw < 0)
            or np.any(effective < 0)
        ):
            raise ValueError("leverage scores must be finite and non-negative")
        object.__setattr__(self, "raw_scores", raw)
        object.__setattr__(self, "effective_scores", effective)

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy.value,
            "n_scores": int(self.raw_scores.size),
            "raw_min": float(np.min(self.raw_scores)) if self.raw_scores.size else None,
            "raw_max": float(np.max(self.raw_scores)) if self.raw_scores.size else None,
            "effective_min": (
                float(np.min(self.effective_scores))
                if self.effective_scores.size
                else None
            ),
            "effective_max": (
                float(np.max(self.effective_scores))
                if self.effective_scores.size
                else None
            ),
            "cap_value": self.cap_value,
            "ridge_lambda": self.ridge_lambda,
        }


@dataclass(frozen=True)
class RowInclusionProbabilityContract:
    """First-order inclusion probabilities for an exact-size row sketch."""

    probabilities: np.ndarray
    target_size: int

    def __post_init__(self) -> None:
        probabilities = _readonly_vector(self.probabilities, dtype=float)
        target_size = int(self.target_size)
        if target_size < 0 or target_size > probabilities.size:
            raise ValueError("target_size must be in [0, n_candidates]")
        if probabilities.size and (
            not np.all(np.isfinite(probabilities))
            or np.any(probabilities < -1e-12)
            or np.any(probabilities > 1.0 + 1e-12)
        ):
            raise ValueError("inclusion probabilities must be finite and in [0, 1]")
        probabilities = np.clip(probabilities, 0.0, 1.0)
        if not np.isclose(float(np.sum(probabilities)), target_size, atol=1e-8):
            raise ValueError("inclusion probabilities must sum to target_size")
        probabilities.setflags(write=False)
        object.__setattr__(self, "probabilities", probabilities)
        object.__setattr__(self, "target_size", target_size)

    @property
    def saturated_count(self) -> int:
        return int(np.sum(self.probabilities >= 1.0 - 1e-12))

    @property
    def entropy(self) -> float:
        probabilities = self.probabilities
        terms = np.zeros_like(probabilities)
        interior = (probabilities > 0.0) & (probabilities < 1.0)
        p = probabilities[interior]
        terms[interior] = -(p * np.log(p) + (1.0 - p) * np.log1p(-p))
        return float(np.sum(terms))

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_size": int(self.target_size),
            "expected_size": float(np.sum(self.probabilities)),
            "saturated_count": self.saturated_count,
            "min_probability": (
                float(np.min(self.probabilities))
                if self.probabilities.size
                else None
            ),
            "max_probability": (
                float(np.max(self.probabilities))
                if self.probabilities.size
                else None
            ),
            "inclusion_entropy": self.entropy,
        }


@dataclass(frozen=True)
class ExactBudgetSketchPlan:
    """Immutable result of exact dependent rounding for one partition."""

    policy: RowSamplingPolicy
    reweighting: TrainingReweighting
    candidate_indices: np.ndarray
    selected_indices: np.ndarray
    inclusion: RowInclusionProbabilityContract
    selected_probabilities: np.ndarray
    inverse_probability_weights: np.ndarray
    training_weights: np.ndarray
    scores: LeverageScoreContract
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        candidates = _readonly_vector(self.candidate_indices, dtype=int)
        selected = _readonly_vector(self.selected_indices, dtype=int)
        selected_probabilities = _readonly_vector(
            self.selected_probabilities,
            dtype=float,
        )
        inverse_weights = _readonly_vector(
            self.inverse_probability_weights,
            dtype=float,
        )
        training_weights = _readonly_vector(self.training_weights, dtype=float)
        if candidates.size != self.inclusion.probabilities.size:
            raise ValueError("candidate_indices must align with inclusion probabilities")
        if candidates.size != self.scores.raw_scores.size:
            raise ValueError("candidate_indices must align with score contract")
        if selected.size != self.inclusion.target_size:
            raise ValueError("selected_indices must match the exact target size")
        if np.unique(candidates).size != candidates.size:
            raise ValueError("candidate_indices must be unique")
        if np.unique(selected).size != selected.size:
            raise ValueError("selected_indices must be unique")
        if selected.size and not np.all(np.isin(selected, candidates)):
            raise ValueError("selected_indices must be drawn from candidate_indices")
        aligned_sizes = {
            selected.size,
            selected_probabilities.size,
            inverse_weights.size,
            training_weights.size,
        }
        if len(aligned_sizes) != 1:
            raise ValueError("selected probabilities and weights must align")
        if selected.size and (
            np.any(selected_probabilities <= 0.0)
            or not np.all(np.isfinite(inverse_weights))
            or not np.all(np.isfinite(training_weights))
            or np.any(inverse_weights <= 0.0)
            or np.any(training_weights <= 0.0)
        ):
            raise ValueError("selected-row probabilities and weights must be positive")
        object.__setattr__(self, "candidate_indices", candidates)
        object.__setattr__(self, "selected_indices", selected)
        object.__setattr__(self, "selected_probabilities", selected_probabilities)
        object.__setattr__(self, "inverse_probability_weights", inverse_weights)
        object.__setattr__(self, "training_weights", training_weights)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def exact_budget(self) -> bool:
        return self.selected_indices.size == self.inclusion.target_size

    def to_dict(self, *, include_arrays: bool = False) -> dict[str, Any]:
        payload = {
            "policy": self.policy.value,
            "reweighting": self.reweighting.value,
            "n_candidates": int(self.candidate_indices.size),
            "selected_size": int(self.selected_indices.size),
            "exact_budget": self.exact_budget,
            "effective_sample_size": float(
                np.square(np.sum(self.training_weights))
                / max(float(np.sum(np.square(self.training_weights))), 1e-12)
            ) if self.training_weights.size else 0.0,
            "weight_min": (
                float(np.min(self.training_weights))
                if self.training_weights.size
                else None
            ),
            "weight_max": (
                float(np.max(self.training_weights))
                if self.training_weights.size
                else None
            ),
            "inclusion": self.inclusion.to_dict(),
            "scores": self.scores.to_dict(),
            "metadata": dict(self.metadata),
        }
        if include_arrays:
            payload.update({
                "candidate_indices": self.candidate_indices.tolist(),
                "selected_indices": self.selected_indices.tolist(),
                "inclusion_probabilities": self.inclusion.probabilities.tolist(),
                "selected_probabilities": self.selected_probabilities.tolist(),
                "inverse_probability_weights": (
                    self.inverse_probability_weights.tolist()
                ),
                "training_weights": self.training_weights.tolist(),
            })
        return payload


@dataclass(frozen=True)
class SubspacePreservationContract:
    """How accurately a row sketch preserves the source matrix geometry."""

    rank: int
    gram_relative_error: float
    projection_cost_relative_error: float
    max_principal_angle_degrees: float
    mean_principal_angle_degrees: float

    def __post_init__(self) -> None:
        if int(self.rank) < 1:
            raise ValueError("rank must be positive")
        values = (
            self.gram_relative_error,
            self.projection_cost_relative_error,
            self.max_principal_angle_degrees,
            self.mean_principal_angle_degrees,
        )
        if not all(np.isfinite(float(value)) and float(value) >= 0.0 for value in values):
            raise ValueError("subspace diagnostics must be finite and non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": int(self.rank),
            "gram_relative_error": float(self.gram_relative_error),
            "projection_cost_relative_error": float(
                self.projection_cost_relative_error
            ),
            "max_principal_angle_degrees": float(
                self.max_principal_angle_degrees
            ),
            "mean_principal_angle_degrees": float(
                self.mean_principal_angle_degrees
            ),
        }
