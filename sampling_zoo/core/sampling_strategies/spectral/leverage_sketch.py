"""Pure exact-budget planning for leverage-preserving row sketches."""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

from .sketch_contracts import (
    ExactBudgetSketchPlan,
    LeverageScoreContract,
    RowInclusionProbabilityContract,
    RowSamplingPolicy,
    SubspacePreservationContract,
    TrainingReweighting,
)


def normalize_nonnegative_scores(scores: Sequence[float] | np.ndarray) -> np.ndarray:
    """Return a finite probability vector, with a uniform zero-score fallback."""

    values = np.asarray(scores, dtype=float).reshape(-1)
    values = np.maximum(
        np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0),
        0.0,
    )
    if values.size == 0:
        return values
    total = float(np.sum(values))
    if total <= np.finfo(float).eps:
        return np.full(values.size, 1.0 / values.size)
    return values / total


def saturated_inclusion_probabilities(
    weights: Sequence[float] | np.ndarray,
    target_size: int,
) -> np.ndarray:
    """Solve ``pi_i = min(1, c * w_i)`` with ``sum(pi) = target_size``."""

    normalized = normalize_nonnegative_scores(weights)
    target_size = max(0, min(int(target_size), normalized.size))
    if target_size == 0:
        return np.zeros(normalized.size, dtype=float)
    if target_size == normalized.size:
        return np.ones(normalized.size, dtype=float)

    positive = normalized > 0.0
    if int(np.sum(positive)) < target_size:
        floor = np.finfo(float).eps
        normalized = normalize_nonnegative_scores(np.maximum(normalized, floor))

    lower = 0.0
    upper = 1.0
    while float(np.sum(np.minimum(1.0, upper * normalized))) < target_size:
        upper *= 2.0
    for _ in range(100):
        scale = 0.5 * (lower + upper)
        current = float(np.sum(np.minimum(1.0, scale * normalized)))
        if current < target_size:
            lower = scale
        else:
            upper = scale
    probabilities = np.minimum(1.0, upper * normalized)
    residual = float(target_size - np.sum(probabilities))
    if abs(residual) > 1e-10:
        fractional = np.flatnonzero(
            (probabilities > 1e-12) & (probabilities < 1.0 - 1e-12)
        )
        if fractional.size:
            probabilities[int(fractional[-1])] += residual
    return np.clip(probabilities, 0.0, 1.0)


def dependent_round_exact(
    probabilities: Sequence[float] | np.ndarray,
    *,
    random_state: int | np.random.Generator | None = None,
) -> np.ndarray:
    """Dependent-round first-order marginals to an exact-size unique sample."""

    original = np.asarray(probabilities, dtype=float).reshape(-1)
    if original.size == 0:
        return np.asarray([], dtype=int)
    if not np.all(np.isfinite(original)) or np.any(original < 0) or np.any(original > 1):
        raise ValueError("probabilities must be finite and in [0, 1]")
    target_size = int(round(float(np.sum(original))))
    if not np.isclose(float(np.sum(original)), target_size, atol=1e-8):
        raise ValueError("probabilities must sum to an integer exact budget")
    rng = (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )
    rounded = original.copy()
    tolerance = 1e-12
    while True:
        fractional = np.flatnonzero(
            (rounded > tolerance) & (rounded < 1.0 - tolerance)
        )
        if fractional.size < 2:
            break
        first, second = map(int, fractional[:2])
        alpha = min(1.0 - rounded[first], rounded[second])
        beta = min(rounded[first], 1.0 - rounded[second])
        denominator = alpha + beta
        if denominator <= tolerance:
            break
        if rng.random() < beta / denominator:
            rounded[first] += alpha
            rounded[second] -= alpha
        else:
            rounded[first] -= beta
            rounded[second] += beta
        rounded[np.abs(rounded) <= tolerance] = 0.0
        rounded[np.abs(rounded - 1.0) <= tolerance] = 1.0

    selected = np.flatnonzero(rounded >= 0.5)
    if selected.size != target_size:
        order = np.argsort(-rounded, kind="mergesort")
        selected = np.sort(order[:target_size])
    return selected.astype(int, copy=False)


def _score_contract(
    policy: RowSamplingPolicy,
    raw_scores: np.ndarray,
    *,
    cap_quantile: float,
    mixture_alpha: float,
    uniform_floor: float,
    ridge_scores: Optional[np.ndarray],
    ridge_lambda: Optional[float],
) -> LeverageScoreContract:
    raw = np.maximum(
        np.nan_to_num(np.asarray(raw_scores, dtype=float), nan=0.0),
        0.0,
    )
    cap_value: Optional[float] = None
    if policy is RowSamplingPolicy.UNIFORM:
        effective = np.ones_like(raw)
    elif policy is RowSamplingPolicy.CAPPED_LEVERAGE:
        if not 0.0 < float(cap_quantile) <= 1.0:
            raise ValueError("leverage_cap_quantile must be in (0, 1]")
        cap_value = float(np.quantile(raw, float(cap_quantile)))
        effective = np.minimum(raw, cap_value)
    elif policy is RowSamplingPolicy.ROBUST_LEVERAGE_MIXTURE:
        alpha = float(mixture_alpha)
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("leverage_mixture_alpha must be in [0, 1]")
        base = normalize_nonnegative_scores(raw)
        effective = (1.0 - alpha) * base + alpha / max(raw.size, 1)
    elif policy is RowSamplingPolicy.SATURATED_RIDGE_LEVERAGE:
        if ridge_scores is None:
            raise ValueError("ridge_scores are required for saturated_ridge_leverage")
        effective = np.asarray(ridge_scores, dtype=float).reshape(-1)
        if effective.size != raw.size:
            raise ValueError("ridge_scores must align with leverage scores")
    else:
        effective = raw.copy()
    if effective.size:
        effective = np.maximum(effective, float(uniform_floor))
    return LeverageScoreContract(
        policy=policy,
        raw_scores=raw,
        effective_scores=effective,
        cap_value=cap_value,
        ridge_lambda=ridge_lambda,
    )


def build_exact_budget_sketch_plan(
    candidate_indices: Sequence[int] | np.ndarray,
    *,
    target_size: int,
    policy: str | RowSamplingPolicy,
    scores: Sequence[float] | np.ndarray,
    random_state: int | np.random.Generator | None = None,
    leverage_cap_quantile: float = 0.95,
    leverage_mixture_alpha: float = 0.25,
    leverage_uniform_floor: float = 1e-12,
    ridge_scores: Optional[Sequence[float] | np.ndarray] = None,
    ridge_lambda: Optional[float] = None,
    reweighting: str | TrainingReweighting = TrainingReweighting.NONE,
    metadata: Optional[dict[str, Any]] = None,
) -> ExactBudgetSketchPlan:
    """Build an exact-size sketch and retain all first-order sampling data."""

    candidates = np.asarray(candidate_indices, dtype=int).reshape(-1)
    if np.unique(candidates).size != candidates.size:
        raise ValueError("candidate_indices must be unique")
    target_size = max(0, min(int(target_size), candidates.size))
    resolved_policy = (
        policy if isinstance(policy, RowSamplingPolicy) else RowSamplingPolicy(str(policy))
    )
    resolved_reweighting = (
        reweighting
        if isinstance(reweighting, TrainingReweighting)
        else TrainingReweighting(str(reweighting))
    )
    score_values = np.asarray(scores, dtype=float).reshape(-1)
    if candidates.size and score_values.size <= int(np.max(candidates)):
        raise ValueError("scores must align with candidate indices")
    local_scores = score_values[candidates] if candidates.size else np.asarray([], dtype=float)
    local_ridge = None
    if ridge_scores is not None:
        ridge_values = np.asarray(ridge_scores, dtype=float).reshape(-1)
        if candidates.size and ridge_values.size <= int(np.max(candidates)):
            raise ValueError("ridge_scores must align with candidate indices")
        local_ridge = ridge_values[candidates]
    score_contract = _score_contract(
        resolved_policy,
        local_scores,
        cap_quantile=leverage_cap_quantile,
        mixture_alpha=leverage_mixture_alpha,
        uniform_floor=leverage_uniform_floor,
        ridge_scores=local_ridge,
        ridge_lambda=ridge_lambda,
    )

    if resolved_policy is RowSamplingPolicy.ALL:
        selected_local = np.arange(target_size, dtype=int)
        probabilities = np.zeros(candidates.size, dtype=float)
        probabilities[selected_local] = 1.0
    elif resolved_policy is RowSamplingPolicy.LEVERAGE:
        order = np.argsort(-score_contract.effective_scores, kind="mergesort")
        selected_local = np.sort(order[:target_size])
        probabilities = np.zeros(candidates.size, dtype=float)
        probabilities[selected_local] = 1.0
    else:
        probabilities = saturated_inclusion_probabilities(
            score_contract.effective_scores,
            target_size,
        )
        selected_local = dependent_round_exact(
            probabilities,
            random_state=random_state,
        )

    inclusion = RowInclusionProbabilityContract(
        probabilities=probabilities,
        target_size=target_size,
    )
    selected = candidates[selected_local]
    selected_probabilities = probabilities[selected_local]
    inverse = np.divide(
        1.0,
        selected_probabilities,
        out=np.ones_like(selected_probabilities),
        where=selected_probabilities > 0.0,
    )
    if resolved_reweighting is TrainingReweighting.INVERSE_PROBABILITY and inverse.size:
        training_weights = inverse / float(np.mean(inverse))
    else:
        training_weights = np.ones(selected.size, dtype=float)
    return ExactBudgetSketchPlan(
        policy=resolved_policy,
        reweighting=resolved_reweighting,
        candidate_indices=candidates,
        selected_indices=selected,
        inclusion=inclusion,
        selected_probabilities=selected_probabilities,
        inverse_probability_weights=inverse,
        training_weights=training_weights,
        scores=score_contract,
        metadata={} if metadata is None else metadata,
    )


def build_deterministic_sketch_plan(
    candidate_indices: Sequence[int] | np.ndarray,
    selected_indices: Sequence[int] | np.ndarray,
    *,
    policy: str | RowSamplingPolicy,
    scores: Sequence[float] | np.ndarray,
    metadata: Optional[dict[str, Any]] = None,
) -> ExactBudgetSketchPlan:
    """Wrap max-volume and hybrid selections in the common sketch contract."""

    candidates = np.asarray(candidate_indices, dtype=int).reshape(-1)
    selected = np.asarray(selected_indices, dtype=int).reshape(-1)
    resolved_policy = (
        policy if isinstance(policy, RowSamplingPolicy) else RowSamplingPolicy(str(policy))
    )
    score_values = np.asarray(scores, dtype=float).reshape(-1)
    if candidates.size and score_values.size <= int(np.max(candidates)):
        raise ValueError("scores must align with candidate indices")
    local_scores = score_values[candidates] if candidates.size else np.asarray([], dtype=float)
    selected_mask = np.isin(candidates, selected)
    probabilities = selected_mask.astype(float)
    score_contract = LeverageScoreContract(
        policy=resolved_policy,
        raw_scores=local_scores,
        effective_scores=local_scores,
    )
    return ExactBudgetSketchPlan(
        policy=resolved_policy,
        reweighting=TrainingReweighting.NONE,
        candidate_indices=candidates,
        selected_indices=selected,
        inclusion=RowInclusionProbabilityContract(
            probabilities=probabilities,
            target_size=selected.size,
        ),
        selected_probabilities=np.ones(selected.size, dtype=float),
        inverse_probability_weights=np.ones(selected.size, dtype=float),
        training_weights=np.ones(selected.size, dtype=float),
        scores=score_contract,
        metadata={} if metadata is None else metadata,
    )


def combine_stratified_sketch_plans(
    candidate_indices: Sequence[int] | np.ndarray,
    plans: Sequence[ExactBudgetSketchPlan],
    *,
    scores: Sequence[float] | np.ndarray,
) -> ExactBudgetSketchPlan:
    """Combine disjoint class-level sketches into one partition-level contract."""

    candidates = np.asarray(candidate_indices, dtype=int).reshape(-1)
    plans = tuple(plans)
    if not plans:
        raise ValueError("at least one stratum sketch plan is required")
    policies = {plan.policy for plan in plans}
    reweighting = {plan.reweighting for plan in plans}
    if len(policies) != 1 or len(reweighting) != 1:
        raise ValueError("stratified sketch plans must share policy and reweighting")
    covered = np.concatenate([plan.candidate_indices for plan in plans])
    if np.unique(covered).size != covered.size or set(covered) != set(candidates):
        raise ValueError("stratum candidates must form a disjoint partition")

    local_positions = {
        int(index): position for position, index in enumerate(candidates.tolist())
    }
    probabilities = np.zeros(candidates.size, dtype=float)
    effective_scores = np.zeros(candidates.size, dtype=float)
    for plan in plans:
        positions = np.asarray(
            [local_positions[int(index)] for index in plan.candidate_indices],
            dtype=int,
        )
        probabilities[positions] = plan.inclusion.probabilities
        effective_scores[positions] = plan.scores.effective_scores
    selected = np.concatenate([plan.selected_indices for plan in plans])
    selected_probabilities = np.concatenate(
        [plan.selected_probabilities for plan in plans]
    )
    inverse_weights = np.concatenate(
        [plan.inverse_probability_weights for plan in plans]
    )
    training_weights = np.concatenate([plan.training_weights for plan in plans])
    score_values = np.asarray(scores, dtype=float).reshape(-1)
    raw_scores = score_values[candidates]
    return ExactBudgetSketchPlan(
        policy=next(iter(policies)),
        reweighting=next(iter(reweighting)),
        candidate_indices=candidates,
        selected_indices=selected,
        inclusion=RowInclusionProbabilityContract(
            probabilities=probabilities,
            target_size=selected.size,
        ),
        selected_probabilities=selected_probabilities,
        inverse_probability_weights=inverse_weights,
        training_weights=training_weights,
        scores=LeverageScoreContract(
            policy=next(iter(policies)),
            raw_scores=raw_scores,
            effective_scores=effective_scores,
        ),
        metadata={"stratified": True, "n_strata": len(plans)},
    )


def evaluate_subspace_preservation(
    matrix: np.ndarray,
    plan: ExactBudgetSketchPlan,
    *,
    rank: int,
) -> SubspacePreservationContract:
    """Compare the inverse-probability row sketch with the source Gram geometry."""

    source = np.asarray(matrix, dtype=float)
    if source.ndim != 2 or source.shape[0] != plan.candidate_indices.size:
        raise ValueError("matrix rows must align with the sketch candidates")
    if not np.all(np.isfinite(source)):
        raise ValueError("matrix must contain only finite values")
    rank = max(1, min(int(rank), min(source.shape)))
    local_positions = {
        int(index): position
        for position, index in enumerate(plan.candidate_indices.tolist())
    }
    selected_local = np.asarray(
        [local_positions[int(index)] for index in plan.selected_indices],
        dtype=int,
    )
    selected_matrix = source[selected_local]
    weighted = selected_matrix * np.sqrt(
        plan.inverse_probability_weights
    ).reshape(-1, 1)

    source_gram = source.T @ source
    sketch_gram = weighted.T @ weighted
    denominator = max(float(np.linalg.norm(source_gram, ord="fro")), 1e-12)
    gram_error = float(np.linalg.norm(sketch_gram - source_gram, ord="fro") / denominator)

    _source_u, _source_s, source_vt = np.linalg.svd(source, full_matrices=False)
    _sketch_u, _sketch_s, sketch_vt = np.linalg.svd(weighted, full_matrices=False)
    source_basis = source_vt[:rank].T
    sketch_rank = min(rank, sketch_vt.shape[0])
    sketch_basis = sketch_vt[:sketch_rank].T
    common_rank = min(source_basis.shape[1], sketch_basis.shape[1])
    singular = np.linalg.svd(
        source_basis[:, :common_rank].T @ sketch_basis[:, :common_rank],
        compute_uv=False,
    )
    angles = np.rad2deg(np.arccos(np.clip(singular, 0.0, 1.0)))

    source_residual = source - source @ source_basis @ source_basis.T
    sketch_residual = source - source @ sketch_basis @ sketch_basis.T
    source_cost = float(np.sum(source_residual * source_residual))
    sketch_cost = float(np.sum(sketch_residual * sketch_residual))
    projection_error = abs(sketch_cost - source_cost) / max(source_cost, 1e-12)
    return SubspacePreservationContract(
        rank=common_rank,
        gram_relative_error=gram_error,
        projection_cost_relative_error=float(projection_error),
        max_principal_angle_degrees=float(np.max(angles)) if angles.size else 0.0,
        mean_principal_angle_degrees=float(np.mean(angles)) if angles.size else 0.0,
    )
