"""Shared row-selection primitives for spectral partitions."""

from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np

from .leverage_sketch import (
    build_deterministic_sketch_plan,
    build_exact_budget_sketch_plan,
)
from .sketch_contracts import ExactBudgetSketchPlan, RowSamplingPolicy


def partition_membership_fingerprint(
    labels: Sequence[int] | np.ndarray,
) -> str:
    """Return a label-permutation-invariant fingerprint of row memberships."""

    values = np.asarray(labels).reshape(-1)
    canonical_ids: dict[object, int] = {}
    canonical = np.empty(values.size, dtype=np.int64)
    for index, value in enumerate(values.tolist()):
        canonical[index] = canonical_ids.setdefault(value, len(canonical_ids))
    return hashlib.sha256(canonical.astype("<i8", copy=False).tobytes()).hexdigest()


def select_partition_indices(
    candidate_indices: Sequence[int] | np.ndarray,
    *,
    target_size: int,
    selection_method: str,
    scores: np.ndarray,
    embedding: np.ndarray,
    random_state: int | np.random.Generator | None = None,
    leverage_cap_quantile: float = 0.95,
    leverage_mixture_alpha: float = 0.25,
    leverage_uniform_floor: float = 1e-12,
    ridge_scores: np.ndarray | None = None,
    ridge_lambda: float | None = None,
    training_reweighting: str = "none",
) -> np.ndarray:
    """Return selected indices while retaining the legacy array interface."""

    return build_partition_sketch_plan(
        candidate_indices,
        target_size=target_size,
        selection_method=selection_method,
        scores=scores,
        embedding=embedding,
        random_state=random_state,
        leverage_cap_quantile=leverage_cap_quantile,
        leverage_mixture_alpha=leverage_mixture_alpha,
        leverage_uniform_floor=leverage_uniform_floor,
        ridge_scores=ridge_scores,
        ridge_lambda=ridge_lambda,
        training_reweighting=training_reweighting,
    ).selected_indices.copy()


def build_partition_sketch_plan(
    candidate_indices: Sequence[int] | np.ndarray,
    *,
    target_size: int,
    selection_method: str,
    scores: np.ndarray,
    embedding: np.ndarray,
    random_state: int | np.random.Generator | None = None,
    leverage_cap_quantile: float = 0.95,
    leverage_mixture_alpha: float = 0.25,
    leverage_uniform_floor: float = 1e-12,
    ridge_scores: np.ndarray | None = None,
    ridge_lambda: float | None = None,
    training_reweighting: str = "none",
) -> ExactBudgetSketchPlan:
    """Plan one exact-size row sketch without sampler or model side effects."""

    indices = np.asarray(candidate_indices, dtype=int)
    target_size = max(0, min(int(target_size), indices.size))
    method = str(selection_method).strip().lower()
    exact_policies = {
        RowSamplingPolicy.ALL.value,
        RowSamplingPolicy.UNIFORM.value,
        RowSamplingPolicy.LEVERAGE.value,
        RowSamplingPolicy.CAPPED_LEVERAGE.value,
        RowSamplingPolicy.SATURATED_LEVERAGE.value,
        RowSamplingPolicy.ROBUST_LEVERAGE_MIXTURE.value,
        RowSamplingPolicy.SATURATED_RIDGE_LEVERAGE.value,
    }
    if method in exact_policies:
        return build_exact_budget_sketch_plan(
            indices,
            target_size=target_size,
            policy=method,
            scores=scores,
            random_state=random_state,
            leverage_cap_quantile=leverage_cap_quantile,
            leverage_mixture_alpha=leverage_mixture_alpha,
            leverage_uniform_floor=leverage_uniform_floor,
            ridge_scores=ridge_scores,
            ridge_lambda=ridge_lambda,
            reweighting=training_reweighting,
        )
    if method == "maxvol":
        selected = greedy_maxvol_indices(indices, target_size, embedding)
        return build_deterministic_sketch_plan(
            indices,
            selected,
            policy=method,
            scores=scores,
        )
    if method != "hybrid":
        raise ValueError(
            "selection_method must be one of: "
            + ", ".join(sorted(exact_policies | {"maxvol", "hybrid"}))
        )

    leverage_size = max(1, target_size // 2)
    picked = _top_score_indices(indices, scores, leverage_size).tolist()
    remaining = np.setdiff1d(indices, np.asarray(picked, dtype=int), assume_unique=False)
    if len(picked) < target_size and remaining.size:
        extra = greedy_maxvol_indices(
            remaining,
            target_size - len(picked),
            embedding,
        )
        picked.extend(extra.tolist())
    selected = np.asarray(picked[:target_size], dtype=int)
    return build_deterministic_sketch_plan(
        indices,
        selected,
        policy=method,
        scores=scores,
    )


def greedy_maxvol_indices(
    candidate_indices: Sequence[int] | np.ndarray,
    target_size: int,
    embedding: np.ndarray,
) -> np.ndarray:
    indices = np.asarray(candidate_indices, dtype=int)
    target_size = max(0, min(int(target_size), indices.size))
    if indices.size <= target_size:
        return indices.copy()
    rows = np.asarray(embedding, dtype=float)[indices]
    first = int(np.argmax(np.sum(rows * rows, axis=1)))
    picked_local = [first]
    basis = _orthonormal_basis(rows[[first]])
    while len(picked_local) < target_size:
        projection = rows @ basis.T if basis.size else 0.0
        residual = rows - projection @ basis if basis.size else rows
        residual_norms = np.sum(residual * residual, axis=1)
        residual_norms[picked_local] = -np.inf
        next_index = int(np.argmax(residual_norms))
        if not np.isfinite(residual_norms[next_index]):
            break
        picked_local.append(next_index)
        basis = _orthonormal_basis(rows[picked_local])
    return indices[np.asarray(picked_local, dtype=int)]


def _top_score_indices(
    indices: np.ndarray,
    scores: np.ndarray,
    target_size: int,
) -> np.ndarray:
    score_values = np.asarray(scores, dtype=float)
    if score_values.ndim != 1 or score_values.size <= int(np.max(indices)):
        raise ValueError("scores must align with candidate indices")
    order = np.argsort(-score_values[indices], kind="mergesort")
    return indices[order[:target_size]].copy()


def _orthonormal_basis(rows: np.ndarray) -> np.ndarray:
    if rows.size == 0:
        return np.empty((0, 0))
    basis, _ = np.linalg.qr(rows.T, mode="reduced")
    return basis.T
