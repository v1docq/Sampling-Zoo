"""Shared row-selection primitives for spectral partitions."""

from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np


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
) -> np.ndarray:
    indices = np.asarray(candidate_indices, dtype=int)
    target_size = max(0, min(int(target_size), indices.size))
    if target_size == 0:
        return np.asarray([], dtype=int)
    if indices.size <= target_size:
        return indices.copy()
    method = str(selection_method).strip().lower()
    if method == "all":
        return indices[:target_size].copy()
    if method == "leverage":
        return _top_score_indices(indices, scores, target_size)
    if method == "capped_leverage":
        return _capped_leverage_indices(
            indices,
            scores,
            target_size,
            random_state=random_state,
            cap_quantile=leverage_cap_quantile,
        )
    if method == "maxvol":
        return greedy_maxvol_indices(indices, target_size, embedding)
    if method != "hybrid":
        raise ValueError(
            "selection_method must be all, leverage, capped_leverage, maxvol, or hybrid"
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
    return np.asarray(picked[:target_size], dtype=int)


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


def _capped_leverage_indices(
    indices: np.ndarray,
    scores: np.ndarray,
    target_size: int,
    *,
    random_state: int | np.random.Generator | None,
    cap_quantile: float,
) -> np.ndarray:
    if not 0 < float(cap_quantile) <= 1:
        raise ValueError("leverage_cap_quantile must be in (0, 1]")
    score_values = np.asarray(scores, dtype=float)
    if score_values.ndim != 1 or score_values.size <= int(np.max(indices)):
        raise ValueError("scores must align with candidate indices")

    local_scores = np.nan_to_num(
        score_values[indices],
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    local_scores = np.maximum(local_scores, 0.0)
    cap = float(np.quantile(local_scores, float(cap_quantile)))
    weights = np.minimum(local_scores, cap)
    weight_sum = float(np.sum(weights))
    probabilities = (
        None if weight_sum <= np.finfo(float).eps else weights / weight_sum
    )
    rng = (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )
    picked_local = rng.choice(
        indices.size,
        size=target_size,
        replace=False,
        p=probabilities,
    )
    return indices[np.asarray(picked_local, dtype=int)]


def _orthonormal_basis(rows: np.ndarray) -> np.ndarray:
    if rows.size == 0:
        return np.empty((0, 0))
    basis, _ = np.linalg.qr(rows.T, mode="reduced")
    return basis.T
