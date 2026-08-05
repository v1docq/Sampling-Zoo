from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Optional

import numpy as np
from sklearn.utils.extmath import randomized_svd


@dataclass(frozen=True)
class RMTSpectralBasis:
    """SVD factors and leverage scores returned by RMT linear algebra backends."""

    U: np.ndarray
    singular_values: np.ndarray
    Vt: np.ndarray
    leverage_scores: np.ndarray

    def truncate(self, rank: int) -> "RMTSpectralBasis":
        rank = max(1, min(int(rank), self.singular_values.shape[0]))
        U = self.U[:, :rank]
        singular_values = self.singular_values[:rank]
        Vt = self.Vt[:rank, :]
        return RMTSpectralBasis(
            U=U,
            singular_values=singular_values,
            Vt=Vt,
            leverage_scores=compute_leverage_scores(U),
        )


@dataclass(frozen=True)
class RMTSubspaceComparison:
    """Backend result for one equal-rank subspace comparison."""

    rank: int
    principal_angles_degrees: tuple[float, ...]
    mean_principal_angle_degrees: float
    max_principal_angle_degrees: float
    projection_distance: float
    normalized_projection_distance: float
    min_canonical_correlation: float


def compute_leverage_scores(U: np.ndarray) -> np.ndarray:
    scores = np.sum(U * U, axis=1)
    score_sum = float(np.sum(scores))
    if not np.isfinite(score_sum) or score_sum <= 0:
        return np.full(U.shape[0], 1.0 / U.shape[0])
    return scores / score_sum


class RandomizedSVDBackend:
    def __init__(self, oversample_factor=0, power_iterations=0, random_state=None):
        self.oversample_factor = oversample_factor 
        self.power_iterations = power_iterations   
        self.random_state = random_state

    def compute_basis(self, X, rank, **kwargs):
        """
        Вычисляет приближённый SVD с oversampling и power iterations. В данный момент реализован только randomized SVD из sklearn.
        Arguments:
        X: входная матрица  
        rank: целевой ранг (int или float в (0,1], обозначающий долю от min(n_samples, n_features))
        **kwargs: дополнительные параметры для randomized_svd
        Returns:
        Q: approximate range basis (ортогональная матрица, аппроксимирующая X)
        S: singular values
        Vt: right singular vectors
        """
        if isinstance(rank, float):
            rank = int(rank * min(X.shape))

        n_components = min(rank + self.oversample_factor, X.shape[1])  # Oversampling
        
        # Randomized SVD с power iterations
        U, S, Vt = randomized_svd(
            X,
            n_components=n_components,
            n_iter=self.power_iterations,
            random_state=self.random_state,
            **kwargs
        )
        
        Q = U[:, :rank] 
        S = S[:rank]
        Vt = Vt[:rank, :]
        
        return Q, S, Vt


class MatrixRMTBackend:
    """NumPy/sklearn backend for RMT contraction linear algebra."""

    def __init__(
        self,
        oversample_factor: int = 10,
        power_iterations: int = 2,
        random_state: Optional[int] = None,
    ) -> None:
        self.oversample_factor = int(oversample_factor)
        self.power_iterations = int(power_iterations)
        self.random_state = random_state

    def build_mode0_unfolding(self, X: np.ndarray, view_specs: Iterable[Any]) -> np.ndarray:
        views: list[np.ndarray] = []
        for spec in view_specs:
            base = X if spec.columns is None else X[:, spec.columns]
            Z = base if spec.projection is None else base @ spec.projection
            if Z.shape[1] != spec.output_dim:
                raise RuntimeError("View output dimension mismatch")
            views.append(Z)
        if not views:
            raise RuntimeError("No random contraction views were generated")
        return np.concatenate(views, axis=1)

    def compute_spectral_basis(self, M: np.ndarray, rank: int) -> RMTSpectralBasis:
        n_samples, n_features = M.shape
        n_components = min(n_samples, n_features, int(rank) + self.oversample_factor)
        if n_components < 1:
            raise ValueError("Cannot compute randomized SVD: empty transformed matrix")
        U, S, Vt = randomized_svd(
            M,
            n_components=n_components,
            n_iter=self.power_iterations,
            random_state=self.random_state,
        )
        U = U[:, :rank]
        S = S[:rank]
        Vt = Vt[:rank, :]
        return RMTSpectralBasis(
            U=U,
            singular_values=S,
            Vt=Vt,
            leverage_scores=compute_leverage_scores(U),
        )

    @staticmethod
    def compare_subspace_prefixes(
        reference_basis: np.ndarray,
        candidate_basis: np.ndarray,
        max_rank: int,
    ) -> tuple[RMTSubspaceComparison, ...]:
        reference, candidate, max_rank = _validate_subspace_inputs(
            reference_basis,
            candidate_basis,
            max_rank,
        )
        return tuple(
            _compare_matrix_subspaces(
                reference[:, :rank],
                candidate[:, :rank],
                rank,
            )
            for rank in range(1, max_rank + 1)
        )

    @staticmethod
    def project_new_unfolding(
        M_new: np.ndarray,
        right_basis: np.ndarray,
        singular_values: np.ndarray,
    ) -> np.ndarray:
        embedding = M_new @ right_basis.T
        return embedding / np.maximum(singular_values, 1e-12)

    @staticmethod
    def routing_probability(
        embedding: np.ndarray,
        active_centroids: np.ndarray,
        temperature: float,
    ) -> np.ndarray:
        d2 = MatrixRMTBackend.pairwise_squared_euclidean(embedding, active_centroids)
        return MatrixRMTBackend.normalize_routing_values(
            d2,
            kind="distance",
            temperature=temperature,
        )

    @staticmethod
    def pairwise_squared_euclidean(
        embedding: np.ndarray,
        centers: np.ndarray,
    ) -> np.ndarray:
        embedding = np.asarray(embedding, dtype=float)
        centers = np.asarray(centers, dtype=float)
        return np.sum((embedding[:, None, :] - centers[None, :, :]) ** 2, axis=2)

    @staticmethod
    def pairwise_mahalanobis(
        embedding: np.ndarray,
        centers: np.ndarray,
        precisions: np.ndarray,
    ) -> np.ndarray:
        embedding = np.asarray(embedding, dtype=float)
        centers = np.asarray(centers, dtype=float)
        precisions = np.asarray(precisions, dtype=float)
        diff = embedding[:, None, :] - centers[None, :, :]
        return np.einsum("nkd,kde,nke->nk", diff, precisions, diff, optimize=True)

    @staticmethod
    def pairwise_cosine_distance(
        embedding: np.ndarray,
        centers: np.ndarray,
    ) -> np.ndarray:
        embedding = np.asarray(embedding, dtype=float)
        centers = np.asarray(centers, dtype=float)
        numerator = embedding @ centers.T
        denominator = (
            np.linalg.norm(embedding, axis=1, keepdims=True)
            * np.linalg.norm(centers, axis=1, keepdims=True).T
        )
        similarity = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator, dtype=float),
            where=denominator > 1e-12,
        )
        return np.clip(1.0 - similarity, 0.0, 2.0)

    @classmethod
    def compute_routing_values(
        cls,
        embedding: np.ndarray,
        centers: np.ndarray,
        *,
        metric: str,
        kernel: str,
        scales: Optional[np.ndarray] = None,
        precisions: Optional[np.ndarray] = None,
        log_determinants: Optional[np.ndarray] = None,
        priors: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if kernel == "gmm_posterior":
            if precisions is None or log_determinants is None or priors is None:
                raise ValueError("gmm_posterior requires precisions, log determinants, and priors")
            d2 = cls.pairwise_mahalanobis(embedding, centers, precisions)
            dimension = int(np.asarray(embedding).shape[1])
            return (
                -0.5
                * (
                    d2
                    + np.asarray(log_determinants, dtype=float)[None, :]
                    + dimension * np.log(2.0 * np.pi)
                )
                + np.log(np.maximum(np.asarray(priors, dtype=float), 1e-12))[None, :]
            )
        if metric == "squared_euclidean":
            return cls.pairwise_squared_euclidean(embedding, centers)
        if metric == "median_scaled_euclidean":
            if scales is None:
                raise ValueError("median_scaled_euclidean requires partition scales")
            d2 = cls.pairwise_squared_euclidean(embedding, centers)
            return d2 / np.maximum(np.asarray(scales, dtype=float)[None, :], 1e-12)
        if metric in {"diag_shrinkage_mahalanobis", "full_shrinkage_mahalanobis"}:
            if precisions is None:
                raise ValueError(f"{metric} requires precision matrices")
            return cls.pairwise_mahalanobis(embedding, centers, precisions)
        if metric == "cosine":
            return cls.pairwise_cosine_distance(embedding, centers)
        raise ValueError(f"Unsupported routing metric: {metric}")

    @staticmethod
    def normalize_routing_values(
        values: np.ndarray,
        *,
        kind: str,
        temperature: float,
    ) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        if values.ndim != 2 or values.shape[1] < 1:
            raise ValueError("Routing values must be a 2D matrix with at least one partition")
        temp = max(float(temperature), 1e-8)
        if kind == "distance":
            logits = -values / temp
        elif kind == "log_density":
            logits = values / temp
        else:
            raise ValueError(f"Unsupported routing value kind: {kind}")
        finite = np.isfinite(logits)
        safe_logits = np.where(finite, logits, -np.inf)
        row_max = np.max(safe_logits, axis=1, keepdims=True)
        all_invalid = ~np.isfinite(row_max[:, 0])
        safe_logits = safe_logits - np.where(np.isfinite(row_max), row_max, 0.0)
        probabilities = np.exp(safe_logits)
        row_sums = np.sum(probabilities, axis=1, keepdims=True)
        normalized = np.divide(
            probabilities,
            row_sums,
            out=np.full_like(probabilities, 1.0 / probabilities.shape[1]),
            where=row_sums > 0,
        )
        if np.any(all_invalid):
            normalized[all_invalid] = 1.0 / probabilities.shape[1]
        return normalized


def _validate_subspace_inputs(
    reference_basis: np.ndarray,
    candidate_basis: np.ndarray,
    max_rank: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    reference = np.asarray(reference_basis, dtype=np.float64)
    candidate = np.asarray(candidate_basis, dtype=np.float64)
    if reference.ndim != 2 or candidate.ndim != 2:
        raise ValueError("Subspace bases must be 2D matrices")
    if reference.shape[0] != candidate.shape[0]:
        raise ValueError("Subspace bases must have the same number of rows")
    if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(candidate)):
        raise ValueError("Subspace bases must contain only finite values")
    max_rank = int(max_rank)
    if max_rank < 1:
        raise ValueError("max_rank must be positive")
    if max_rank > min(reference.shape[1], candidate.shape[1]):
        raise ValueError("max_rank exceeds the available basis dimensions")
    return reference, candidate, max_rank


def _compare_matrix_subspaces(
    reference_basis: np.ndarray,
    candidate_basis: np.ndarray,
    rank: int,
) -> RMTSubspaceComparison:
    reference_q, _ = np.linalg.qr(reference_basis, mode="reduced")
    candidate_q, _ = np.linalg.qr(candidate_basis, mode="reduced")
    canonical_correlations = np.linalg.svd(
        reference_q.T @ candidate_q,
        compute_uv=False,
    )
    canonical_correlations = np.clip(canonical_correlations, 0.0, 1.0)
    angles = np.degrees(np.arccos(canonical_correlations))
    overlap_squared = float(np.sum(canonical_correlations ** 2))
    projection_distance = math.sqrt(max(0.0, 2.0 * rank - 2.0 * overlap_squared))
    normalized_distance = projection_distance / math.sqrt(2.0 * rank)
    return RMTSubspaceComparison(
        rank=int(rank),
        principal_angles_degrees=tuple(map(float, angles)),
        mean_principal_angle_degrees=float(np.mean(angles)),
        max_principal_angle_degrees=float(np.max(angles)),
        projection_distance=float(projection_distance),
        normalized_projection_distance=float(normalized_distance),
        min_canonical_correlation=float(np.min(canonical_correlations)),
    )
