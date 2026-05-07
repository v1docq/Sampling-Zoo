from __future__ import annotations

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
        d2 = np.sum((embedding[:, None, :] - active_centroids[None, :, :]) ** 2, axis=2)
        temp = max(float(temperature), 1e-8)
        logits = -d2 / temp
        logits = logits - np.max(logits, axis=1, keepdims=True)
        proba = np.exp(logits)
        return proba / np.maximum(np.sum(proba, axis=1, keepdims=True), 1e-12)
