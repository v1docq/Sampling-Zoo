from __future__ import annotations

import math
from typing import Any, Iterable, Optional

import numpy as np

from .matrix_backend import (
    RMTSpectralBasis,
    RMTSubspaceComparison,
    compute_leverage_scores,
)

try:
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None

try:
    import tensorly
except Exception:  # pragma: no cover - optional dependency
    tensorly = None

class HOSVDBackend:
    def __init__(self, random_state=None):
        self.random_state = random_state

    def compute_hosvd(self, X: np.ndarray, rank: list[int] | float, **kwargs):
        """
        Вычисляет HOSVD тензора X с заданными рангами по каждому измерению. Пока использует tensorly с бэкендом PyTorch.
        Arguments:  
        X: входной тензор
        rank: список рангов для каждого измерения тензора или float в (0,1], обозначающий долю от размера соответствующего измерения
        **kwargs: дополнительные параметры для HOSVD
        Returns:
        core_tensor: core tensor after HOSVD
        factors: list of factor matrices for each mode
        """
        if torch is None or tensorly is None:
            raise ImportError("HOSVDBackend requires torch and tensorly to be installed")
        tensorly.set_backend('pytorch')
        torch.manual_seed(self.random_state if self.random_state is not None else 0)

        if isinstance(rank, float):
            rank = [int(rank * dim) for dim in X.shape]
        X_tensor = tensorly.tensor(X, dtype=torch.float32)

        # Вычисление HOSVD
        core_tensor, factors = tensorly.decomposition.hosvd(X_tensor, ranks=rank, **kwargs)

        return core_tensor.numpy(), [factor.numpy() for factor in factors]


class TensorRMTBackend:
    """Torch backend for RMT contraction linear algebra."""

    def __init__(
        self,
        oversample_factor: int = 10,
        power_iterations: int = 2,
        random_state: Optional[int] = None,
        device: str = "cpu",
        dtype: str = "float32",
    ) -> None:
        if torch is None:
            raise ImportError("TensorRMTBackend requires torch to be installed")
        self.oversample_factor = int(oversample_factor)
        self.power_iterations = int(power_iterations)
        self.random_state = 0 if random_state is None else int(random_state)
        self.device = self._resolve_device(device)
        self.dtype = torch.float32 if dtype == "float32" else torch.float64

    @staticmethod
    def _resolve_device(device: str) -> Any:
        requested = torch.device(device)
        if requested.type == "cuda" and not torch.cuda.is_available():
            requested = torch.device("cpu")
        return requested

    def _to_tensor(self, X: Any) -> Any:
        if torch.is_tensor(X):
            return X.to(device=self.device, dtype=self.dtype)
        return torch.as_tensor(X, dtype=self.dtype, device=self.device)

    def build_mode0_unfolding(self, X: np.ndarray, view_specs: Iterable[Any]) -> Any:
        X_tensor = self._to_tensor(X)
        views: list[Any] = []
        for spec in view_specs:
            if spec.columns is None:
                base = X_tensor
            else:
                columns = torch.as_tensor(spec.columns, dtype=torch.long, device=X_tensor.device)
                base = torch.index_select(X_tensor, dim=1, index=columns)
            if spec.projection is None:
                Z = base
            else:
                projection = torch.as_tensor(spec.projection, dtype=X_tensor.dtype, device=X_tensor.device)
                Z = base @ projection
            if int(Z.shape[1]) != spec.output_dim:
                raise RuntimeError("View output dimension mismatch")
            views.append(Z)
        if not views:
            raise RuntimeError("No random contraction views were generated")
        return torch.cat(views, dim=1)

    def compute_spectral_basis(self, M: Any, rank: int) -> RMTSpectralBasis:
        M = self._to_tensor(M)
        rank = int(rank)
        n_components = min(int(M.shape[0]), int(M.shape[1]), rank + self.oversample_factor)
        if n_components < 1:
            raise ValueError("Cannot compute randomized SVD: empty transformed matrix")
        torch.manual_seed(self.random_state)
        omega = torch.randn((int(M.shape[1]), n_components), dtype=M.dtype, device=M.device)
        Y = M @ omega
        for _ in range(max(0, self.power_iterations)):
            Q_iter, _ = torch.linalg.qr(Y, mode="reduced")
            Y = M @ (M.T @ Q_iter)
        Q, _ = torch.linalg.qr(Y, mode="reduced")
        B = Q.T @ M
        U_hat, S, Vh = torch.linalg.svd(B, full_matrices=False)
        U = (Q @ U_hat)[:, :rank]
        singular_values = S[:rank]
        Vt = Vh[:rank, :]
        U_np = U.detach().cpu().numpy()
        return RMTSpectralBasis(
            U=U_np,
            singular_values=singular_values.detach().cpu().numpy(),
            Vt=Vt.detach().cpu().numpy(),
            leverage_scores=compute_leverage_scores(U_np),
        )

    def compare_subspace_prefixes(
        self,
        reference_basis: np.ndarray,
        candidate_basis: np.ndarray,
        max_rank: int,
    ) -> tuple[RMTSubspaceComparison, ...]:
        reference = self._to_tensor(reference_basis)
        candidate = self._to_tensor(candidate_basis)
        max_rank = self._validate_subspace_inputs(reference, candidate, max_rank)
        return tuple(
            self._compare_tensor_subspaces(
                reference[:, :rank],
                candidate[:, :rank],
                rank,
            )
            for rank in range(1, max_rank + 1)
        )

    @staticmethod
    def _validate_subspace_inputs(
        reference_basis: Any,
        candidate_basis: Any,
        max_rank: int,
    ) -> int:
        if reference_basis.ndim != 2 or candidate_basis.ndim != 2:
            raise ValueError("Subspace bases must be 2D matrices")
        if int(reference_basis.shape[0]) != int(candidate_basis.shape[0]):
            raise ValueError("Subspace bases must have the same number of rows")
        if not bool(torch.all(torch.isfinite(reference_basis))):
            raise ValueError("Subspace bases must contain only finite values")
        if not bool(torch.all(torch.isfinite(candidate_basis))):
            raise ValueError("Subspace bases must contain only finite values")
        max_rank = int(max_rank)
        if max_rank < 1:
            raise ValueError("max_rank must be positive")
        available_rank = min(
            int(reference_basis.shape[1]),
            int(candidate_basis.shape[1]),
        )
        if max_rank > available_rank:
            raise ValueError("max_rank exceeds the available basis dimensions")
        return max_rank

    @staticmethod
    def _compare_tensor_subspaces(
        reference_basis: Any,
        candidate_basis: Any,
        rank: int,
    ) -> RMTSubspaceComparison:
        reference_q, _ = torch.linalg.qr(reference_basis, mode="reduced")
        candidate_q, _ = torch.linalg.qr(candidate_basis, mode="reduced")
        canonical_correlations = torch.linalg.svdvals(reference_q.T @ candidate_q)
        canonical_correlations = torch.clamp(canonical_correlations, min=0.0, max=1.0)
        angles = torch.rad2deg(torch.arccos(canonical_correlations))
        overlap_squared = torch.sum(canonical_correlations ** 2)
        projection_squared = torch.clamp(
            2.0 * rank - 2.0 * overlap_squared,
            min=0.0,
        )
        projection_distance = torch.sqrt(projection_squared)
        normalized_distance = projection_distance / math.sqrt(2.0 * rank)
        angles_np = angles.detach().cpu().numpy()
        return RMTSubspaceComparison(
            rank=int(rank),
            principal_angles_degrees=tuple(map(float, angles_np)),
            mean_principal_angle_degrees=float(
                torch.mean(angles).detach().cpu().item()
            ),
            max_principal_angle_degrees=float(
                torch.max(angles).detach().cpu().item()
            ),
            projection_distance=float(projection_distance.detach().cpu().item()),
            normalized_projection_distance=float(
                normalized_distance.detach().cpu().item()
            ),
            min_canonical_correlation=float(
                torch.min(canonical_correlations).detach().cpu().item()
            ),
        )

    def project_new_unfolding(
        self,
        M_new: Any,
        right_basis: np.ndarray,
        singular_values: np.ndarray,
    ) -> np.ndarray:
        M_new = self._to_tensor(M_new)
        basis = torch.as_tensor(right_basis, dtype=M_new.dtype, device=M_new.device)
        sigma = torch.as_tensor(singular_values, dtype=M_new.dtype, device=M_new.device)
        embedding = M_new @ basis.T
        embedding = embedding / torch.clamp(sigma, min=1e-12)
        return embedding.detach().cpu().numpy()

    def routing_probability(
        self,
        embedding: np.ndarray,
        active_centroids: np.ndarray,
        temperature: float,
    ) -> np.ndarray:
        emb = self._to_tensor(embedding)
        centroids = torch.as_tensor(active_centroids, dtype=emb.dtype, device=emb.device)
        d2 = torch.sum((emb[:, None, :] - centroids[None, :, :]) ** 2, dim=2)
        logits = -d2 / max(float(temperature), 1e-8)
        logits = logits - torch.max(logits, dim=1, keepdim=True).values
        proba = torch.softmax(logits, dim=1)
        return proba.detach().cpu().numpy()
