from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.utils.extmath import randomized_svd

from .base_sampler import SpectralSamplerBase
from ...utils.progress import progress_bar
from ...utils.utils import safe_index

try:  # optional backend
    import torch
except Exception:  # pragma: no cover - torch is optional
    torch = None


ArrayLike = Union[np.ndarray, pd.DataFrame]


@dataclass(frozen=True)
class ViewSpec:
    """Description of one random feature contraction/view."""

    columns: Optional[np.ndarray]
    projection: Optional[np.ndarray]
    output_dim: int


@dataclass(frozen=True)
class RMTContractionConfig:
    """Normalized construction parameters for RMT contraction sampling."""

    n_partitions: int = 5
    n_views: int = 16
    view_size: Optional[Union[int, float]] = None
    projection_dim: Optional[int] = None
    approx_rank: Union[int, float] = 16
    view_strategy: str = "gaussian"
    chunk_fraction: float = 1.0
    chunks_percent: float = 100.0
    min_chunk_size: int = 1
    max_chunk_size: Optional[int] = None
    selection_method: str = "hybrid"
    routing_temperature: float = 1.0
    routing_shrinkage: float = 0.05
    include_categorical: bool = True
    max_one_hot_cardinality: int = 128
    max_encoded_features: Optional[int] = 4096
    oversample_factor: int = 10
    power_iterations: int = 2
    backend: str = "auto"
    device: str = "cpu"
    dtype: str = "float32"
    max_unfolding_elements: Optional[int] = None
    show_progress: bool = True
    random_state: Union[int, None] = 42

    @classmethod
    def from_overrides(
        cls,
        config: Optional["RMTContractionConfig"],
        overrides: Dict[str, Any],
    ) -> "RMTContractionConfig":
        base = config or cls()
        valid_fields = cls.__dataclass_fields__
        accepted = {key: value for key, value in overrides.items() if key in valid_fields}
        return replace(base, **accepted)


class RMTContractionTensorSampler(SpectralSamplerBase):
    """
    Chunking sampler based on random feature contractions and sample-mode spectra.

    The sampler converts a flat tabular matrix X in R^{n x p} into the mode-0 unfolding
    of a synthetic tensor T in R^{n x J x K}: each view j is a random contraction of the
    feature mode. The sample-mode unfolding M = T_(0) is decomposed by randomized SVD;
    row leverage scores and low-dimensional sample embeddings are then used to build
    train-time chunks and inference-time routing probabilities.

    This is intentionally a chunking sampler, not only a subset sampler:
    - fit(...) creates self.partitions with train indices for each chunk;
    - predict_partitions(...) maps new points to chunks;
    - predict_partition_proba(...) returns soft memberships for routed ensembles.
    """

    def __init__(
        self,
        config: Optional[RMTContractionConfig] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        if config is not None and not isinstance(config, RMTContractionConfig):
            positional_names = (
                "n_partitions",
                "n_views",
                "view_size",
                "projection_dim",
                "approx_rank",
                "view_strategy",
                "chunk_fraction",
                "chunks_percent",
                "min_chunk_size",
                "max_chunk_size",
                "selection_method",
                "routing_temperature",
                "routing_shrinkage",
                "include_categorical",
                "max_one_hot_cardinality",
                "max_encoded_features",
                "oversample_factor",
                "power_iterations",
                "backend",
                "device",
                "dtype",
                "max_unfolding_elements",
                "show_progress",
                "random_state",
            )
            values = (config, *args)
            if len(values) > len(positional_names):
                raise TypeError(f"Expected at most {len(positional_names)} positional arguments")
            for name, value in zip(positional_names, values):
                kwargs.setdefault(name, value)
            config = None
        elif args:
            raise TypeError("Positional overrides require positional n_partitions as the first argument")
        cfg = RMTContractionConfig.from_overrides(config, kwargs)
        super().__init__(
            sample_size=None,
            approx_rank=cfg.approx_rank,
            random_state=cfg.random_state,
            n_partitions=cfg.n_partitions,
            n_views=cfg.n_views,
            view_strategy=cfg.view_strategy,
            chunk_fraction=cfg.chunk_fraction,
            chunks_percent=cfg.chunks_percent,
            min_chunk_size=cfg.min_chunk_size,
            max_chunk_size=cfg.max_chunk_size,
            selection_method=cfg.selection_method,
            routing_temperature=cfg.routing_temperature,
            routing_shrinkage=cfg.routing_shrinkage,
            backend=cfg.backend,
            device=cfg.device,
            dtype=cfg.dtype,
            include_categorical=cfg.include_categorical,
            max_one_hot_cardinality=cfg.max_one_hot_cardinality,
            max_encoded_features=cfg.max_encoded_features,
            show_progress=cfg.show_progress,
        )
        self.config = cfg
        self.view_size = cfg.view_size
        self.projection_dim = cfg.projection_dim
        self.oversample_factor = int(cfg.oversample_factor)
        self.power_iterations = int(cfg.power_iterations)
        self.max_unfolding_elements = cfg.max_unfolding_elements

    def fit(
        self,
        data: ArrayLike,
        target: Optional[Union[np.ndarray, pd.Series]] = None,
        **kwargs: Any,
    ) -> "RMTContractionTensorSampler":
        with progress_bar(
            enabled=self.show_progress,
            desc="RMT sampler fit",
            total=5,
        ) as stage:
            rng = self._start_fit()
            X_num = self._fit_transform_features(data)
            stage.update(1)
            M = self._build_fit_unfolding(X_num, rng)
            stage.update(1)
            U, S, Vt, scores, rank = self._fit_spectral_basis(M)
            stage.update(1)
            self._store_spectral_basis(U, S, Vt, scores)
            self._fit_clusters_and_partitions(scores, target)
            stage.update(1)
            self._build_diagnostics(M, rank)
            stage.update(1)
        return self

    def _start_fit(self) -> np.random.Generator:
        self.backend_ = self._resolve_backend()
        return np.random.default_rng(self.random_state)

    def _build_fit_unfolding(self, X_num: np.ndarray, rng: np.random.Generator) -> Any:
        return self._build_mode0_unfolding(X_num, fit=True, rng=rng)

    def _fit_spectral_basis(
        self,
        M: Any,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        n_samples, n_features = self._matrix_shape(M)
        rank = self._resolve_rank(self.approx_rank, n_samples, n_features)
        n_components = min(n_samples, n_features, rank + self.oversample_factor)
        if n_components < 1:
            raise ValueError("Cannot compute randomized SVD: empty transformed matrix")
        U, S, Vt, scores = self._compute_spectral_basis(M, rank, n_components)
        return U, S, Vt, scores, rank

    def _store_spectral_basis(
        self,
        U: np.ndarray,
        S: np.ndarray,
        Vt: np.ndarray,
        scores: np.ndarray,
    ) -> None:
        self.singular_values_ = S
        self.right_basis_ = Vt
        self.sample_embedding_ = U
        self.leverage_scores_ = scores

    def _fit_clusters_and_partitions(
        self,
        scores: np.ndarray,
        target: Optional[Union[np.ndarray, pd.Series]],
    ) -> None:
        if self.sample_embedding_ is None:
            raise RuntimeError("Sample embedding is not available")
        n_clusters = min(self.n_partitions, self.sample_embedding_.shape[0])
        self.clusterer_ = self._make_kmeans(n_clusters=n_clusters)
        labels = self.clusterer_.fit_predict(self.sample_embedding_)
        self.cluster_labels_ = labels
        self._build_partitions_from_labels(labels, scores, target)

    def get_partitions(
        self,
        data: Optional[ArrayLike] = None,
        target: Optional[Union[np.ndarray, pd.Series]] = None,
    ) -> Dict[Any, Any]:
        if self.partitions is None or not self.partitions:
            raise ValueError("Sampler not fitted. Call fit() first.")
        if data is None:
            return self.partitions
        return {
            name: {
                "feature": safe_index(data, indices),
                **({"target": safe_index(target, indices)} if target is not None else {}),
            }
            for name, indices in self.partitions.items()
        }

    def sample_indices(self, replace: bool = False) -> List[int]:
        if self.leverage_scores_ is None:
            raise RuntimeError("Sampler not fitted. Call fit() first.")
        n_samples = self.leverage_scores_.shape[0]
        sample_size = min(n_samples, sum(len(v) for v in self.partitions.values()))
        rng = np.random.default_rng(self.random_state)
        return rng.choice(
            np.arange(n_samples),
            size=sample_size,
            replace=replace,
            p=self.leverage_scores_,
        ).tolist()

    def predict_partitions(self, X: ArrayLike) -> np.ndarray:
        proba = self.predict_partition_proba(X)
        best = np.argmax(proba, axis=1)
        cluster_ids = [self.partition_to_cluster_[name] for name in self.partition_names_]
        return np.asarray([cluster_ids[i] for i in best], dtype=int)

    def predict_partition_proba(self, X: ArrayLike) -> np.ndarray:
        if self.clusterer_ is None or self.sample_embedding_ is None:
            raise RuntimeError("Sampler not fitted. Call fit() first.")
        if not self.partition_names_:
            raise RuntimeError("No partitions available. Call fit() first.")

        X_num = self._transform_features(X)
        M_new = self._build_mode0_unfolding(X_num, fit=False, rng=None)
        embedding = self._project_new_unfolding(M_new)

        centroids = self.clusterer_.cluster_centers_
        active_cluster_ids = np.asarray([self.partition_to_cluster_[name] for name in self.partition_names_], dtype=int)
        active_centroids = centroids[active_cluster_ids]
        proba = self._routing_probability(embedding, active_centroids)

        if self.routing_shrinkage > 0:
            m = proba.shape[1]
            lam = min(max(self.routing_shrinkage, 0.0), 1.0)
            proba = (1.0 - lam) * proba + lam / m
        return proba

    def transform_embedding(self, X: ArrayLike) -> np.ndarray:
        """Return the latent sample-mode embedding used by the router."""
        X_num = self._transform_features(X)
        M_new = self._build_mode0_unfolding(X_num, fit=False, rng=None)
        return self._project_new_unfolding(M_new)

    @staticmethod
    def _matrix_shape(X: Any) -> Tuple[int, int]:
        return int(X.shape[0]), int(X.shape[1])

    def _check_unfolding_size(self, n_samples: int, n_features: int) -> None:
        if self.max_unfolding_elements is None:
            return
        total = int(n_samples) * int(n_features)
        if total > int(self.max_unfolding_elements):
            raise MemoryError(
                "RMT mode-0 unfolding would exceed max_unfolding_elements: "
                f"{total} > {self.max_unfolding_elements}"
            )

    def _build_mode0_unfolding(
        self,
        X: np.ndarray,
        fit: bool,
        rng: Optional[np.random.Generator],
    ) -> np.ndarray:
        n_samples, n_features = X.shape
        if fit:
            if rng is None:
                rng = np.random.default_rng(self.random_state)
            self.view_specs_ = self._make_view_specs(n_features, rng)

        self._check_unfolding_size(n_samples, sum(spec.output_dim for spec in self.view_specs_))
        if self.backend_ == "torch":
            return self._build_mode0_unfolding_torch(X)

        views: List[np.ndarray] = []
        for spec in self.view_specs_:
            if spec.columns is None:
                base = X
            else:
                base = X[:, spec.columns]
            if spec.projection is None:
                Z = base
            else:
                Z = base @ spec.projection
            if Z.shape[1] != spec.output_dim:
                raise RuntimeError("View output dimension mismatch")
            views.append(Z)
        if not views:
            raise RuntimeError("No random contraction views were generated")
        return np.concatenate(views, axis=1)

    def _build_mode0_unfolding_torch(self, X: np.ndarray) -> Any:
        X_tensor = self._to_torch_matrix(X)
        views: List[Any] = []
        for spec in self.view_specs_:
            if spec.columns is None:
                base = X_tensor
            else:
                columns = torch.as_tensor(spec.columns, dtype=torch.long, device=X_tensor.device)
                base = torch.index_select(X_tensor, dim=1, index=columns)
            if spec.projection is None:
                Z = base
            else:
                projection = torch.as_tensor(
                    spec.projection,
                    dtype=X_tensor.dtype,
                    device=X_tensor.device,
                )
                Z = base @ projection
            if int(Z.shape[1]) != spec.output_dim:
                raise RuntimeError("View output dimension mismatch")
            views.append(Z)
        if not views:
            raise RuntimeError("No random contraction views were generated")
        return torch.cat(views, dim=1)

    def _make_view_specs(self, n_features: int, rng: np.random.Generator) -> List[ViewSpec]:
        view_size = self._resolve_view_size(n_features)
        projection_dim = self.projection_dim or view_size
        projection_dim = max(1, int(min(projection_dim, view_size if self.view_strategy == "subsample" else projection_dim)))

        specs: List[ViewSpec] = []
        for _ in range(self.n_views):
            if self.view_strategy == "subsample":
                replace = view_size > n_features
                cols = np.sort(rng.choice(n_features, size=view_size, replace=replace))
                if projection_dim < view_size:
                    proj = rng.normal(0.0, 1.0 / math.sqrt(projection_dim), size=(view_size, projection_dim))
                else:
                    proj = None
                specs.append(ViewSpec(columns=cols, projection=proj, output_dim=projection_dim if proj is not None else view_size))
            else:
                proj = rng.normal(0.0, 1.0 / math.sqrt(projection_dim), size=(n_features, projection_dim))
                specs.append(ViewSpec(columns=None, projection=proj, output_dim=projection_dim))
        return specs

    def _resolve_view_size(self, n_features: int) -> int:
        if self.view_strategy == "gaussian":
            if self.view_size is None:
                return n_features
        if self.view_size is None:
            return max(1, min(n_features, int(math.ceil(math.sqrt(n_features)))))
        if isinstance(self.view_size, float):
            if not (0 < self.view_size <= 1):
                raise ValueError("float view_size must be in (0, 1]")
            return max(1, int(math.ceil(self.view_size * n_features)))
        return max(1, int(self.view_size))

    @staticmethod
    def _resolve_rank(rank: Union[int, float], n_samples: int, n_features: int) -> int:
        max_rank = max(1, min(n_samples, n_features))
        if isinstance(rank, float):
            if not (0 < rank <= 1):
                raise ValueError("float approx_rank must be in (0, 1]")
            rank = int(math.ceil(rank * max_rank))
        return max(1, min(int(rank), max_rank))

    def _compute_spectral_basis(
        self,
        M: Any,
        rank: int,
        n_components: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.backend_ == "torch":
            U_t, S_t, Vt_t = self._torch_randomized_svd(M, rank=rank, n_components=n_components)
            scores_t = torch.sum(U_t * U_t, dim=1)
            score_sum = torch.sum(scores_t)
            if not bool(torch.isfinite(score_sum)) or float(score_sum.detach().cpu()) <= 0.0:
                scores = np.full(int(U_t.shape[0]), 1.0 / int(U_t.shape[0]))
            else:
                scores = (scores_t / score_sum).detach().cpu().numpy()
            return (
                U_t.detach().cpu().numpy(),
                S_t.detach().cpu().numpy(),
                Vt_t.detach().cpu().numpy(),
                scores,
            )

        U, S, Vt = randomized_svd(
            M,
            n_components=n_components,
            n_iter=self.power_iterations,
            random_state=self.random_state,
        )
        U = U[:, :rank]
        S = S[:rank]
        Vt = Vt[:rank, :]
        scores = np.sum(U * U, axis=1)
        score_sum = float(np.sum(scores))
        if not np.isfinite(score_sum) or score_sum <= 0:
            scores = np.full(M.shape[0], 1.0 / M.shape[0])
        else:
            scores = scores / score_sum
        return U, S, Vt, scores

    def _torch_randomized_svd(self, M: Any, rank: int, n_components: int) -> Tuple[Any, Any, Any]:
        if torch is None:
            raise RuntimeError("Torch backend selected but torch is unavailable")
        torch.manual_seed(self.random_state)
        omega = torch.randn(
            (int(M.shape[1]), int(n_components)),
            dtype=M.dtype,
            device=M.device,
        )
        Y = M @ omega
        for _ in range(max(0, self.power_iterations)):
            Q_iter, _ = torch.linalg.qr(Y, mode="reduced")
            Y = M @ (M.T @ Q_iter)
        Q, _ = torch.linalg.qr(Y, mode="reduced")
        B = Q.T @ M
        U_hat, S, Vh = torch.linalg.svd(B, full_matrices=False)
        U = Q @ U_hat
        return U[:, :rank], S[:rank], Vh[:rank, :]

    def _make_kmeans(self, n_clusters: int) -> KMeans:
        try:
            return KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init="auto")
        except TypeError:
            return KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)

    def _build_partitions_from_labels(
        self,
        labels: np.ndarray,
        scores: np.ndarray,
        target: Optional[Union[np.ndarray, pd.Series]] = None,
    ) -> None:
        partitions: Dict[str, np.ndarray] = {}
        cluster_quality: List[Tuple[str, float]] = []

        for cluster_id in sorted(np.unique(labels).tolist()):
            cluster_idx = np.where(labels == cluster_id)[0]
            selected = self._select_from_cluster(cluster_idx, scores)
            if selected.size == 0:
                continue
            name = f"chunk_{int(cluster_id)}"
            partitions[name] = selected
            cluster_quality.append((name, float(np.mean(scores[selected]))))

        if self.chunks_percent < 100.0 and partitions:
            keep_n = max(1, int(math.ceil(len(partitions) * self.chunks_percent / 100.0)))
            keep_names = {
                name for name, _ in sorted(cluster_quality, key=lambda x: x[1], reverse=True)[:keep_n]
            }
            partitions = {name: idx for name, idx in partitions.items() if name in keep_names}

        self.partitions = partitions
        self.partition_names_ = list(partitions.keys())
        self.partition_to_cluster_ = {name: int(name.split("_")[-1]) for name in self.partition_names_}

    def _select_from_cluster(self, cluster_idx: np.ndarray, scores: np.ndarray) -> np.ndarray:
        if cluster_idx.size == 0:
            return cluster_idx
        if self.selection_method == "all" and self.chunk_fraction >= 1.0 and self.max_chunk_size is None:
            return np.asarray(cluster_idx, dtype=int)

        target_size = int(math.ceil(cluster_idx.size * self.chunk_fraction))
        target_size = max(self.min_chunk_size, target_size)
        if self.max_chunk_size is not None:
            target_size = min(target_size, int(self.max_chunk_size))
        target_size = min(target_size, cluster_idx.size)

        if self.selection_method == "all":
            return np.asarray(cluster_idx[:target_size], dtype=int)

        if self.selection_method == "leverage":
            order = np.argsort(-scores[cluster_idx], kind="mergesort")
            return np.asarray(cluster_idx[order[:target_size]], dtype=int)

        if self.sample_embedding_ is None:
            raise RuntimeError("Sample embedding is not available")

        if self.selection_method == "maxvol":
            return self._greedy_maxvol_indices(cluster_idx, target_size)

        # hybrid: half leverage, half diversity via greedy MaxVol/farthest-span proxy
        first = max(1, target_size // 2)
        order = np.argsort(-scores[cluster_idx], kind="mergesort")
        picked = list(cluster_idx[order[:first]])
        remaining = np.setdiff1d(cluster_idx, np.asarray(picked, dtype=int), assume_unique=False)
        if len(picked) < target_size and remaining.size > 0:
            extra = self._greedy_maxvol_indices(remaining, target_size - len(picked))
            picked.extend(extra.tolist())
        return np.asarray(picked[:target_size], dtype=int)

    def _greedy_maxvol_indices(self, candidate_idx: np.ndarray, target_size: int) -> np.ndarray:
        """Small, dependency-light proxy for MaxVol: greedily maximize residual norm."""
        E = self.sample_embedding_[candidate_idx]
        if candidate_idx.size <= target_size:
            return np.asarray(candidate_idx, dtype=int)
        norms = np.sum(E * E, axis=1)
        first = int(np.argmax(norms))
        picked_local = [first]
        Q = self._orthonormal_basis(E[[first]])

        while len(picked_local) < target_size:
            proj = E @ Q.T if Q.size else 0.0
            residual = E - proj @ Q if Q.size else E
            res_norms = np.sum(residual * residual, axis=1)
            res_norms[picked_local] = -np.inf
            nxt = int(np.argmax(res_norms))
            if not np.isfinite(res_norms[nxt]):
                break
            picked_local.append(nxt)
            Q = self._orthonormal_basis(E[picked_local])
        return np.asarray(candidate_idx[picked_local], dtype=int)

    @staticmethod
    def _orthonormal_basis(rows: np.ndarray) -> np.ndarray:
        if rows.size == 0:
            return np.empty((0, 0))
        # QR on transpose gives an orthonormal row-span basis after transpose.
        Q, _ = np.linalg.qr(rows.T, mode="reduced")
        return Q.T

    def _project_new_unfolding(self, M_new: np.ndarray) -> np.ndarray:
        if self.right_basis_ is None or self.singular_values_ is None:
            raise RuntimeError("Spectral basis is not fitted")
        if self.backend_ == "torch":
            if torch is None:
                raise RuntimeError("Torch backend selected but torch is unavailable")
            if not torch.is_tensor(M_new):
                M_new = self._to_torch_matrix(M_new)
            right_basis = torch.as_tensor(
                self.right_basis_,
                dtype=M_new.dtype,
                device=M_new.device,
            )
            singular_values = torch.as_tensor(
                self.singular_values_,
                dtype=M_new.dtype,
                device=M_new.device,
            )
            embedding = M_new @ right_basis.T
            embedding = embedding / torch.clamp(singular_values, min=1e-12)
            return embedding.detach().cpu().numpy()
        embedding = M_new @ self.right_basis_.T
        embedding = embedding / np.maximum(self.singular_values_, 1e-12)
        return embedding

    def _routing_probability(self, embedding: np.ndarray, active_centroids: np.ndarray) -> np.ndarray:
        if self.backend_ == "torch":
            if torch is None:
                raise RuntimeError("Torch backend selected but torch is unavailable")
            emb = self._to_torch_matrix(embedding)
            centroids = torch.as_tensor(active_centroids, dtype=emb.dtype, device=emb.device)
            d2 = torch.sum((emb[:, None, :] - centroids[None, :, :]) ** 2, dim=2)
            logits = -d2 / max(self.routing_temperature, 1e-8)
            logits = logits - torch.max(logits, dim=1, keepdim=True).values
            proba = torch.softmax(logits, dim=1)
            return proba.detach().cpu().numpy()

        d2 = np.sum((embedding[:, None, :] - active_centroids[None, :, :]) ** 2, axis=2)
        temp = max(self.routing_temperature, 1e-8)
        logits = -d2 / temp
        logits = logits - np.max(logits, axis=1, keepdims=True)
        proba = np.exp(logits)
        return proba / np.maximum(np.sum(proba, axis=1, keepdims=True), 1e-12)

    def _build_diagnostics(self, M: Any, rank: int) -> None:
        scores = self.leverage_scores_
        entropy = None
        eff_n = None
        if scores is not None:
            entropy = float(-np.sum(scores * np.log(scores + 1e-12)))
            eff_n = float(np.exp(entropy))
        self.diagnostics_ = {
            "backend": self.backend_,
            "device": self.device if self.backend_ == "torch" else None,
            "dtype": self.dtype,
            "mode0_unfolding_shape": tuple(map(int, M.shape)),
            "raw_encoded_feature_count": self.raw_encoded_feature_count_,
            "encoded_feature_count": self.encoded_feature_count_,
            "encoded_feature_cap_applied": self.encoded_feature_subset_ is not None,
            "n_views": int(self.n_views),
            "approx_rank": int(rank),
            "singular_values": self.singular_values_.tolist() if self.singular_values_ is not None else [],
            "leverage_entropy": entropy,
            "effective_sample_count": eff_n,
            "n_partitions": int(len(self.partitions)),
            "chunk_sizes": {name: int(len(idx)) for name, idx in self.partitions.items()},
        }
