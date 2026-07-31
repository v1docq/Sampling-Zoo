from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from .backend.matrix_backend import MatrixRMTBackend
from .backend.tensor_backend import TensorRMTBackend
from .base_sampler import SpectralSamplerBase
from .cluster_selection import ClusterSelectionResult, SpectralClusterSelector
from .null_diagnostics import (
    SpectralNullDiagnostic,
    SpectralNullDiagnosticConfig,
    SpectralNullDiagnosticResult,
)
from .subspace_diagnostics import (
    SpectralSubspaceDiagnostic,
    SpectralSubspaceDiagnosticConfig,
    SpectralSubspaceDiagnosticResult,
)
from ...utils.progress import progress_bar
from ...utils.utils import safe_index


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
    partition_selection_method: str = "fixed"
    cluster_algorithms: Tuple[str, ...] = ("kmeans",)
    cluster_selection_metric: str = "balanced_silhouette"
    cluster_ensemble_method: str = "best_score"
    min_partitions: int = 2
    max_partitions: Optional[int] = None
    min_auto_partition_size: int = 256
    partition_selection_sample_size: int = 5000
    max_cluster_imbalance_ratio: float = 5.0
    min_cluster_fraction: float = 0.05
    imbalance_penalty_weight: float = 0.15
    tiny_cluster_penalty_weight: float = 0.30
    target_contrast_weight: float = 0.0
    cluster_vote_temperature: float = 0.05
    n_views: Union[int, str] = "auto"
    n_views_policy: str = "auto"
    min_views: int = 4
    max_views: int = 64
    target_feature_coverage: float = 0.95
    spectrum_stability_tolerance: float = 0.05
    embedding_mode: str = "sv_scaled"
    view_size: Optional[Union[int, float]] = None
    projection_dim: Optional[int] = None
    initial_rank_fraction: float = 0.25
    rank_selection_method: str = "explained_variance"
    explained_variance_threshold: float = 0.95
    min_rank: int = 1
    null_diagnostic_enabled: bool = False
    null_model_policies: Tuple[str, ...] = (
        "feature_permutation",
        "moment_matched_gaussian",
        "view_resampling",
    )
    null_resamples: int = 16
    null_quantile: float = 0.95
    null_min_selection_frequency: float = 0.80
    null_primary_policy: str = "feature_permutation"
    subspace_diagnostic_enabled: bool = False
    subspace_resamples: int = 16
    subspace_quantile: float = 0.90
    subspace_max_principal_angle_degrees: float = 15.0
    subspace_max_normalized_projection_distance: float = 0.25
    subspace_max_rank: Optional[int] = 64
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
        if "approx_rank" in overrides:
            raise ValueError(
                "approx_rank is no longer supported by RMTContractionTensorSampler. "
                "Use initial_rank_fraction, rank_selection_method, and "
                "explained_variance_threshold instead."
            )
        base = config or cls()
        valid_fields = cls.__dataclass_fields__
        accepted = {key: value for key, value in overrides.items() if key in valid_fields}
        return replace(base, **accepted)


@dataclass(frozen=True)
class RankSelectionInfo:
    initial_rank: int
    selected_rank: int
    rank_selection_method: str
    explained_variance_threshold: float
    explained_variance_at_selected_rank: float


@dataclass(frozen=True)
class NViewsSelectionInfo:
    requested_n_views: Union[int, str]
    resolved_n_views: int
    n_views_policy: str
    target_feature_coverage: Optional[float]
    estimated_feature_coverage: Optional[float]
    max_views_by_unfolding: Optional[int]
    spectrum_stability_tolerance: Optional[float]
    spectrum_stability_change: Optional[float]
    spectrum_stability_candidates: Tuple[int, ...]


@dataclass(frozen=True)
class PartitionSelectionInfo:
    requested_n_partitions: int
    selected_n_partitions: int
    partition_selection_method: str
    selected_algorithm: Optional[str]
    cluster_algorithms: Tuple[str, ...]
    cluster_selection_metric: str
    cluster_ensemble_method: str
    candidates: Tuple[int, ...]
    scores: Tuple[Tuple[int, Optional[float]], ...]
    candidate_details: Tuple[Dict[str, Any], ...]
    candidate_plan: Dict[str, Any]
    candidate_failures: Tuple[Dict[str, Any], ...]
    min_auto_partition_size: int
    selection_sample_size: int


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
        cfg = self._normalize_config_inputs(config, args, kwargs)
        requested_n_views = self._normalize_n_views_request(cfg.n_views)
        base_n_views = cfg.min_views if requested_n_views == "auto" else int(requested_n_views)
        super().__init__(
            sample_size=None,
            approx_rank=1.0,
            random_state=cfg.random_state,
            n_partitions=cfg.n_partitions,
            n_views=base_n_views,
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
        self.n_views_requested = requested_n_views
        self.partition_selection_method = self._validate_choice(
            "partition_selection_method",
            cfg.partition_selection_method,
            ("fixed", "auto"),
        )
        self.cluster_algorithms = self._normalize_cluster_algorithms(cfg.cluster_algorithms)
        self.cluster_selection_metric = self._validate_choice(
            "cluster_selection_metric",
            cfg.cluster_selection_metric,
            ("silhouette", "balanced_silhouette"),
        )
        self.cluster_ensemble_method = self._validate_choice(
            "cluster_ensemble_method",
            cfg.cluster_ensemble_method,
            ("best_score", "weighted_vote"),
        )
        self.min_partitions = self._validate_positive_int("min_partitions", cfg.min_partitions)
        self.max_partitions = (
            None
            if cfg.max_partitions is None
            else self._validate_positive_int("max_partitions", cfg.max_partitions)
        )
        if self.max_partitions is not None and self.max_partitions < self.min_partitions:
            raise ValueError("max_partitions must be greater than or equal to min_partitions")
        self.min_auto_partition_size = self._validate_positive_int(
            "min_auto_partition_size",
            cfg.min_auto_partition_size,
        )
        self.partition_selection_sample_size = self._validate_positive_int(
            "partition_selection_sample_size",
            cfg.partition_selection_sample_size,
        )
        self.max_cluster_imbalance_ratio = self._validate_positive_float(
            "max_cluster_imbalance_ratio",
            cfg.max_cluster_imbalance_ratio,
        )
        self.min_cluster_fraction = self._validate_fraction(
            "min_cluster_fraction",
            cfg.min_cluster_fraction,
        )
        self.imbalance_penalty_weight = float(cfg.imbalance_penalty_weight)
        self.tiny_cluster_penalty_weight = float(cfg.tiny_cluster_penalty_weight)
        self.target_contrast_weight = float(cfg.target_contrast_weight)
        self.cluster_vote_temperature = self._validate_positive_float(
            "cluster_vote_temperature",
            cfg.cluster_vote_temperature,
        )
        self.n_views_policy = self._validate_choice(
            "n_views_policy",
            cfg.n_views_policy,
            ("auto", "coverage", "spectrum_stability"),
        )
        self.min_views = self._validate_positive_int("min_views", cfg.min_views)
        self.max_views = self._validate_positive_int("max_views", cfg.max_views)
        if self.max_views < self.min_views:
            raise ValueError("max_views must be greater than or equal to min_views")
        self.target_feature_coverage = self._validate_fraction(
            "target_feature_coverage",
            cfg.target_feature_coverage,
        )
        self.spectrum_stability_tolerance = self._validate_positive_float(
            "spectrum_stability_tolerance",
            cfg.spectrum_stability_tolerance,
        )
        self.embedding_mode = self._validate_choice(
            "embedding_mode",
            cfg.embedding_mode,
            ("sv_scaled", "whitened"),
        )
        if self.n_views_requested != "auto":
            self.n_views = int(self.n_views_requested)
        self.view_size = cfg.view_size
        self.projection_dim = cfg.projection_dim
        self.initial_rank_fraction = self._validate_fraction(
            "initial_rank_fraction",
            cfg.initial_rank_fraction,
        )
        self.rank_selection_method = self._validate_choice(
            "rank_selection_method",
            cfg.rank_selection_method,
            ("explained_variance",),
        )
        self.explained_variance_threshold = self._validate_fraction(
            "explained_variance_threshold",
            cfg.explained_variance_threshold,
        )
        self.min_rank = self._validate_positive_int("min_rank", cfg.min_rank)
        null_config = SpectralNullDiagnosticConfig.from_values(
            enabled=cfg.null_diagnostic_enabled,
            policies=cfg.null_model_policies,
            n_resamples=cfg.null_resamples,
            quantile=cfg.null_quantile,
            min_selection_frequency=cfg.null_min_selection_frequency,
            primary_policy=cfg.null_primary_policy,
            random_state=cfg.random_state,
        )
        self.spectral_null_diagnostic = SpectralNullDiagnostic(null_config)
        subspace_config = SpectralSubspaceDiagnosticConfig.from_values(
            enabled=cfg.subspace_diagnostic_enabled,
            n_resamples=cfg.subspace_resamples,
            quantile=cfg.subspace_quantile,
            max_principal_angle_degrees=(
                cfg.subspace_max_principal_angle_degrees
            ),
            max_normalized_projection_distance=(
                cfg.subspace_max_normalized_projection_distance
            ),
            max_rank=cfg.subspace_max_rank,
            random_state=cfg.random_state,
        )
        self.spectral_subspace_diagnostic = SpectralSubspaceDiagnostic(
            subspace_config
        )
        self.oversample_factor = int(cfg.oversample_factor)
        self.power_iterations = int(cfg.power_iterations)
        self.max_unfolding_elements = cfg.max_unfolding_elements
        self._rmt_backend: Optional[Union[MatrixRMTBackend, TensorRMTBackend]] = None
        self.cluster_selector_ = self._make_cluster_selector()
        self.cluster_centers_: Optional[np.ndarray] = None
        self.initial_singular_values_: Optional[np.ndarray] = None
        self.left_basis_: Optional[np.ndarray] = None
        self.spectral_null_diagnostic_result_ = SpectralNullDiagnosticResult.disabled(
            null_config.primary_policy
        )
        self.spectral_subspace_diagnostic_result_ = (
            SpectralSubspaceDiagnosticResult.disabled()
        )
        self.rank_selection_info_: Optional[RankSelectionInfo] = None
        self.n_views_selection_info_: Optional[NViewsSelectionInfo] = None
        self.partition_selection_info_: Optional[PartitionSelectionInfo] = None

    @staticmethod
    def _normalize_config_inputs(
        config: Optional[RMTContractionConfig],
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> RMTContractionConfig:
        kwargs = dict(kwargs)
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
                "n_views_policy",
                "min_views",
                "max_views",
                "target_feature_coverage",
                "spectrum_stability_tolerance",
                "embedding_mode",
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
        return RMTContractionConfig.from_overrides(config, kwargs)

    @staticmethod
    def _normalize_n_views_request(value: Union[int, str]) -> Union[int, str]:
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized != "auto":
                raise ValueError("n_views must be a positive integer or 'auto'")
            return "auto"
        try:
            n_views = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("n_views must be a positive integer or 'auto'") from exc
        if n_views < 1:
            raise ValueError("n_views must be positive")
        return n_views

    @staticmethod
    def _normalize_cluster_algorithms(value: Union[str, Sequence[str]]) -> Tuple[str, ...]:
        if isinstance(value, str):
            algorithms = (value,)
        else:
            algorithms = tuple(value)
        if not algorithms:
            raise ValueError("cluster_algorithms must contain at least one algorithm")
        return tuple(str(algorithm).strip().lower() for algorithm in algorithms)

    @staticmethod
    def _validate_positive_float(name: str, value: float) -> float:
        value = float(value)
        if value <= 0:
            raise ValueError(f"{name} must be positive")
        return value

    def _make_cluster_selector(self) -> SpectralClusterSelector:
        return SpectralClusterSelector(
            algorithms=self.cluster_algorithms,
            selection_metric=self.cluster_selection_metric,
            ensemble_method=self.cluster_ensemble_method,
            min_partitions=self.min_partitions,
            max_partitions=self.max_partitions,
            min_auto_partition_size=self.min_auto_partition_size,
            selection_sample_size=self.partition_selection_sample_size,
            max_cluster_imbalance_ratio=self.max_cluster_imbalance_ratio,
            min_cluster_fraction=self.min_cluster_fraction,
            imbalance_penalty_weight=self.imbalance_penalty_weight,
            tiny_cluster_penalty_weight=self.tiny_cluster_penalty_weight,
            target_contrast_weight=self.target_contrast_weight,
            vote_temperature=self.cluster_vote_temperature,
            random_state=self.random_state,
            show_progress=self.show_progress,
        )

    def fit(
        self,
        data: ArrayLike,
        target: Optional[Union[np.ndarray, pd.Series]] = None,
        **kwargs: Any,
    ) -> "RMTContractionTensorSampler":
        with progress_bar(
            enabled=self.show_progress,
            desc="RMT sampler fit",
            total=7,
        ) as stage:
            rng = self._start_fit()
            X_num = self._fit_transform_features(data)
            stage.update(1)
            M = self._build_fit_unfolding(X_num, rng)
            stage.update(1)
            U, S, Vt, scores, rank = self._fit_spectral_basis(M)
            stage.update(1)
            self._store_spectral_basis(U, S, Vt, scores)
            self._fit_spectral_null_diagnostic(X_num)
            stage.update(1)
            self._fit_spectral_subspace_diagnostic(X_num)
            stage.update(1)
            self._fit_clusters_and_partitions(scores, target)
            stage.update(1)
            self._build_diagnostics(M, rank)
            stage.update(1)
        return self

    def _start_fit(self) -> np.random.Generator:
        self.backend_ = self._resolve_backend()
        self._rmt_backend = self._make_rmt_backend()
        self.view_specs_ = []
        self.initial_singular_values_ = None
        self.left_basis_ = None
        self.spectral_null_diagnostic_result_ = SpectralNullDiagnosticResult.disabled(
            self.spectral_null_diagnostic.config.primary_policy
        )
        self.spectral_subspace_diagnostic_result_ = (
            SpectralSubspaceDiagnosticResult.disabled()
        )
        self.rank_selection_info_ = None
        self.n_views_selection_info_ = None
        self.partition_selection_info_ = None
        self.cluster_centers_ = None
        if self.n_views_requested == "auto":
            self.n_views = self.min_views
        return np.random.default_rng(self.random_state)

    def _make_rmt_backend(self) -> Union[MatrixRMTBackend, TensorRMTBackend]:
        if self.backend_ == "torch":
            return TensorRMTBackend(
                oversample_factor=self.oversample_factor,
                power_iterations=self.power_iterations,
                random_state=self.random_state,
                device=self.device,
                dtype=self.dtype,
            )
        return MatrixRMTBackend(
            oversample_factor=self.oversample_factor,
            power_iterations=self.power_iterations,
            random_state=self.random_state,
        )

    def _get_rmt_backend(self) -> Union[MatrixRMTBackend, TensorRMTBackend]:
        if self._rmt_backend is None:
            raise RuntimeError("RMT backend is not initialized")
        return self._rmt_backend

    def _build_fit_unfolding(self, X_num: np.ndarray, rng: np.random.Generator) -> Any:
        if self.n_views_requested == "auto":
            policy = self._resolve_n_views_policy()
            if policy == "spectrum_stability":
                return self._build_spectrum_stable_fit_unfolding(X_num, rng)
            self.n_views = self._resolve_coverage_n_views(*X_num.shape)
        return self._build_mode0_unfolding(X_num, fit=True, rng=rng)

    def _fit_spectral_basis(
        self,
        M: Any,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, RankSelectionInfo]:
        n_samples, n_features = self._matrix_shape(M)
        initial_rank = self._resolve_initial_rank(n_samples, n_features)
        basis = self._get_rmt_backend().compute_spectral_basis(M, rank=initial_rank)
        self.initial_singular_values_ = np.asarray(
            basis.singular_values,
            dtype=np.float64,
        ).copy()
        selected_rank, explained_variance = self._select_rank_from_spectrum(basis.singular_values)
        selected_basis = basis.truncate(selected_rank)
        rank_info = RankSelectionInfo(
            initial_rank=initial_rank,
            selected_rank=selected_rank,
            rank_selection_method=self.rank_selection_method,
            explained_variance_threshold=self.explained_variance_threshold,
            explained_variance_at_selected_rank=explained_variance,
        )
        self.rank_selection_info_ = rank_info
        return (
            selected_basis.U,
            selected_basis.singular_values,
            selected_basis.Vt,
            selected_basis.leverage_scores,
            rank_info,
        )

    def _fit_spectral_null_diagnostic(self, X: np.ndarray) -> None:
        if self.initial_singular_values_ is None:
            raise RuntimeError("Initial singular spectrum is not available")

        diagnostic = self.spectral_null_diagnostic
        if not diagnostic.config.enabled:
            self.spectral_null_diagnostic_result_ = SpectralNullDiagnosticResult.disabled(
                diagnostic.config.primary_policy
            )
            return

        rank = int(self.initial_singular_values_.size)

        def evaluate_spectrum(
            reference_X: np.ndarray,
            rng: np.random.Generator,
            resample_views: bool,
        ) -> np.ndarray:
            specs = (
                self._make_view_specs(reference_X.shape[1], rng, n_views=self.n_views)
                if resample_views
                else self.view_specs_
            )
            width = sum(spec.output_dim for spec in specs)
            self._check_unfolding_size(reference_X.shape[0], width)
            unfolding = self._get_rmt_backend().build_mode0_unfolding(reference_X, specs)
            return self._get_rmt_backend().compute_spectral_basis(
                unfolding,
                rank=rank,
            ).singular_values

        with progress_bar(
            enabled=self.show_progress,
            desc="RMT spectral null diagnostics",
            total=diagnostic.config.work_units,
        ) as progress:
            self.spectral_null_diagnostic_result_ = diagnostic.evaluate(
                encoded_features=X,
                observed_singular_values=self.initial_singular_values_,
                spectrum_evaluator=evaluate_spectrum,
                on_progress=lambda: progress.update(1),
            )

    def _fit_spectral_subspace_diagnostic(self, X: np.ndarray) -> None:
        if self.left_basis_ is None:
            raise RuntimeError("Left spectral basis is not available")

        diagnostic = self.spectral_subspace_diagnostic
        if not diagnostic.config.enabled:
            self.spectral_subspace_diagnostic_result_ = (
                SpectralSubspaceDiagnosticResult.disabled()
            )
            return

        rank = diagnostic.resolve_comparison_rank(
            int(self.left_basis_.shape[1])
        )

        def evaluate_basis(rng: np.random.Generator) -> np.ndarray:
            specs = self._make_view_specs(
                X.shape[1],
                rng,
                n_views=self.n_views,
            )
            width = sum(spec.output_dim for spec in specs)
            self._check_unfolding_size(X.shape[0], width)
            unfolding = self._get_rmt_backend().build_mode0_unfolding(X, specs)
            return self._get_rmt_backend().compute_spectral_basis(
                unfolding,
                rank=rank,
            ).U

        with progress_bar(
            enabled=self.show_progress,
            desc="RMT subspace stability",
            total=diagnostic.config.work_units,
        ) as progress:
            self.spectral_subspace_diagnostic_result_ = diagnostic.evaluate(
                reference_basis=self.left_basis_,
                basis_evaluator=evaluate_basis,
                subspace_comparator=(
                    self._get_rmt_backend().compare_subspace_prefixes
                ),
                on_progress=lambda: progress.update(1),
            )

    def _resolve_initial_rank(self, n_samples: int, n_features: int) -> int:
        max_rank = max(1, min(n_samples, n_features))
        initial_rank = int(math.ceil(self.initial_rank_fraction * max_rank))
        initial_rank = max(self.min_rank, initial_rank)
        return max(1, min(initial_rank, max_rank))

    def _select_rank_from_spectrum(self, singular_values: np.ndarray) -> Tuple[int, float]:
        if self.rank_selection_method != "explained_variance":
            raise ValueError(f"Unsupported rank_selection_method: {self.rank_selection_method}")
        values = np.asarray(singular_values, dtype=np.float64)
        if values.size == 0:
            return self.min_rank, 0.0
        energy = values * values
        total_energy = float(np.sum(energy))
        if not np.isfinite(total_energy) or total_energy <= 0:
            rank = min(self.min_rank, values.size)
            return rank, 0.0
        cumulative = np.cumsum(energy) / total_energy
        idx = int(np.searchsorted(cumulative, self.explained_variance_threshold, side="left"))
        rank = min(max(self.min_rank, idx + 1), values.size)
        return rank, float(cumulative[rank - 1])

    def _store_spectral_basis(
        self,
        U: np.ndarray,
        S: np.ndarray,
        Vt: np.ndarray,
        scores: np.ndarray,
    ) -> None:
        self.left_basis_ = U
        self.singular_values_ = S
        self.right_basis_ = Vt
        if self.embedding_mode == "sv_scaled":
            self.sample_embedding_ = U * S.reshape(1, -1)
        else:
            self.sample_embedding_ = U
        self.leverage_scores_ = scores

    def _fit_clusters_and_partitions(
        self,
        scores: np.ndarray,
        target: Optional[Union[np.ndarray, pd.Series]],
    ) -> None:
        if self.sample_embedding_ is None:
            raise RuntimeError("Sample embedding is not available")
        labels = self._fit_cluster_labels(self.sample_embedding_, target)
        self.cluster_labels_ = labels
        self._build_partitions_from_labels(labels, scores, target)

    def _fit_cluster_labels(
        self,
        embedding: np.ndarray,
        target: Optional[Union[np.ndarray, pd.Series]],
    ) -> np.ndarray:
        if self.partition_selection_method == "auto":
            return self._fit_auto_partition_clusters(embedding, target)
        n_clusters = min(self.n_partitions, embedding.shape[0])
        self.clusterer_ = self._make_kmeans(n_clusters=n_clusters)
        labels = self.clusterer_.fit_predict(embedding)
        self.cluster_centers_ = self.clusterer_.cluster_centers_
        self.partition_selection_info_ = PartitionSelectionInfo(
            requested_n_partitions=int(self.n_partitions),
            selected_n_partitions=int(n_clusters),
            partition_selection_method="fixed",
            selected_algorithm="kmeans",
            cluster_algorithms=("kmeans",),
            cluster_selection_metric="fixed",
            cluster_ensemble_method="none",
            candidates=(int(n_clusters),),
            scores=((int(n_clusters), None),),
            candidate_details=(),
            candidate_plan={},
            candidate_failures=(),
            min_auto_partition_size=int(self.min_auto_partition_size),
            selection_sample_size=0,
        )
        return labels

    def _fit_auto_partition_clusters(
        self,
        embedding: np.ndarray,
        target: Optional[Union[np.ndarray, pd.Series]],
    ) -> np.ndarray:
        result = self.cluster_selector_.select(embedding, target=target)
        self.clusterer_ = result.estimator
        self.cluster_centers_ = result.centers
        self.partition_selection_info_ = self._partition_info_from_selection(result)
        return result.labels

    def _partition_info_from_selection(self, result: ClusterSelectionResult) -> PartitionSelectionInfo:
        candidate_counts = tuple(
            sorted({int(candidate.n_clusters) for candidate in result.candidates})
        )
        candidate_scores = tuple(
            (int(candidate.n_clusters), float(candidate.score))
            for candidate in result.candidates
        )
        candidate_details = tuple(result.diagnostics.get("candidates", ()))
        candidate_plan = result.candidate_plan.to_dict()
        candidate_failures = tuple(
            failure.to_dict() for failure in result.candidate_failures
        )
        return PartitionSelectionInfo(
            requested_n_partitions=int(self.n_partitions),
            selected_n_partitions=int(result.selected_n_clusters),
            partition_selection_method="auto",
            selected_algorithm=result.selected_algorithm,
            cluster_algorithms=tuple(result.diagnostics.get("cluster_algorithms", self.cluster_algorithms)),
            cluster_selection_metric=str(result.diagnostics.get("cluster_selection_metric", self.cluster_selection_metric)),
            cluster_ensemble_method=str(result.diagnostics.get("cluster_ensemble_method", self.cluster_ensemble_method)),
            candidates=candidate_counts,
            scores=candidate_scores,
            candidate_details=candidate_details,
            candidate_plan=candidate_plan,
            candidate_failures=candidate_failures,
            min_auto_partition_size=int(self.min_auto_partition_size),
            selection_sample_size=int(min(self.partition_selection_sample_size, result.labels.shape[0])),
        )

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
        if self.clusterer_ is None or self.sample_embedding_ is None or self.cluster_centers_ is None:
            raise RuntimeError("Sampler not fitted. Call fit() first.")
        if not self.partition_names_:
            raise RuntimeError("No partitions available. Call fit() first.")

        X_num = self._transform_features(X)
        M_new = self._build_mode0_unfolding(X_num, fit=False, rng=None)
        embedding = self._project_new_unfolding(M_new)

        active_cluster_ids = np.asarray([self.partition_to_cluster_[name] for name in self.partition_names_], dtype=int)
        active_centroids = self.cluster_centers_[active_cluster_ids]
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
            self.view_specs_ = self._make_view_specs(n_features, rng, n_views=self.n_views)
            self._store_static_or_coverage_n_views_info(n_samples, n_features)

        self._check_unfolding_size(n_samples, sum(spec.output_dim for spec in self.view_specs_))
        return self._get_rmt_backend().build_mode0_unfolding(X, self.view_specs_)

    def _make_view_specs(
        self,
        n_features: int,
        rng: np.random.Generator,
        n_views: Optional[int] = None,
    ) -> List[ViewSpec]:
        n_views = int(self.n_views if n_views is None else n_views)
        view_size = self._resolve_view_size(n_features)
        projection_dim = self.projection_dim or view_size
        projection_dim = max(1, int(min(projection_dim, view_size if self.view_strategy == "subsample" else projection_dim)))

        specs: List[ViewSpec] = []
        for _ in range(n_views):
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

    def _resolve_n_views_policy(self) -> str:
        if self.n_views_policy != "auto":
            return self.n_views_policy
        if self.view_strategy == "subsample":
            return "coverage"
        return "spectrum_stability"

    def _resolve_coverage_n_views(self, n_samples: int, n_features: int) -> int:
        view_size = min(self._resolve_view_size(n_features), n_features)
        if view_size >= n_features:
            needed = 1
        elif self.target_feature_coverage >= 1.0:
            needed = self.max_views
        else:
            miss_probability = 1.0 - (view_size / max(n_features, 1))
            needed = int(math.ceil(math.log(1.0 - self.target_feature_coverage) / math.log(miss_probability)))
        output_dim = self._output_dim_per_view(n_features)
        cap = self._max_views_by_unfolding_elements(n_samples, output_dim)
        upper = self.max_views if cap is None else min(self.max_views, cap)
        return max(1, min(max(self.min_views, needed), upper))

    def _build_spectrum_stable_fit_unfolding(self, X: np.ndarray, rng: np.random.Generator) -> Any:
        n_samples, n_features = X.shape
        output_dim = self._output_dim_per_view(n_features)
        cap = self._max_views_by_unfolding_elements(n_samples, output_dim)
        upper = self.max_views if cap is None else min(self.max_views, cap)
        upper = max(1, upper)
        lower = max(1, min(self.min_views, upper))
        candidates = self._spectrum_candidate_views(lower, upper)

        max_specs = self._make_view_specs(n_features, rng, n_views=max(candidates))
        previous_spectrum: Optional[np.ndarray] = None
        selected_specs = max_specs[: candidates[-1]]
        selected_M = None
        selected_change: Optional[float] = None

        for candidate in candidates:
            specs = max_specs[:candidate]
            width = sum(spec.output_dim for spec in specs)
            self._check_unfolding_size(n_samples, width)
            M = self._get_rmt_backend().build_mode0_unfolding(X, specs)
            initial_rank = self._resolve_initial_rank(*self._matrix_shape(M))
            basis = self._get_rmt_backend().compute_spectral_basis(M, rank=initial_rank)
            spectrum = np.asarray(basis.singular_values, dtype=np.float64)
            change = None
            if previous_spectrum is not None:
                change = self._spectrum_relative_change(previous_spectrum, spectrum)
            selected_specs = specs
            selected_M = M
            selected_change = change
            if change is not None and change <= self.spectrum_stability_tolerance:
                break
            previous_spectrum = spectrum

        if selected_M is None:
            raise RuntimeError("Spectrum-stability n_views selection did not build an unfolding")

        self.view_specs_ = selected_specs
        self.n_views = len(selected_specs)
        self.n_views_selection_info_ = NViewsSelectionInfo(
            requested_n_views=self.n_views_requested,
            resolved_n_views=int(self.n_views),
            n_views_policy="spectrum_stability",
            target_feature_coverage=None,
            estimated_feature_coverage=1.0,
            max_views_by_unfolding=cap,
            spectrum_stability_tolerance=float(self.spectrum_stability_tolerance),
            spectrum_stability_change=selected_change,
            spectrum_stability_candidates=tuple(candidates),
        )
        return selected_M

    @staticmethod
    def _spectrum_relative_change(previous: np.ndarray, current: np.ndarray) -> float:
        previous = np.asarray(previous, dtype=np.float64)
        current = np.asarray(current, dtype=np.float64)
        k = int(min(previous.size, current.size))
        if k == 0:
            return 0.0
        previous = previous[:k]
        current = current[:k]
        previous_norm = previous / (np.linalg.norm(previous) + 1e-12)
        current_norm = current / (np.linalg.norm(current) + 1e-12)
        return float(np.linalg.norm(current_norm - previous_norm) / (np.linalg.norm(previous_norm) + 1e-12))

    @staticmethod
    def _spectrum_candidate_views(min_views: int, max_views: int) -> List[int]:
        candidates = [int(min_views)]
        while candidates[-1] < max_views:
            candidates.append(min(max_views, candidates[-1] * 2))
        return sorted(set(candidates))

    def _store_static_or_coverage_n_views_info(self, n_samples: int, n_features: int) -> None:
        if self.n_views_selection_info_ is not None:
            return
        output_dim = self._output_dim_per_view(n_features)
        cap = self._max_views_by_unfolding_elements(n_samples, output_dim)
        policy = "static" if self.n_views_requested != "auto" else self._resolve_n_views_policy()
        coverage = None
        target = None
        if policy == "coverage":
            target = float(self.target_feature_coverage)
            coverage = self._estimate_feature_coverage(n_features, int(self.n_views))
        self.n_views_selection_info_ = NViewsSelectionInfo(
            requested_n_views=self.n_views_requested,
            resolved_n_views=int(self.n_views),
            n_views_policy=policy,
            target_feature_coverage=target,
            estimated_feature_coverage=coverage,
            max_views_by_unfolding=cap,
            spectrum_stability_tolerance=None,
            spectrum_stability_change=None,
            spectrum_stability_candidates=(),
        )

    def _estimate_feature_coverage(self, n_features: int, n_views: int) -> float:
        if n_features <= 0:
            return 0.0
        view_size = min(self._resolve_view_size(n_features), n_features)
        if view_size >= n_features:
            return 1.0
        return float(1.0 - (1.0 - view_size / n_features) ** n_views)

    def _output_dim_per_view(self, n_features: int) -> int:
        view_size = self._resolve_view_size(n_features)
        projection_dim = self.projection_dim or view_size
        if self.view_strategy == "subsample":
            projection_dim = min(int(projection_dim), view_size)
        return max(1, int(projection_dim))

    def _max_views_by_unfolding_elements(self, n_samples: int, output_dim_per_view: int) -> Optional[int]:
        if self.max_unfolding_elements is None:
            return None
        denominator = max(1, int(n_samples) * int(output_dim_per_view))
        return max(1, int(self.max_unfolding_elements) // denominator)

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
        embedding = self._get_rmt_backend().project_new_unfolding(
            M_new,
            self.right_basis_,
            self.singular_values_,
        )
        if self.embedding_mode == "sv_scaled":
            embedding = embedding * self.singular_values_.reshape(1, -1)
        return embedding

    def _routing_probability(self, embedding: np.ndarray, active_centroids: np.ndarray) -> np.ndarray:
        return self._get_rmt_backend().routing_probability(
            embedding,
            active_centroids,
            self.routing_temperature,
        )

    def _build_diagnostics(self, M: Any, rank_info: RankSelectionInfo) -> None:
        scores = self.leverage_scores_
        entropy = None
        eff_n = None
        if scores is not None:
            entropy = float(-np.sum(scores * np.log(scores + 1e-12)))
            eff_n = float(np.exp(entropy))
        partition_info = self.partition_selection_info_
        null_result = self.spectral_null_diagnostic_result_
        null_diagnostic = null_result.to_dict()
        subspace_result = self.spectral_subspace_diagnostic_result_
        subspace_diagnostic = subspace_result.to_dict()
        full_rank_subspace = subspace_result.comparison_rank_diagnostic
        self.diagnostics_ = {
            "backend": self.backend_,
            "device": self.device if self.backend_ == "torch" else None,
            "dtype": self.dtype,
            "mode0_unfolding_shape": tuple(map(int, M.shape)),
            "raw_encoded_feature_count": self.raw_encoded_feature_count_,
            "encoded_feature_count": self.encoded_feature_count_,
            "encoded_feature_cap_applied": self.encoded_feature_subset_ is not None,
            "view_strategy": self.view_strategy,
            "embedding_mode": self.embedding_mode,
            "n_views_requested": self.n_views_requested,
            "n_views": int(self.n_views),
            "n_views_policy": self.n_views_selection_info_.n_views_policy if self.n_views_selection_info_ else None,
            "target_feature_coverage": self.n_views_selection_info_.target_feature_coverage if self.n_views_selection_info_ else None,
            "estimated_feature_coverage": self.n_views_selection_info_.estimated_feature_coverage if self.n_views_selection_info_ else None,
            "max_views_by_unfolding": self.n_views_selection_info_.max_views_by_unfolding if self.n_views_selection_info_ else None,
            "spectrum_stability_tolerance": self.n_views_selection_info_.spectrum_stability_tolerance if self.n_views_selection_info_ else None,
            "spectrum_stability_change": self.n_views_selection_info_.spectrum_stability_change if self.n_views_selection_info_ else None,
            "spectrum_stability_candidates": list(self.n_views_selection_info_.spectrum_stability_candidates) if self.n_views_selection_info_ else [],
            "initial_rank": int(rank_info.initial_rank),
            "selected_rank": int(rank_info.selected_rank),
            "rank_by_explained_variance": int(rank_info.selected_rank),
            "rank_by_null_edge": null_result.rank_by_null_edge,
            "rank_by_stability": null_result.rank_by_stability,
            "rank_by_subspace_stability": (
                subspace_result.rank_by_subspace_stability
            ),
            "selected_rank_reason": "explained_variance",
            "rank_selection_method": rank_info.rank_selection_method,
            "explained_variance_threshold": rank_info.explained_variance_threshold,
            "explained_variance_at_selected_rank": rank_info.explained_variance_at_selected_rank,
            "initial_singular_values": (
                self.initial_singular_values_.tolist()
                if self.initial_singular_values_ is not None
                else []
            ),
            "singular_values": self.singular_values_.tolist() if self.singular_values_ is not None else [],
            "null_model_status": null_result.status.value,
            "null_primary_policy": null_result.primary_policy.value,
            "null_empirical_bulk_edge": null_result.empirical_bulk_edge,
            "null_stable_outlier_count": null_result.rank_by_stability,
            "null_max_outlier_excess": null_result.max_outlier_excess,
            "null_successful_resamples": null_result.successful_resamples,
            "spectral_null_diagnostic": null_diagnostic,
            "subspace_stability_status": subspace_result.status.value,
            "subspace_comparison_rank": subspace_result.comparison_rank,
            "subspace_rank_source": subspace_result.rank_source,
            "subspace_max_angle_quantile_degrees": (
                full_rank_subspace.max_angle_quantile_degrees
                if full_rank_subspace is not None
                else None
            ),
            "subspace_normalized_projection_distance_quantile": (
                full_rank_subspace.normalized_projection_distance_quantile
                if full_rank_subspace is not None
                else None
            ),
            "subspace_stability_frequency": (
                full_rank_subspace.stability_frequency
                if full_rank_subspace is not None
                else None
            ),
            "subspace_successful_resamples": (
                subspace_result.successful_resamples
            ),
            "spectral_subspace_diagnostic": subspace_diagnostic,
            "leverage_entropy": entropy,
            "effective_sample_count": eff_n,
            "n_partitions_requested": int(self.n_partitions),
            "selected_n_partitions": int(partition_info.selected_n_partitions) if partition_info else int(len(self.partitions)),
            "partition_selection_method": partition_info.partition_selection_method if partition_info else "fixed",
            "selected_cluster_algorithm": partition_info.selected_algorithm if partition_info else None,
            "cluster_algorithms": list(partition_info.cluster_algorithms) if partition_info else [],
            "cluster_selection_metric": partition_info.cluster_selection_metric if partition_info else None,
            "cluster_ensemble_method": partition_info.cluster_ensemble_method if partition_info else None,
            "partition_selection_candidates": list(partition_info.candidates) if partition_info else [],
            "partition_selection_scores": {
                str(candidate): score
                for candidate, score in partition_info.scores
            } if partition_info else {},
            "partition_selection_candidate_details": list(partition_info.candidate_details) if partition_info else [],
            "partition_selection_candidate_plan": dict(partition_info.candidate_plan) if partition_info else {},
            "partition_selection_candidate_failures": list(partition_info.candidate_failures) if partition_info else [],
            "max_cluster_imbalance_ratio": float(self.max_cluster_imbalance_ratio),
            "min_cluster_fraction": float(self.min_cluster_fraction),
            "min_auto_partition_size": int(partition_info.min_auto_partition_size) if partition_info else None,
            "partition_selection_sample_size": int(partition_info.selection_sample_size) if partition_info else None,
            "n_partitions": int(len(self.partitions)),
            "chunk_sizes": {name: int(len(idx)) for name, idx in self.partitions.items()},
        }
