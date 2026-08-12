from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from time import perf_counter
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from .backend.matrix_backend import MatrixRMTBackend
from .backend.tensor_backend import TensorRMTBackend
from .base_sampler import SpectralSamplerBase
from .bulk_spike import (
    ComponentSplitStatus,
    RowSpectralParticipationContract,
    SpectralComponentSplitContract,
    build_spectral_component_split,
    compute_row_spectral_participation,
)
from .bulk_spike_topology import (
    BulkSpikeExpertTopologySpec,
    BulkSpikeHierarchicalRouter,
    BulkSpikePartitionContract,
    BulkSpikeTopologyBuilder,
    BulkSpikeTopologyMode,
)
from .cluster_selection import ClusterSelectionResult, SpectralClusterSelector
from .classification_sampling import (
    ClassCoverageSelectionPlan,
    infer_target_type,
    select_class_aware_partition_indices,
)
from .partition_sampling import (
    build_partition_sketch_plan,
    partition_membership_fingerprint,
)
from .leverage_sketch import evaluate_subspace_preservation
from .sketch_contracts import (
    ExactBudgetSketchPlan,
    SubspacePreservationContract,
)
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
from .routing_contracts import (
    PartitionGeometryContract,
    PartitionGeometrySpec,
    RoutingDistanceContract,
    RoutingValueKind,
    RoutingWeightContract,
)
from .routing_geometry import PartitionGeometryBuilder, route_partition_geometry
from ...utils.progress import progress_bar
from ...utils.utils import safe_index
from ...experiment.budgeting import PartitionBudgetPlan, build_partition_budget_plan


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
    cluster_target_type: str = "auto"
    missing_class_penalty_weight: float = 0.25
    single_class_penalty_weight: float = 0.50
    class_distribution_drift_weight: float = 0.25
    class_coverage_policy: str = "auto"
    min_samples_per_class: int = 1
    class_allocation_policy: str = "minimum_then_global"
    validation_proxy_fraction: float = 0.2
    validation_proxy_min_partition_rows: int = 8
    validation_proxy_smoothing: float = 1.0
    sampling_budget_ratio: float = 1.0
    budget_feasibility_mode: str = "off"
    min_sampled_rows_per_partition: int = 1
    budget_max_imbalance_ratio: Optional[float] = None
    budget_min_partition_fraction: float = 0.0
    include_single_partition_candidate: bool = False
    downstream_proxy_model_factory: Optional[Callable[[], Any]] = field(
        default=None,
        repr=False,
        compare=False,
    )
    downstream_proxy_shortlist_size: int = 3
    downstream_complexity_penalty_weight: float = 0.01
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
    component_diagnostic_enabled: bool = False
    expert_topology: str = "standard"
    topology_max_experts: int = 5
    topology_min_partition_size: int = 32
    topology_budget_policy: str = "excess_energy"
    topology_fixed_spike_share: float = 0.25
    topology_min_spike_share: float = 0.10
    topology_max_spike_share: float = 0.50
    topology_signal_threshold: float = 0.50
    topology_spike_router: str = "gmm_posterior"
    topology_covariance_shrinkage: float = 0.10
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
    leverage_cap_quantile: float = 0.95
    leverage_mixture_alpha: float = 0.25
    leverage_uniform_floor: float = 1e-12
    ridge_leverage_lambda: Union[float, str] = "auto"
    training_reweighting: str = "none"
    routing_representation: str = "source_centroid"
    routing_metric: str = "squared_euclidean"
    routing_kernel: str = "softmax"
    routing_temperature: float = 1.0
    routing_shrinkage: float = 0.05
    routing_covariance_shrinkage: float = 0.10
    routing_min_scale: float = 1e-8
    routing_min_prior: float = 1e-8
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
    cluster_target_type: str
    resolved_cluster_target_type: str
    candidates: Tuple[int, ...]
    scores: Tuple[Tuple[int, Optional[float]], ...]
    selected_candidate: Dict[str, Any]
    candidate_details: Tuple[Dict[str, Any], ...]
    candidate_plan: Dict[str, Any]
    candidate_failures: Tuple[Dict[str, Any], ...]
    consensus: Dict[str, Any]
    validation_proxy_plan: Dict[str, Any]
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
            leverage_cap_quantile=cfg.leverage_cap_quantile,
            leverage_mixture_alpha=cfg.leverage_mixture_alpha,
            leverage_uniform_floor=cfg.leverage_uniform_floor,
            ridge_leverage_lambda=cfg.ridge_leverage_lambda,
            training_reweighting=cfg.training_reweighting,
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
            (
                "silhouette",
                "balanced_silhouette",
                "validation_proxy",
                "budget_aware_validation_proxy",
                "downstream_proxy",
            ),
        )
        self.cluster_ensemble_method = self._validate_choice(
            "cluster_ensemble_method",
            cfg.cluster_ensemble_method,
            ("best_score", "weighted_vote", "coassociation"),
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
        self.cluster_target_type = self._validate_choice(
            "cluster_target_type",
            cfg.cluster_target_type,
            ("auto", "regression", "classification"),
        )
        self.missing_class_penalty_weight = self._validate_nonnegative_float(
            "missing_class_penalty_weight",
            cfg.missing_class_penalty_weight,
        )
        self.single_class_penalty_weight = self._validate_nonnegative_float(
            "single_class_penalty_weight",
            cfg.single_class_penalty_weight,
        )
        self.class_distribution_drift_weight = self._validate_nonnegative_float(
            "class_distribution_drift_weight",
            cfg.class_distribution_drift_weight,
        )
        self.class_coverage_policy = self._validate_choice(
            "class_coverage_policy",
            cfg.class_coverage_policy,
            ("auto", "off", "preserve_local_classes"),
        )
        self.min_samples_per_class = self._validate_positive_int(
            "min_samples_per_class",
            cfg.min_samples_per_class,
        )
        self.class_allocation_policy = self._validate_choice(
            "class_allocation_policy",
            cfg.class_allocation_policy,
            ("minimum_then_global", "proportional"),
        )
        self.validation_proxy_fraction = self._validate_fraction(
            "validation_proxy_fraction",
            cfg.validation_proxy_fraction,
        )
        if self.validation_proxy_fraction >= 0.5:
            raise ValueError("validation_proxy_fraction must be in (0, 0.5)")
        self.validation_proxy_min_partition_rows = self._validate_positive_int(
            "validation_proxy_min_partition_rows",
            cfg.validation_proxy_min_partition_rows,
        )
        self.validation_proxy_smoothing = self._validate_positive_float(
            "validation_proxy_smoothing",
            cfg.validation_proxy_smoothing,
        )
        self.sampling_budget_ratio = self._validate_fraction(
            "sampling_budget_ratio",
            cfg.sampling_budget_ratio,
        )
        self.budget_feasibility_mode = self._validate_choice(
            "budget_feasibility_mode",
            cfg.budget_feasibility_mode,
            ("off", "hard"),
        )
        self.min_sampled_rows_per_partition = self._validate_positive_int(
            "min_sampled_rows_per_partition",
            cfg.min_sampled_rows_per_partition,
        )
        self.budget_max_imbalance_ratio = (
            None
            if cfg.budget_max_imbalance_ratio is None
            else self._validate_positive_float(
                "budget_max_imbalance_ratio",
                cfg.budget_max_imbalance_ratio,
            )
        )
        self.budget_min_partition_fraction = self._validate_nonnegative_fraction(
            "budget_min_partition_fraction",
            cfg.budget_min_partition_fraction,
        )
        self.include_single_partition_candidate = bool(
            cfg.include_single_partition_candidate
        )
        self.downstream_proxy_model_factory = cfg.downstream_proxy_model_factory
        self.downstream_proxy_shortlist_size = self._validate_positive_int(
            "downstream_proxy_shortlist_size",
            cfg.downstream_proxy_shortlist_size,
        )
        self.downstream_complexity_penalty_weight = (
            self._validate_nonnegative_float(
                "downstream_complexity_penalty_weight",
                cfg.downstream_complexity_penalty_weight,
            )
        )
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
        self.ridge_leverage_lambda = self._normalize_ridge_leverage_lambda(
            cfg.ridge_leverage_lambda
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
        self.component_diagnostic_enabled = bool(cfg.component_diagnostic_enabled)
        if self.component_diagnostic_enabled and not null_config.enabled:
            raise ValueError(
                "component_diagnostic_enabled=True requires "
                "null_diagnostic_enabled=True"
            )
        self.expert_topology = self._validate_choice(
            "expert_topology",
            cfg.expert_topology,
            ("standard", "bulk_single_spike", "bulk_multi_spike"),
        )
        self.topology_spec_ = BulkSpikeExpertTopologySpec(
            name=self.expert_topology,
            mode={
                "standard": "standard",
                "bulk_single_spike": "single_spike",
                "bulk_multi_spike": "multi_spike",
            }[self.expert_topology],
            max_experts=cfg.topology_max_experts,
            min_partition_size=cfg.topology_min_partition_size,
            budget_policy=cfg.topology_budget_policy,
            fixed_spike_share=cfg.topology_fixed_spike_share,
            min_spike_share=cfg.topology_min_spike_share,
            max_spike_share=cfg.topology_max_spike_share,
            signal_threshold=cfg.topology_signal_threshold,
            spike_router=cfg.topology_spike_router,
            covariance_shrinkage=cfg.topology_covariance_shrinkage,
        )
        if self.expert_topology != "standard" and not self.component_diagnostic_enabled:
            raise ValueError(
                "bulk/spike expert_topology requires component_diagnostic_enabled=True"
            )
        self.topology_builder_ = BulkSpikeTopologyBuilder()
        self.hierarchical_router_ = BulkSpikeHierarchicalRouter()
        self.bulk_spike_topology_: Optional[BulkSpikePartitionContract] = None
        self.spike_partition_geometry_: Optional[PartitionGeometryContract] = None
        self.initial_right_basis_: Optional[np.ndarray] = None
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
        self.routing_geometry_spec = PartitionGeometrySpec(
            representation=cfg.routing_representation,
            metric=cfg.routing_metric,
            kernel=cfg.routing_kernel,
            temperature=cfg.routing_temperature,
            covariance_shrinkage=cfg.routing_covariance_shrinkage,
            uniform_shrinkage=cfg.routing_shrinkage,
            min_scale=cfg.routing_min_scale,
            min_prior=cfg.routing_min_prior,
        )
        self.geometry_builder_ = PartitionGeometryBuilder()
        self.oversample_factor = int(cfg.oversample_factor)
        self.power_iterations = int(cfg.power_iterations)
        self.max_unfolding_elements = cfg.max_unfolding_elements
        self._rmt_backend: Optional[Union[MatrixRMTBackend, TensorRMTBackend]] = None
        self.cluster_selector_ = self._make_cluster_selector()
        self.cluster_centers_: Optional[np.ndarray] = None
        self.initial_singular_values_: Optional[np.ndarray] = None
        self.initial_left_basis_: Optional[np.ndarray] = None
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
        self.partition_budget_plan_: Optional[PartitionBudgetPlan] = None
        self.class_coverage_selection_plans_: Dict[
            str, ClassCoverageSelectionPlan
        ] = {}
        self.resolved_class_coverage_policy_ = "off"
        self.class_coverage_guaranteed_ = False
        self.partition_geometry_: Optional[PartitionGeometryContract] = None
        self.last_routing_distances_: Optional[RoutingDistanceContract] = None
        self.last_routing_weights_: Optional[RoutingWeightContract] = None
        self.partition_subspace_preservation_: Dict[
            str, SubspacePreservationContract
        ] = {}
        self.spectral_component_split_ = (
            SpectralComponentSplitContract.unavailable(
                (),
                status=ComponentSplitStatus.DISABLED,
                min_selection_frequency=null_config.min_selection_frequency,
                reason="component diagnostics are disabled",
            )
        )
        self.row_spectral_participation_: Optional[
            RowSpectralParticipationContract
        ] = None

    @staticmethod
    def _normalize_ridge_leverage_lambda(value: Union[float, str]) -> Union[float, str]:
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized != "auto":
                raise ValueError(
                    "ridge_leverage_lambda must be a non-negative float or 'auto'"
                )
            return normalized
        normalized = float(value)
        if not np.isfinite(normalized) or normalized < 0.0:
            raise ValueError(
                "ridge_leverage_lambda must be a non-negative float or 'auto'"
            )
        return normalized

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

    @staticmethod
    def _validate_nonnegative_float(name: str, value: float) -> float:
        value = float(value)
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be a non-negative finite value")
        return value

    @staticmethod
    def _validate_nonnegative_fraction(name: str, value: float) -> float:
        normalized = float(value)
        if not np.isfinite(normalized) or not 0 <= normalized <= 1:
            raise ValueError(f"{name} must be in [0, 1]")
        return normalized

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
            target_type=self.cluster_target_type,
            missing_class_penalty_weight=self.missing_class_penalty_weight,
            single_class_penalty_weight=self.single_class_penalty_weight,
            class_distribution_drift_weight=(
                self.class_distribution_drift_weight
            ),
            validation_proxy_fraction=self.validation_proxy_fraction,
            validation_proxy_min_partition_rows=(
                self.validation_proxy_min_partition_rows
            ),
            validation_proxy_smoothing=self.validation_proxy_smoothing,
            sampling_budget_ratio=self.sampling_budget_ratio,
            budget_feasibility_mode=self.budget_feasibility_mode,
            min_sampled_rows_per_partition=(
                self.min_sampled_rows_per_partition
            ),
            budget_max_imbalance_ratio=self.budget_max_imbalance_ratio,
            budget_min_partition_fraction=(
                self.budget_min_partition_fraction
            ),
            include_single_partition_candidate=(
                self.include_single_partition_candidate
            ),
            downstream_proxy_model_factory=(
                self.downstream_proxy_model_factory
            ),
            downstream_proxy_shortlist_size=(
                self.downstream_proxy_shortlist_size
            ),
            downstream_complexity_penalty_weight=(
                self.downstream_complexity_penalty_weight
            ),
            selection_method=self.selection_method,
            leverage_cap_quantile=self.leverage_cap_quantile,
            routing_temperature=self.routing_temperature,
            routing_shrinkage=self.routing_shrinkage,
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
        fit_started = perf_counter()
        stage_seconds: Dict[str, float] = {}
        with progress_bar(
            enabled=self.show_progress,
            desc="RMT sampler fit",
            total=9,
        ) as stage:
            started = perf_counter()
            rng = self._start_fit()
            stage_seconds["initialize"] = perf_counter() - started
            started = perf_counter()
            X_num = self._fit_transform_features(data)
            stage_seconds["preprocessing"] = perf_counter() - started
            stage.update(1)
            started = perf_counter()
            M = self._build_fit_unfolding(X_num, rng)
            stage_seconds["unfolding_and_view_selection"] = perf_counter() - started
            stage.update(1)
            started = perf_counter()
            U, S, Vt, scores, rank = self._fit_spectral_basis(M)
            stage_seconds["spectral_basis"] = perf_counter() - started
            stage.update(1)
            self._store_spectral_basis(U, S, Vt, scores)
            started = perf_counter()
            self._fit_spectral_null_diagnostic(X_num)
            stage_seconds["null_diagnostics"] = perf_counter() - started
            stage.update(1)
            started = perf_counter()
            self._fit_component_diagnostics()
            stage_seconds["component_diagnostics"] = perf_counter() - started
            stage.update(1)
            started = perf_counter()
            self._fit_spectral_subspace_diagnostic(X_num)
            stage_seconds["subspace_diagnostics"] = perf_counter() - started
            stage.update(1)
            started = perf_counter()
            self._fit_clusters_and_partitions(X_num, scores, target)
            self._fit_bulk_spike_topology(target)
            stage_seconds["partition_selection_and_sampling"] = perf_counter() - started
            stage.update(1)
            started = perf_counter()
            self._fit_partition_geometry()
            stage_seconds["partition_geometry"] = perf_counter() - started
            stage.update(1)
            started = perf_counter()
            self._build_diagnostics(M, rank)
            stage_seconds["diagnostics"] = perf_counter() - started
            stage.update(1)
        self.runtime_diagnostics_ = {
            "stage_seconds": {
                name: float(value) for name, value in stage_seconds.items()
            },
            "total_seconds": float(perf_counter() - fit_started),
            "cold_start": True,
        }
        self.diagnostics_["runtime"] = dict(self.runtime_diagnostics_)
        return self

    def _start_fit(self) -> np.random.Generator:
        self.backend_ = self._resolve_backend()
        self._rmt_backend = self._make_rmt_backend()
        self.view_specs_ = []
        self.initial_singular_values_ = None
        self.initial_left_basis_ = None
        self.initial_right_basis_ = None
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
        self.partition_budget_plan_ = None
        self.class_coverage_selection_plans_ = {}
        self.partition_training_weights_ = {}
        self.partition_inclusion_probabilities_ = {}
        self.partition_sketch_plans_ = {}
        self.partition_subspace_preservation_ = {}
        self.resolved_class_coverage_policy_ = "off"
        self.class_coverage_guaranteed_ = False
        self.cluster_centers_ = None
        self.partition_geometry_ = None
        self.bulk_spike_topology_ = None
        self.spike_partition_geometry_ = None
        self.last_routing_distances_ = None
        self.last_routing_weights_ = None
        self.spectral_component_split_ = (
            SpectralComponentSplitContract.unavailable(
                (),
                status=ComponentSplitStatus.DISABLED,
                min_selection_frequency=(
                    self.spectral_null_diagnostic.config.min_selection_frequency
                ),
                reason="component diagnostics are disabled",
            )
        )
        self.row_spectral_participation_ = None
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
        self.initial_left_basis_ = np.asarray(basis.U, dtype=np.float64).copy()
        self.initial_right_basis_ = np.asarray(basis.Vt, dtype=np.float64).copy()
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

    def _fit_component_diagnostics(self) -> None:
        singular = (
            np.asarray(self.initial_singular_values_, dtype=float)
            if self.initial_singular_values_ is not None
            else np.asarray([], dtype=float)
        )
        threshold = (
            self.spectral_null_diagnostic.config.min_selection_frequency
        )
        if not self.component_diagnostic_enabled:
            self.spectral_component_split_ = (
                SpectralComponentSplitContract.unavailable(
                    singular,
                    status=ComponentSplitStatus.DISABLED,
                    min_selection_frequency=threshold,
                    reason="component diagnostics are disabled",
                )
            )
            self.row_spectral_participation_ = None
            return
        if self.initial_left_basis_ is None:
            raise RuntimeError("Initial left spectral basis is not available")
        split = build_spectral_component_split(
            singular,
            self.spectral_null_diagnostic_result_,
            min_selection_frequency=threshold,
        )
        self.spectral_component_split_ = split
        self.row_spectral_participation_ = compute_row_spectral_participation(
            self.initial_left_basis_,
            split,
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
        features: np.ndarray,
        scores: np.ndarray,
        target: Optional[Union[np.ndarray, pd.Series]],
    ) -> None:
        if self.sample_embedding_ is None:
            raise RuntimeError("Sample embedding is not available")
        labels = self._fit_cluster_labels(
            self.sample_embedding_,
            target,
            downstream_features=features,
            sample_scores=scores,
        )
        self.cluster_labels_ = labels
        self._build_partitions_from_labels(labels, scores, target)

    def _fit_bulk_spike_topology(
        self,
        target: Optional[Union[np.ndarray, pd.Series]],
    ) -> None:
        if self.expert_topology == "standard":
            return
        if self.row_spectral_participation_ is None:
            raise RuntimeError("Bulk/spike topology requires row spectral participation")
        total_budget = int(sum(len(rows) for rows in self.partitions.values()))
        topology = self.topology_builder_.build(
            self.topology_spec_,
            self.row_spectral_participation_,
            total_budget=total_budget,
            random_state=self.random_state,
        )
        topology = self._repair_topology_class_coverage(topology, target)
        self.bulk_spike_topology_ = topology
        self.partitions = {
            name: rows.copy()
            for name, rows in zip(topology.partition_names, topology.row_indices)
        }
        self.partition_names_ = list(topology.partition_names)
        self.partition_to_cluster_ = {
            name: index for index, name in enumerate(topology.partition_names)
        }
        self.cluster_labels_ = self._bulk_spike_source_labels(topology)
        self.partition_sketch_plans_ = {}
        self.partition_training_weights_ = {}
        self.partition_inclusion_probabilities_ = {}
        self.partition_subspace_preservation_ = {}

    def _repair_topology_class_coverage(
        self,
        topology: BulkSpikePartitionContract,
        target: Optional[Union[np.ndarray, pd.Series]],
    ) -> BulkSpikePartitionContract:
        if target is None or infer_target_type(target, self.cluster_target_type) != "classification":
            self.class_coverage_guaranteed_ = False
            return topology
        values = np.asarray(target).reshape(-1)
        rows = [np.asarray(indices, dtype=int).copy() for indices in topology.row_indices]
        selected = set(np.concatenate(rows).tolist())
        all_indices = np.arange(values.size, dtype=int)
        global_classes = np.unique(values)
        scores = np.asarray(self.leverage_scores_, dtype=float)
        repairs = {}
        for partition_index, (name, indices) in enumerate(
            zip(topology.partition_names, rows)
        ):
            before = np.unique(values[indices])
            removed_rows = []
            added_rows = []
            if indices.size >= global_classes.size:
                missing_classes = [
                    label for label in global_classes if label not in before
                ]
                for missing_class in missing_classes:
                    candidates = np.asarray(
                        [
                            index
                            for index in all_indices
                            if index not in selected
                            and values[index] == missing_class
                        ],
                        dtype=int,
                    )
                    if candidates.size == 0:
                        continue
                    current_values, current_counts = np.unique(
                        values[indices],
                        return_counts=True,
                    )
                    removable_classes = set(
                        current_values[current_counts > 1].tolist()
                    )
                    victim_positions = np.asarray(
                        [
                            position
                            for position, row in enumerate(indices)
                            if values[row] in removable_classes
                        ],
                        dtype=int,
                    )
                    if victim_positions.size == 0:
                        continue
                    replacement = int(candidates[np.argmax(scores[candidates])])
                    victim_position = int(
                        victim_positions[
                            np.argmin(scores[indices[victim_positions]])
                        ]
                    )
                    victim = int(indices[victim_position])
                    indices[victim_position] = replacement
                    selected.remove(victim)
                    selected.add(replacement)
                    removed_rows.append(victim)
                    added_rows.append(replacement)
            rows[partition_index] = np.sort(indices)
            after = np.unique(values[rows[partition_index]])
            if removed_rows or after.size != before.size:
                repairs[name] = {
                    "removed_rows": removed_rows,
                    "added_rows": added_rows,
                    "classes_before": before.tolist(),
                    "classes_after": after.tolist(),
                    "missing_classes_after": [
                        label for label in global_classes if label not in after
                    ],
                }
        self.class_coverage_guaranteed_ = bool(
            rows
            and all(
                np.unique(values[indices]).size == global_classes.size
                for indices in rows
            )
        )
        return BulkSpikePartitionContract(
            topology_name=topology.topology_name,
            mode=topology.mode,
            partition_names=topology.partition_names,
            regimes=topology.regimes,
            row_indices=tuple(rows),
            total_budget=topology.total_budget,
            max_experts=topology.max_experts,
            metadata={
                **dict(topology.metadata),
                "classification_exact_budget_repairs": repairs,
                "global_classes": global_classes.tolist(),
                "class_coverage_guaranteed": self.class_coverage_guaranteed_,
            },
        )

    def _bulk_spike_source_labels(
        self,
        topology: BulkSpikePartitionContract,
    ) -> np.ndarray:
        participation = self.row_spectral_participation_
        if participation is None:
            raise RuntimeError("Row spectral participation is not fitted")
        labels = np.full(participation.n_rows, -1, dtype=int)
        bulk_columns = [
            index for index, regime in enumerate(topology.regimes) if regime == "bulk"
        ]
        spike_columns = [
            index
            for index, regime in enumerate(topology.regimes)
            if regime.startswith("spike")
        ]
        active = participation.signalness >= self.topology_spec_.signal_threshold
        if bulk_columns:
            labels[~active] = bulk_columns[0]
        if len(spike_columns) == 1:
            labels[active] = spike_columns[0]
        elif spike_columns:
            components = [
                int(topology.regimes[index].split(":", 1)[1])
                for index in spike_columns
            ]
            assignments = np.argmax(
                participation.spike_signatures[:, components],
                axis=1,
            )
            labels[active] = np.asarray(spike_columns)[assignments[active]]
        return labels

    def _fit_cluster_labels(
        self,
        embedding: np.ndarray,
        target: Optional[Union[np.ndarray, pd.Series]],
        *,
        downstream_features: np.ndarray,
        sample_scores: np.ndarray,
    ) -> np.ndarray:
        if self.partition_selection_method == "auto":
            return self._fit_auto_partition_clusters(
                embedding,
                target,
                downstream_features=downstream_features,
                sample_scores=sample_scores,
            )
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
            cluster_target_type=self.cluster_target_type,
            resolved_cluster_target_type="not_used",
            candidates=(int(n_clusters),),
            scores=((int(n_clusters), None),),
            selected_candidate={},
            candidate_details=(),
            candidate_plan={},
            candidate_failures=(),
            consensus={},
            validation_proxy_plan={},
            min_auto_partition_size=int(self.min_auto_partition_size),
            selection_sample_size=0,
        )
        return labels

    def _fit_auto_partition_clusters(
        self,
        embedding: np.ndarray,
        target: Optional[Union[np.ndarray, pd.Series]],
        *,
        downstream_features: np.ndarray,
        sample_scores: np.ndarray,
    ) -> np.ndarray:
        result = self.cluster_selector_.select(
            embedding,
            target=target,
            downstream_features=downstream_features,
            sample_scores=sample_scores,
        )
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
            cluster_target_type=str(
                result.diagnostics.get("cluster_target_type", self.cluster_target_type)
            ),
            resolved_cluster_target_type=str(
                result.diagnostics.get("resolved_cluster_target_type", "none")
            ),
            candidates=candidate_counts,
            scores=candidate_scores,
            selected_candidate=dict(
                result.diagnostics.get("selected_candidate") or {}
            ),
            candidate_details=candidate_details,
            candidate_plan=candidate_plan,
            candidate_failures=candidate_failures,
            consensus=dict(result.diagnostics.get("consensus") or {}),
            validation_proxy_plan=dict(
                result.diagnostics.get("validation_proxy_plan") or {}
            ),
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
            name: self._build_partition_payload(name, indices, data, target)
            for name, indices in self.partitions.items()
        }

    def _build_partition_payload(
        self,
        name: str,
        indices: np.ndarray,
        data: ArrayLike,
        target: Optional[Union[np.ndarray, pd.Series]],
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"feature": safe_index(data, indices)}
        if target is not None:
            payload["target"] = safe_index(target, indices)
        if self.training_reweighting == "inverse_probability":
            weights = self.partition_training_weights_.get(name)
            if weights is None or len(weights) != len(indices):
                raise RuntimeError(
                    f"Training weights do not align with partition {name!r}"
                )
            payload["sample_weight"] = np.asarray(weights, dtype=float).copy()
        return payload

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
        """Return routing probabilities aligned with ``partition_names_``."""

        return self.predict_partition_routing(X).weights

    def predict_partition_routing(
        self,
        X: ArrayLike,
        *,
        geometry_spec: Optional[PartitionGeometrySpec] = None,
        temperature: Optional[float] = None,
    ) -> RoutingWeightContract:
        """Route raw rows and retain the typed distance/weight diagnostics."""

        _distances, weights = self.predict_partition_routing_contracts(
            X,
            geometry_spec=geometry_spec,
            temperature=temperature,
        )
        return weights

    def predict_partition_routing_contracts(
        self,
        X: ArrayLike,
        *,
        geometry_spec: Optional[PartitionGeometrySpec] = None,
        temperature: Optional[float] = None,
    ) -> Tuple[RoutingDistanceContract, RoutingWeightContract]:
        """Return both pre-normalization proximity values and routing weights."""

        if self.sample_embedding_ is None:
            raise RuntimeError("Sampler not fitted. Call fit() first.")
        if not self.partition_names_:
            raise RuntimeError("No partitions available. Call fit() first.")
        if self.bulk_spike_topology_ is not None:
            if geometry_spec is not None:
                raise ValueError(
                    "Bulk/spike topology uses its fitted hierarchical routing geometry"
                )
            return self._predict_bulk_spike_routing(X, temperature=temperature)
        geometry = (
            self.partition_geometry_
            if geometry_spec is None
            else self.build_partition_geometry(geometry_spec)
        )
        if geometry is None:
            raise RuntimeError("Partition geometry is not fitted")
        return self.route_embedding(
            self.transform_embedding(X),
            geometry=geometry,
            temperature=temperature,
        )

    def _predict_bulk_spike_routing(
        self,
        X: ArrayLike,
        *,
        temperature: Optional[float],
    ) -> Tuple[RoutingDistanceContract, RoutingWeightContract]:
        topology = self.bulk_spike_topology_
        if topology is None:
            raise RuntimeError("Bulk/spike topology is not fitted")
        participation = self.transform_spectral_participation(X)
        spike_columns = [
            index
            for index, regime in enumerate(topology.regimes)
            if regime.startswith("spike")
        ]
        conditional = None
        if len(spike_columns) > 1:
            if self.spike_partition_geometry_ is None:
                raise RuntimeError("Conditional spike geometry is not fitted")
            components = [
                int(topology.regimes[index].split(":", 1)[1])
                for index in spike_columns
            ]
            _, conditional_contract = route_partition_geometry(
                backend=self._get_rmt_backend(),
                embedding=participation.spike_signatures[:, components],
                geometry=self.spike_partition_geometry_,
                temperature=(
                    self.routing_temperature if temperature is None else temperature
                ),
            )
            conditional = conditional_contract.weights
        weights = self.hierarchical_router_.route(
            topology,
            spike_probability=participation.signalness,
            conditional_spike_weights=conditional,
        )
        distance = RoutingDistanceContract(
            partition_names=weights.partition_names,
            values=-np.log(np.maximum(weights.weights, 1e-12)),
            kind=RoutingValueKind.DISTANCE,
            diagnostics={"source": "bulk_spike_hierarchical_weights"},
        )
        self.last_routing_distances_ = distance
        self.last_routing_weights_ = weights
        return distance, weights

    def transform_spectral_participation(
        self,
        X: ArrayLike,
    ) -> RowSpectralParticipationContract:
        """Project new rows into the initial spectrum used by bulk/spike diagnostics."""

        if (
            self.initial_right_basis_ is None
            or self.initial_singular_values_ is None
        ):
            raise RuntimeError("Initial spectral basis is not fitted")
        X_num = self._transform_features(X)
        unfolding = self._build_mode0_unfolding(X_num, fit=False, rng=None)
        left_coordinates = self._get_rmt_backend().project_new_unfolding(
            unfolding,
            self.initial_right_basis_,
            self.initial_singular_values_,
        )
        return compute_row_spectral_participation(
            left_coordinates,
            self.spectral_component_split_,
        )

    def route_embedding(
        self,
        embedding: np.ndarray,
        *,
        geometry: Optional[PartitionGeometryContract] = None,
        geometry_spec: Optional[PartitionGeometrySpec] = None,
        temperature: Optional[float] = None,
    ) -> Tuple[RoutingDistanceContract, RoutingWeightContract]:
        """Route an already projected embedding without refitting the sampler."""

        if geometry is not None and geometry_spec is not None:
            raise ValueError("Pass either geometry or geometry_spec, not both")
        selected_geometry = geometry
        if selected_geometry is None:
            selected_geometry = (
                self.partition_geometry_
                if geometry_spec is None
                else self.build_partition_geometry(geometry_spec)
            )
        if selected_geometry is None:
            raise RuntimeError("Partition geometry is not fitted")
        distances, weights = route_partition_geometry(
            backend=self._get_rmt_backend(),
            embedding=np.asarray(embedding, dtype=float),
            geometry=selected_geometry,
            temperature=temperature,
        )
        self.last_routing_distances_ = distances
        self.last_routing_weights_ = weights
        return distances, weights

    def build_partition_geometry(
        self,
        spec: Optional[PartitionGeometrySpec] = None,
    ) -> PartitionGeometryContract:
        """Build an alternative immutable geometry from the fitted partitions."""

        if self.sample_embedding_ is None or self.cluster_labels_ is None:
            raise RuntimeError("Sampler not fitted. Call fit() first.")
        if not self.partitions:
            raise RuntimeError("No partitions available. Call fit() first.")
        return self.geometry_builder_.build(
            embedding=self.sample_embedding_,
            cluster_labels=self.cluster_labels_,
            partitions=self.partitions,
            partition_to_cluster=self.partition_to_cluster_,
            spec=self.routing_geometry_spec if spec is None else spec,
        )

    def _fit_partition_geometry(self) -> None:
        if self.bulk_spike_topology_ is not None:
            self.partition_geometry_ = None
            self._fit_spike_partition_geometry()
            return
        self.partition_geometry_ = self.build_partition_geometry(
            self.routing_geometry_spec
        )

    def _fit_spike_partition_geometry(self) -> None:
        topology = self.bulk_spike_topology_
        participation = self.row_spectral_participation_
        if topology is None or participation is None:
            self.spike_partition_geometry_ = None
            return
        spike_columns = [
            index
            for index, regime in enumerate(topology.regimes)
            if regime.startswith("spike")
        ]
        if len(spike_columns) <= 1:
            self.spike_partition_geometry_ = None
            return
        components = [
            int(topology.regimes[index].split(":", 1)[1])
            for index in spike_columns
        ]
        signatures = participation.spike_signatures[:, components]
        partitions = {
            topology.partition_names[index]: topology.row_indices[index]
            for index in spike_columns
        }
        mapping = {
            topology.partition_names[index]: index for index in spike_columns
        }
        router = self.topology_spec_.spike_router
        metric = (
            "diag_shrinkage_mahalanobis"
            if router == "diag_shrinkage_mahalanobis"
            else "full_shrinkage_mahalanobis"
        )
        kernel = "gmm_posterior" if router == "gmm_posterior" else "softmax"
        self.spike_partition_geometry_ = self.geometry_builder_.build(
            embedding=signatures,
            cluster_labels=self.cluster_labels_,
            partitions=partitions,
            partition_to_cluster=mapping,
            spec=PartitionGeometrySpec(
                metric=metric,
                kernel=kernel,
                covariance_shrinkage=self.topology_spec_.covariance_shrinkage,
                uniform_shrinkage=self.routing_geometry_spec.uniform_shrinkage,
                temperature=self.routing_geometry_spec.temperature,
            ),
        )

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

        cluster_indices = {
            int(cluster_id): np.where(labels == cluster_id)[0]
            for cluster_id in sorted(np.unique(labels).tolist())
        }
        capacities = {
            f"chunk_{cluster_id}": self._cluster_selection_capacity(indices.size)
            for cluster_id, indices in cluster_indices.items()
        }
        hard_budget = self.budget_feasibility_mode == "hard"
        self.partition_budget_plan_ = build_partition_budget_plan(
            capacities,
            total_rows=int(labels.size),
            budget_ratio=self.sampling_budget_ratio,
            min_rows_per_partition=(
                self.min_sampled_rows_per_partition if hard_budget else 1
            ),
            max_imbalance_ratio=(
                self.budget_max_imbalance_ratio if hard_budget else None
            ),
            min_partition_fraction=(
                self.budget_min_partition_fraction if hard_budget else 0.0
            ),
        )
        if hard_budget and not self.partition_budget_plan_.feasible:
            raise ValueError(
                "Selected partition candidate is infeasible for the sampling budget: "
                f"{self.partition_budget_plan_.to_dict()['violations']}"
            )
        allocation = self.partition_budget_plan_.allocation_map

        resolved_target_type = infer_target_type(target, self.cluster_target_type)
        self.resolved_class_coverage_policy_ = self._resolve_class_coverage_policy(
            resolved_target_type
        )
        target_values = None if target is None else np.asarray(target).reshape(-1)
        if target_values is not None and target_values.size != labels.size:
            raise ValueError("target must align with cluster labels")

        row_selection_rng = np.random.default_rng(self.random_state)
        for cluster_id, cluster_idx in cluster_indices.items():
            name = f"chunk_{cluster_id}"
            if self.resolved_class_coverage_policy_ == "preserve_local_classes":
                if target_values is None:
                    raise ValueError(
                        "class-aware row selection requires a classification target"
                    )
                coverage_plan = self._select_class_aware_from_cluster(
                    cluster_idx,
                    scores,
                    target_values,
                    target_size=allocation[name],
                    random_state=row_selection_rng,
                )
                self.class_coverage_selection_plans_[name] = coverage_plan
                if not coverage_plan.feasible:
                    raise ValueError(
                        "Classification partition is infeasible for exact-budget "
                        f"class coverage ({name}): {coverage_plan.to_dict()}"
                    )
                selected = coverage_plan.selected_indices
                if coverage_plan.combined_sketch_plan is None:
                    raise RuntimeError("Class-aware selection did not build a sketch plan")
                self._record_partition_sketch(
                    name,
                    coverage_plan.combined_sketch_plan,
                )
            else:
                sketch_plan = self._build_sketch_from_cluster(
                    cluster_idx,
                    scores,
                    target_size=allocation[name],
                    random_state=row_selection_rng,
                )
                self._record_partition_sketch(name, sketch_plan)
                selected = sketch_plan.selected_indices
            if selected.size == 0:
                continue
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
        active_names = set(self.partition_names_)
        self.partition_sketch_plans_ = {
            name: value
            for name, value in self.partition_sketch_plans_.items()
            if name in active_names
        }
        self.partition_training_weights_ = {
            name: value
            for name, value in self.partition_training_weights_.items()
            if name in active_names
        }
        self.partition_inclusion_probabilities_ = {
            name: value
            for name, value in self.partition_inclusion_probabilities_.items()
            if name in active_names
        }
        self.partition_subspace_preservation_ = {
            name: value
            for name, value in self.partition_subspace_preservation_.items()
            if name in active_names
        }
        self.partition_to_cluster_ = {name: int(name.split("_")[-1]) for name in self.partition_names_}
        self.class_coverage_guaranteed_ = bool(
            self.resolved_class_coverage_policy_ == "preserve_local_classes"
            and partitions
            and all(
                plan.feasible
                and len(plan.selected_class_counts) >= 2
                for name, plan in self.class_coverage_selection_plans_.items()
                if name in partitions
            )
        )

    def _resolve_class_coverage_policy(self, resolved_target_type: str) -> str:
        if self.expert_topology != "standard":
            return "off"
        if self.class_coverage_policy == "auto":
            return (
                "preserve_local_classes"
                if resolved_target_type == "classification"
                else "off"
            )
        if (
            self.class_coverage_policy == "preserve_local_classes"
            and resolved_target_type != "classification"
        ):
            raise ValueError(
                "class_coverage_policy='preserve_local_classes' requires a "
                "classification target"
            )
        return self.class_coverage_policy

    def _cluster_selection_capacity(self, cluster_size: int) -> int:
        target_size = int(math.ceil(cluster_size * self.chunk_fraction))
        target_size = max(self.min_chunk_size, target_size)
        if self.max_chunk_size is not None:
            target_size = min(target_size, int(self.max_chunk_size))
        return min(target_size, int(cluster_size))

    def _select_from_cluster(
        self,
        cluster_idx: np.ndarray,
        scores: np.ndarray,
        *,
        target_size: Optional[int] = None,
        random_state: int | np.random.Generator | None = None,
    ) -> np.ndarray:
        if cluster_idx.size == 0:
            return cluster_idx
        if self.sample_embedding_ is None:
            raise RuntimeError("Sample embedding is not available")
        resolved_target_size = (
            self._cluster_selection_capacity(cluster_idx.size)
            if target_size is None
            else int(target_size)
        )
        return self._build_sketch_from_cluster(
            cluster_idx,
            scores,
            target_size=resolved_target_size,
            random_state=random_state,
        ).selected_indices.copy()

    def _build_sketch_from_cluster(
        self,
        cluster_idx: np.ndarray,
        scores: np.ndarray,
        *,
        target_size: int,
        random_state: int | np.random.Generator | None,
    ) -> ExactBudgetSketchPlan:
        if self.sample_embedding_ is None:
            raise RuntimeError("Sample embedding is not available")
        ridge_scores = None
        ridge_lambda = None
        if self.selection_method == "saturated_ridge_leverage":
            ridge_result = self._get_rmt_backend().compute_ridge_leverage_scores(
                self.sample_embedding_[cluster_idx],
                self.ridge_leverage_lambda,
            )
            ridge_scores = np.zeros_like(np.asarray(scores, dtype=float))
            ridge_scores[cluster_idx] = ridge_result.scores
            ridge_lambda = ridge_result.ridge_lambda
        return build_partition_sketch_plan(
            cluster_idx,
            target_size=int(target_size),
            selection_method=self.selection_method,
            scores=scores,
            embedding=self.sample_embedding_,
            random_state=random_state,
            leverage_cap_quantile=self.leverage_cap_quantile,
            leverage_mixture_alpha=self.leverage_mixture_alpha,
            leverage_uniform_floor=self.leverage_uniform_floor,
            ridge_scores=ridge_scores,
            ridge_lambda=ridge_lambda,
            training_reweighting=self.training_reweighting,
        )

    def _record_partition_sketch(
        self,
        name: str,
        plan: ExactBudgetSketchPlan,
    ) -> None:
        if self.sample_embedding_ is None:
            raise RuntimeError("Sample embedding is not available")
        self.partition_sketch_plans_[name] = plan
        self.partition_training_weights_[name] = plan.training_weights.copy()
        self.partition_inclusion_probabilities_[name] = (
            plan.selected_probabilities.copy()
        )
        rank = min(
            int(self.sample_embedding_.shape[1]),
            max(1, int(plan.selected_indices.size)),
        )
        self.partition_subspace_preservation_[name] = (
            evaluate_subspace_preservation(
                self.sample_embedding_[plan.candidate_indices],
                plan,
                rank=rank,
            )
        )

    def _select_class_aware_from_cluster(
        self,
        cluster_idx: np.ndarray,
        scores: np.ndarray,
        target: np.ndarray,
        *,
        target_size: int,
        random_state: int | np.random.Generator | None,
    ) -> ClassCoverageSelectionPlan:
        if self.sample_embedding_ is None:
            raise RuntimeError("Sample embedding is not available")
        ridge_scores = None
        ridge_lambda = None
        if self.selection_method == "saturated_ridge_leverage":
            ridge_result = self._get_rmt_backend().compute_ridge_leverage_scores(
                self.sample_embedding_[cluster_idx],
                self.ridge_leverage_lambda,
            )
            ridge_scores = np.zeros_like(np.asarray(scores, dtype=float))
            ridge_scores[cluster_idx] = ridge_result.scores
            ridge_lambda = ridge_result.ridge_lambda
        return select_class_aware_partition_indices(
            cluster_idx,
            target=target,
            target_size=target_size,
            min_samples_per_class=self.min_samples_per_class,
            class_allocation_policy=self.class_allocation_policy,
            selection_method=self.selection_method,
            scores=scores,
            embedding=self.sample_embedding_,
            random_state=random_state,
            leverage_cap_quantile=self.leverage_cap_quantile,
            leverage_mixture_alpha=self.leverage_mixture_alpha,
            leverage_uniform_floor=self.leverage_uniform_floor,
            ridge_scores=ridge_scores,
            ridge_lambda=ridge_lambda,
            training_reweighting=self.training_reweighting,
        )

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
            "spectral_component_split": (
                self.spectral_component_split_.to_dict(include_arrays=True)
            ),
            "row_spectral_participation": (
                self.row_spectral_participation_.to_dict()
                if self.row_spectral_participation_ is not None
                else None
            ),
            "expert_topology": self.expert_topology,
            "bulk_spike_topology": (
                self.bulk_spike_topology_.to_dict()
                if self.bulk_spike_topology_ is not None
                else None
            ),
            "spike_partition_geometry": (
                self.spike_partition_geometry_.to_dict()
                if self.spike_partition_geometry_ is not None
                else None
            ),
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
            "row_selection_method": self.selection_method,
            "leverage_cap_quantile": float(self.leverage_cap_quantile),
            "leverage_mixture_alpha": float(self.leverage_mixture_alpha),
            "leverage_uniform_floor": float(self.leverage_uniform_floor),
            "ridge_leverage_lambda": self.ridge_leverage_lambda,
            "training_reweighting": self.training_reweighting,
            "partition_sketch_plans": {
                name: plan.to_dict()
                for name, plan in self.partition_sketch_plans_.items()
            },
            "partition_subspace_preservation": {
                name: diagnostic.to_dict()
                for name, diagnostic in self.partition_subspace_preservation_.items()
            },
            "routing_geometry": (
                self.partition_geometry_.to_dict()
                if self.partition_geometry_ is not None
                else None
            ),
            "partition_membership_fingerprint": (
                partition_membership_fingerprint(self.cluster_labels_)
                if self.cluster_labels_ is not None
                else None
            ),
            "n_partitions_requested": int(self.n_partitions),
            "selected_n_partitions": (
                int(len(self.partitions))
                if self.bulk_spike_topology_ is not None
                else (
                    int(partition_info.selected_n_partitions)
                    if partition_info
                    else int(len(self.partitions))
                )
            ),
            "partition_selection_method": partition_info.partition_selection_method if partition_info else "fixed",
            "selected_cluster_algorithm": partition_info.selected_algorithm if partition_info else None,
            "cluster_algorithms": list(partition_info.cluster_algorithms) if partition_info else [],
            "cluster_selection_metric": partition_info.cluster_selection_metric if partition_info else None,
            "cluster_ensemble_method": partition_info.cluster_ensemble_method if partition_info else None,
            "cluster_target_type": (
                partition_info.cluster_target_type
                if partition_info
                else self.cluster_target_type
            ),
            "resolved_cluster_target_type": (
                partition_info.resolved_cluster_target_type
                if partition_info
                else "none"
            ),
            "missing_class_penalty_weight": float(
                self.missing_class_penalty_weight
            ),
            "single_class_penalty_weight": float(
                self.single_class_penalty_weight
            ),
            "class_distribution_drift_weight": float(
                self.class_distribution_drift_weight
            ),
            "class_coverage_policy": self.class_coverage_policy,
            "resolved_class_coverage_policy": (
                self.resolved_class_coverage_policy_
            ),
            "min_samples_per_class": int(self.min_samples_per_class),
            "class_allocation_policy": self.class_allocation_policy,
            "class_coverage_guaranteed": bool(
                self.class_coverage_guaranteed_
            ),
            "class_coverage_by_partition": {
                name: plan.to_dict()
                for name, plan in self.class_coverage_selection_plans_.items()
            },
            "validation_proxy_fraction": float(
                self.validation_proxy_fraction
            ),
            "validation_proxy_min_partition_rows": int(
                self.validation_proxy_min_partition_rows
            ),
            "validation_proxy_smoothing": float(
                self.validation_proxy_smoothing
            ),
            "sampling_budget_ratio": float(self.sampling_budget_ratio),
            "budget_feasibility_mode": self.budget_feasibility_mode,
            "min_sampled_rows_per_partition": int(
                self.min_sampled_rows_per_partition
            ),
            "budget_max_imbalance_ratio": self.budget_max_imbalance_ratio,
            "budget_min_partition_fraction": float(
                self.budget_min_partition_fraction
            ),
            "partition_budget_plan": (
                self.partition_budget_plan_.to_dict()
                if self.partition_budget_plan_ is not None
                else {}
            ),
            "pre_budget_chunk_sizes": (
                self.partition_budget_plan_.source_size_map
                if self.partition_budget_plan_ is not None
                else {}
            ),
            "post_budget_chunk_sizes": (
                self.partition_budget_plan_.allocation_map
                if self.partition_budget_plan_ is not None
                else {}
            ),
            "partition_selection_candidates": list(partition_info.candidates) if partition_info else [],
            "partition_selection_scores": {
                str(candidate): score
                for candidate, score in partition_info.scores
            } if partition_info else {},
            "partition_selection_selected_candidate": (
                dict(partition_info.selected_candidate)
                if partition_info
                else {}
            ),
            "partition_selection_candidate_details": list(partition_info.candidate_details) if partition_info else [],
            "partition_selection_candidate_plan": dict(partition_info.candidate_plan) if partition_info else {},
            "partition_selection_candidate_failures": list(partition_info.candidate_failures) if partition_info else [],
            "partition_selection_consensus": dict(partition_info.consensus) if partition_info else {},
            "partition_selection_validation_proxy_plan": (
                dict(partition_info.validation_proxy_plan)
                if partition_info
                else {}
            ),
            "max_cluster_imbalance_ratio": float(self.max_cluster_imbalance_ratio),
            "min_cluster_fraction": float(self.min_cluster_fraction),
            "min_auto_partition_size": int(partition_info.min_auto_partition_size) if partition_info else None,
            "partition_selection_sample_size": int(partition_info.selection_sample_size) if partition_info else None,
            "n_partitions": int(len(self.partitions)),
            "chunk_sizes": {name: int(len(idx)) for name, idx in self.partitions.items()},
        }
