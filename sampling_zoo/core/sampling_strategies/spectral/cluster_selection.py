"""Cluster candidate generation and selection for spectral samplers."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.utils.multiclass import type_of_target

from .cluster_consensus import build_weighted_membership_embedding
from .cluster_selection_contracts import (
    ClusterCandidateFitFailure,
    ClusterCandidateKind,
    ClusterCandidatePlan,
    ClusterCandidateRequest,
    ClusterConsensusPlan,
    ClassificationPartitionComponents,
    ClusterScoreComponents,
    ClusterSelectionUnavailableError,
    build_cluster_candidate_plan,
    build_cluster_consensus_plan,
    candidate_fit_failure,
    evaluate_cluster_score_components,
    evaluate_classification_partition_components,
    normalize_cluster_algorithm,
    score_cluster_components,
)
from .partition_validation import (
    PartitionDownstreamProxyEvaluator,
    PartitionValidationProxyEvaluator,
)
from ...experiment.budgeting import build_partition_budget_plan
from ...utils.progress import progress_bar, progress_write

try:  # scikit-learn >= 1.1
    from sklearn.cluster import BisectingKMeans
except Exception:  # pragma: no cover - optional by sklearn version
    BisectingKMeans = None

try:  # scikit-learn >= 1.3
    from sklearn.cluster import HDBSCAN as SklearnHDBSCAN
except Exception:  # pragma: no cover - optional by sklearn version
    SklearnHDBSCAN = None

try:  # external optional package
    import hdbscan as hdbscan_package
except Exception:  # pragma: no cover - optional dependency
    hdbscan_package = None


@dataclass(frozen=True)
class ClusterCandidate:
    algorithm: str
    n_clusters: int
    labels: np.ndarray
    centers: np.ndarray
    estimator: Any
    score: float
    valid: bool
    components: Dict[str, Any]


@dataclass(frozen=True)
class _ClusterCandidateBatch:
    plan: ClusterCandidatePlan
    candidates: Tuple[ClusterCandidate, ...]
    failures: Tuple[ClusterCandidateFitFailure, ...]


@dataclass(frozen=True)
class _ClusterScoringContext:
    target: Optional[Any]
    resolved_target_type: str
    validation_proxy: Optional[PartitionValidationProxyEvaluator] = None
    downstream_proxy: Optional[PartitionDownstreamProxyEvaluator] = None


@dataclass(frozen=True)
class _ClusterConsensusRuntime:
    plan: ClusterConsensusPlan
    candidates: Tuple[ClusterCandidate, ...]
    representation_shape: Tuple[int, int]
    representation_nnz: int
    fallback_to_source_candidate: bool
    best_source_score: float
    score_delta_vs_best_source: float


@dataclass(frozen=True)
class _ClusterSelectionDecision:
    selected: ClusterCandidate
    consensus: Optional[_ClusterConsensusRuntime] = None


@dataclass(frozen=True)
class ClusterSelectionResult:
    labels: np.ndarray
    centers: np.ndarray
    estimator: Any
    selected_algorithm: str
    selected_n_clusters: int
    candidates: Tuple[ClusterCandidate, ...]
    candidate_plan: ClusterCandidatePlan
    candidate_failures: Tuple[ClusterCandidateFitFailure, ...]
    diagnostics: Dict[str, Any]
    consensus_plan: Optional[ClusterConsensusPlan] = None
    consensus_candidates: Tuple[ClusterCandidate, ...] = ()


class _ClusterAdapterUnavailableError(RuntimeError):
    pass


class SpectralClusterSelector:
    """Generate cluster candidates and choose the best partitioning objective."""

    def __init__(
        self,
        *,
        algorithms: Sequence[str] = ("kmeans",),
        selection_metric: str = "balanced_silhouette",
        ensemble_method: str = "best_score",
        min_partitions: int = 2,
        max_partitions: Optional[int] = None,
        min_auto_partition_size: int = 256,
        selection_sample_size: int = 5000,
        max_cluster_imbalance_ratio: float = 5.0,
        min_cluster_fraction: float = 0.05,
        imbalance_penalty_weight: float = 0.15,
        tiny_cluster_penalty_weight: float = 0.30,
        target_contrast_weight: float = 0.0,
        target_type: str = "auto",
        missing_class_penalty_weight: float = 0.25,
        single_class_penalty_weight: float = 0.50,
        class_distribution_drift_weight: float = 0.25,
        validation_proxy_fraction: float = 0.2,
        validation_proxy_min_partition_rows: int = 8,
        validation_proxy_smoothing: float = 1.0,
        sampling_budget_ratio: float = 1.0,
        budget_feasibility_mode: str = "off",
        min_sampled_rows_per_partition: int = 1,
        budget_max_imbalance_ratio: Optional[float] = None,
        budget_min_partition_fraction: float = 0.0,
        include_single_partition_candidate: bool = False,
        downstream_proxy_model_factory: Optional[Callable[[], Any]] = None,
        downstream_proxy_shortlist_size: int = 3,
        downstream_complexity_penalty_weight: float = 0.01,
        selection_method: str = "hybrid",
        leverage_cap_quantile: float = 0.95,
        routing_temperature: float = 1.0,
        routing_shrinkage: float = 0.05,
        vote_temperature: float = 0.05,
        random_state: Optional[int] = 42,
        show_progress: bool = True,
    ) -> None:
        self.algorithms = tuple(self._normalize_algorithm(name) for name in algorithms)
        self.selection_metric = self._validate_choice(
            "cluster_selection_metric",
            selection_metric,
            (
                "silhouette",
                "balanced_silhouette",
                "validation_proxy",
                "budget_aware_validation_proxy",
                "downstream_proxy",
            ),
        )
        self.ensemble_method = self._validate_choice(
            "cluster_ensemble_method",
            ensemble_method,
            ("best_score", "weighted_vote", "coassociation"),
        )
        self.min_partitions = max(1, int(min_partitions))
        self.max_partitions = (
            None if max_partitions is None else max(1, int(max_partitions))
        )
        self.min_auto_partition_size = max(1, int(min_auto_partition_size))
        self.selection_sample_size = max(1, int(selection_sample_size))
        self.max_cluster_imbalance_ratio = float(max_cluster_imbalance_ratio)
        self.min_cluster_fraction = float(min_cluster_fraction)
        self.imbalance_penalty_weight = float(imbalance_penalty_weight)
        self.tiny_cluster_penalty_weight = float(tiny_cluster_penalty_weight)
        self.target_contrast_weight = float(target_contrast_weight)
        self.target_type = self._validate_choice(
            "cluster_target_type",
            target_type,
            ("auto", "regression", "classification"),
        )
        self.missing_class_penalty_weight = self._validate_nonnegative_float(
            "missing_class_penalty_weight",
            missing_class_penalty_weight,
        )
        self.single_class_penalty_weight = self._validate_nonnegative_float(
            "single_class_penalty_weight",
            single_class_penalty_weight,
        )
        self.class_distribution_drift_weight = self._validate_nonnegative_float(
            "class_distribution_drift_weight",
            class_distribution_drift_weight,
        )
        self.validation_proxy_fraction = self._validate_fraction(
            "validation_proxy_fraction",
            validation_proxy_fraction,
        )
        self.validation_proxy_min_partition_rows = max(
            1,
            int(validation_proxy_min_partition_rows),
        )
        self.validation_proxy_smoothing = self._validate_positive_float(
            "validation_proxy_smoothing",
            validation_proxy_smoothing,
        )
        self.sampling_budget_ratio = self._validate_unit_fraction(
            "sampling_budget_ratio",
            sampling_budget_ratio,
        )
        self.budget_feasibility_mode = self._validate_choice(
            "budget_feasibility_mode",
            budget_feasibility_mode,
            ("off", "hard"),
        )
        self.min_sampled_rows_per_partition = max(
            1,
            int(min_sampled_rows_per_partition),
        )
        self.budget_max_imbalance_ratio = (
            None
            if budget_max_imbalance_ratio is None
            else self._validate_positive_float(
                "budget_max_imbalance_ratio",
                budget_max_imbalance_ratio,
            )
        )
        self.budget_min_partition_fraction = self._validate_zero_one_fraction(
            "budget_min_partition_fraction",
            budget_min_partition_fraction,
        )
        self.include_single_partition_candidate = bool(
            include_single_partition_candidate
        )
        self.downstream_proxy_model_factory = downstream_proxy_model_factory
        self.downstream_proxy_shortlist_size = max(
            1,
            int(downstream_proxy_shortlist_size),
        )
        self.downstream_complexity_penalty_weight = (
            self._validate_nonnegative_float(
                "downstream_complexity_penalty_weight",
                downstream_complexity_penalty_weight,
            )
        )
        self.selection_method = self._validate_choice(
            "selection_method",
            selection_method,
            (
                "all",
                "uniform",
                "leverage",
                "capped_leverage",
                "saturated_leverage",
                "robust_leverage_mixture",
                "saturated_ridge_leverage",
                "maxvol",
                "hybrid",
            ),
        )
        self.leverage_cap_quantile = self._validate_unit_fraction(
            "leverage_cap_quantile",
            leverage_cap_quantile,
        )
        self.routing_temperature = self._validate_positive_float(
            "routing_temperature",
            routing_temperature,
        )
        self.routing_shrinkage = self._validate_zero_one_fraction(
            "routing_shrinkage",
            routing_shrinkage,
        )
        if (
            self.selection_metric == "downstream_proxy"
            and self.ensemble_method != "best_score"
        ):
            raise ValueError(
                "downstream_proxy requires cluster_ensemble_method='best_score'"
            )
        self.vote_temperature = max(float(vote_temperature), 1e-6)
        self.random_state = random_state
        self.show_progress = bool(show_progress)

    def select(
        self,
        embedding: np.ndarray,
        target: Optional[Any] = None,
        *,
        downstream_features: Optional[np.ndarray] = None,
        sample_scores: Optional[np.ndarray] = None,
    ) -> ClusterSelectionResult:
        embedding = np.asarray(embedding, dtype=float)
        if embedding.ndim != 2:
            raise ValueError("embedding must be a 2D matrix")
        scoring_context = self._build_scoring_context(
            embedding,
            target,
            downstream_features=downstream_features,
            sample_scores=sample_scores,
        )
        batch = self._build_candidates(embedding, scoring_context)
        if not batch.candidates:
            raise ClusterSelectionUnavailableError(batch.plan, batch.failures)

        selection_candidates = batch.candidates
        if self.selection_metric == "downstream_proxy":
            batch, selection_candidates = self._rerank_downstream_candidates(
                batch,
                scoring_context,
            )
        decision = self._select_candidate(
            selection_candidates,
            embedding,
            scoring_context,
        )
        selected = decision.selected
        progress_write(
            (
                "Selected spectral clusters: "
                f"algorithm={selected.algorithm}, "
                f"k={selected.n_clusters}, "
                f"score={selected.score:.4f}, "
                f"valid={selected.valid}"
            ),
            enabled=self.show_progress,
        )
        return ClusterSelectionResult(
            labels=selected.labels,
            centers=selected.centers,
            estimator=selected.estimator,
            selected_algorithm=selected.algorithm,
            selected_n_clusters=selected.n_clusters,
            candidates=batch.candidates,
            candidate_plan=batch.plan,
            candidate_failures=batch.failures,
            consensus_plan=(decision.consensus.plan if decision.consensus else None),
            consensus_candidates=(
                decision.consensus.candidates if decision.consensus else ()
            ),
            diagnostics=self._build_diagnostics(
                batch,
                decision,
                scoring_context,
            ),
        )

    def build_candidate_plan(self, n_samples: int) -> ClusterCandidatePlan:
        return build_cluster_candidate_plan(
            n_samples=n_samples,
            algorithms=self.algorithms,
            min_partitions=self.min_partitions,
            max_partitions=self.max_partitions,
            min_auto_partition_size=self.min_auto_partition_size,
            include_single_partition_candidate=(
                self.include_single_partition_candidate
            ),
        )

    def _build_candidates(
        self,
        embedding: np.ndarray,
        scoring_context: _ClusterScoringContext,
    ) -> _ClusterCandidateBatch:
        plan = self.build_candidate_plan(embedding.shape[0])
        candidates: List[ClusterCandidate] = []
        failures: List[ClusterCandidateFitFailure] = []
        with progress_bar(
            enabled=self.show_progress,
            desc="Spectral cluster candidates",
            total=max(plan.total_fit_count, 1),
        ) as bar:
            for request in plan.requests:
                self._set_progress_postfix(
                    bar,
                    algorithm=request.algorithm,
                    n_clusters=request.n_clusters,
                )
                outcome = self._fit_candidate_request(
                    request,
                    embedding,
                    scoring_context,
                )
                if isinstance(outcome, ClusterCandidateFitFailure):
                    failures.append(outcome)
                else:
                    candidates.append(outcome)
                bar.update(1)
        return _ClusterCandidateBatch(
            plan=plan,
            candidates=tuple(candidates),
            failures=tuple(failures),
        )

    def _candidate_fit_count(self, counts: Sequence[int]) -> int:
        total = 0
        for algorithm in self.algorithms:
            total += 1 if algorithm == "hdbscan" else len(counts)
        return max(total, 1)

    @staticmethod
    def _set_progress_postfix(
        bar: Any, *, algorithm: str, n_clusters: Optional[int]
    ) -> None:
        postfix = {"algorithm": algorithm}
        if n_clusters is not None:
            postfix["k"] = n_clusters
        try:
            bar.set_postfix(postfix)
        except Exception:
            return

    def _candidate_counts(self, n_samples: int) -> List[int]:
        return list(self.build_candidate_plan(n_samples).eligible_count_candidates)

    def _fit_candidate_request(
        self,
        request: ClusterCandidateRequest,
        embedding: np.ndarray,
        scoring_context: _ClusterScoringContext,
    ) -> Union[ClusterCandidate, ClusterCandidateFitFailure]:
        try:
            if request.kind is ClusterCandidateKind.DENSITY_BASED:
                return self._fit_hdbscan_candidate(embedding, scoring_context)
            if request.n_clusters is None:
                raise ValueError("Count-based request requires n_clusters")
            return self._fit_count_based_candidate(
                request.algorithm,
                embedding,
                request.n_clusters,
                scoring_context,
            )
        except _ClusterAdapterUnavailableError as exc:
            return candidate_fit_failure(request, exc, unavailable=True)
        except Exception as exc:
            return candidate_fit_failure(request, exc)

    def _fit_count_based_candidate(
        self,
        algorithm: str,
        embedding: np.ndarray,
        n_clusters: int,
        scoring_context: _ClusterScoringContext,
    ) -> ClusterCandidate:
        estimator = self._make_count_based_estimator(algorithm, n_clusters)
        labels = estimator.fit_predict(embedding)
        return self._make_candidate(
            algorithm,
            int(n_clusters),
            labels,
            embedding,
            scoring_context,
            estimator,
        )

    def _make_count_based_estimator(self, algorithm: str, n_clusters: int) -> Any:
        if algorithm == "kmeans":
            return self._make_kmeans(n_clusters)
        if algorithm == "bisecting_kmeans":
            if BisectingKMeans is None:
                raise _ClusterAdapterUnavailableError(
                    "BisectingKMeans is not available in this sklearn version"
                )
            return BisectingKMeans(
                n_clusters=n_clusters, random_state=self.random_state
            )
        if algorithm == "gmm":
            return GaussianMixture(
                n_components=n_clusters,
                covariance_type="diag",
                random_state=self.random_state,
                reg_covar=1e-6,
            )
        raise ValueError(f"Unsupported cluster algorithm: {algorithm}")

    def _fit_hdbscan_candidate(
        self,
        embedding: np.ndarray,
        scoring_context: _ClusterScoringContext,
    ) -> ClusterCandidate:
        min_cluster_size = max(
            2, int(math.ceil(self.min_cluster_fraction * embedding.shape[0]))
        )
        if SklearnHDBSCAN is not None:
            estimator = SklearnHDBSCAN(min_cluster_size=min_cluster_size)
        elif hdbscan_package is not None:
            estimator = hdbscan_package.HDBSCAN(min_cluster_size=min_cluster_size)
        else:
            raise _ClusterAdapterUnavailableError("No HDBSCAN adapter is installed")
        labels = estimator.fit_predict(embedding)
        normalized = self._normalize_labels(labels)
        n_clusters = int(np.unique(normalized).size)
        if n_clusters < 1:
            raise ValueError("HDBSCAN returned no clusters")
        return self._make_candidate(
            "hdbscan",
            n_clusters,
            normalized,
            embedding,
            scoring_context,
            estimator,
        )

    def _make_candidate(
        self,
        algorithm: str,
        n_clusters: int,
        labels: np.ndarray,
        embedding: np.ndarray,
        scoring_context: _ClusterScoringContext,
        estimator: Any,
    ) -> ClusterCandidate:
        labels = self._normalize_labels(labels)
        centers = self._centers_from_labels(embedding, labels)
        score_components = self._score_components(
            embedding,
            labels,
            scoring_context,
        )
        components = score_components.to_dict()
        score = self._candidate_score(score_components)
        return ClusterCandidate(
            algorithm=algorithm,
            n_clusters=int(np.unique(labels).size),
            labels=labels,
            centers=centers,
            estimator=estimator,
            score=score,
            valid=score_components.valid,
            components=components,
        )

    def _score_components(
        self,
        embedding: np.ndarray,
        labels: np.ndarray,
        scoring_context: _ClusterScoringContext,
    ) -> ClusterScoreComponents:
        counts = np.bincount(labels, minlength=int(labels.max()) + 1)
        silhouette = self._safe_silhouette(embedding, labels)
        target_contrast = (
            self._target_contrast(labels, scoring_context.target)
            if scoring_context.resolved_target_type == "regression"
            else 0.0
        )
        classification = (
            self._classification_components(labels, scoring_context.target)
            if scoring_context.resolved_target_type == "classification"
            else None
        )
        validation_proxy = (
            scoring_context.validation_proxy.evaluate(labels)
            if scoring_context.validation_proxy is not None
            else None
        )
        budget_plan = None
        if self.budget_feasibility_mode == "hard":
            budget_plan = build_partition_budget_plan(
                {
                    f"chunk_{cluster_id}": int(count)
                    for cluster_id, count in enumerate(counts.tolist())
                },
                total_rows=int(labels.size),
                budget_ratio=self.sampling_budget_ratio,
                min_rows_per_partition=self.min_sampled_rows_per_partition,
                max_imbalance_ratio=self.budget_max_imbalance_ratio,
                min_partition_fraction=self.budget_min_partition_fraction,
            )
        return evaluate_cluster_score_components(
            counts=counts,
            silhouette=silhouette,
            target_contrast=target_contrast,
            max_cluster_imbalance_ratio=self.max_cluster_imbalance_ratio,
            min_cluster_fraction=self.min_cluster_fraction,
            classification=classification,
            validation_proxy=validation_proxy,
            budget_plan=budget_plan,
        )

    def _candidate_score(self, components: ClusterScoreComponents) -> float:
        selection_metric = self.selection_metric
        if selection_metric == "downstream_proxy" and components.downstream_proxy is None:
            selection_metric = "balanced_silhouette"
        return score_cluster_components(
            components,
            selection_metric=selection_metric,
            imbalance_penalty_weight=self.imbalance_penalty_weight,
            tiny_cluster_penalty_weight=self.tiny_cluster_penalty_weight,
            target_contrast_weight=self.target_contrast_weight,
            missing_class_penalty_weight=self.missing_class_penalty_weight,
            single_class_penalty_weight=self.single_class_penalty_weight,
            class_distribution_drift_weight=(
                self.class_distribution_drift_weight
            ),
            downstream_complexity_penalty_weight=(
                self.downstream_complexity_penalty_weight
            ),
        )

    def _build_scoring_context(
        self,
        embedding: np.ndarray,
        target: Optional[Any],
        *,
        downstream_features: Optional[np.ndarray],
        sample_scores: Optional[np.ndarray],
    ) -> _ClusterScoringContext:
        resolved_target_type = self._resolve_target_type(target)
        validation_proxy = None
        if self.selection_metric in {
            "validation_proxy",
            "budget_aware_validation_proxy",
        }:
            if resolved_target_type not in {"regression", "classification"}:
                raise ValueError(
                    "validation_proxy cluster selection requires a supported target"
                )
            validation_proxy = PartitionValidationProxyEvaluator(
                embedding,
                target,
                target_type=resolved_target_type,
                validation_fraction=self.validation_proxy_fraction,
                min_partition_train_rows=(
                    self.validation_proxy_min_partition_rows
                ),
                classification_smoothing=self.validation_proxy_smoothing,
                random_state=self.random_state,
            )
        downstream_proxy = None
        if self.selection_metric == "downstream_proxy":
            if resolved_target_type != "regression":
                raise ValueError(
                    "downstream_proxy cluster selection requires a regression target"
                )
            if downstream_features is None or sample_scores is None:
                raise ValueError(
                    "downstream_proxy requires downstream_features and sample_scores"
                )
            downstream_proxy = PartitionDownstreamProxyEvaluator(
                embedding,
                downstream_features,
                target,
                sample_scores,
                target_type=resolved_target_type,
                model_factory=self.downstream_proxy_model_factory,
                budget_ratio=self.sampling_budget_ratio,
                min_partition_rows=self.min_sampled_rows_per_partition,
                max_imbalance_ratio=self.budget_max_imbalance_ratio,
                min_partition_fraction=self.budget_min_partition_fraction,
                selection_method=self.selection_method,
                leverage_cap_quantile=self.leverage_cap_quantile,
                routing_temperature=self.routing_temperature,
                routing_shrinkage=self.routing_shrinkage,
                validation_fraction=self.validation_proxy_fraction,
                random_state=self.random_state,
            )
        return _ClusterScoringContext(
            target=target,
            resolved_target_type=resolved_target_type,
            validation_proxy=validation_proxy,
            downstream_proxy=downstream_proxy,
        )

    def _rerank_downstream_candidates(
        self,
        batch: _ClusterCandidateBatch,
        scoring_context: _ClusterScoringContext,
    ) -> tuple[_ClusterCandidateBatch, Tuple[ClusterCandidate, ...]]:
        evaluator = scoring_context.downstream_proxy
        if evaluator is None:
            raise RuntimeError("Downstream proxy evaluator is not initialized")
        preliminary = tuple(
            sorted(
                (candidate for candidate in batch.candidates if candidate.valid),
                key=lambda candidate: (
                    -candidate.score,
                    candidate.n_clusters,
                    candidate.algorithm,
                ),
            )[: self.downstream_proxy_shortlist_size]
        )
        if not preliminary:
            raise ValueError(
                "No budget-feasible partition candidate is available for downstream reranking"
            )

        reranked_by_identity: Dict[int, ClusterCandidate] = {}
        reranked: List[ClusterCandidate] = []
        for candidate in preliminary:
            components = evaluator.evaluate(candidate.labels)
            payload = dict(candidate.components)
            payload["preliminary_score"] = float(candidate.score)
            payload["downstream_proxy"] = components.to_dict()
            score = float(
                components.relative_gain_vs_global
                - self.downstream_complexity_penalty_weight
                * max(candidate.n_clusters - 1, 0)
            )
            updated = replace(
                candidate,
                score=score,
                valid=(candidate.valid and components.status == "ok"),
                components=payload,
            )
            reranked.append(updated)
            reranked_by_identity[id(candidate)] = updated

        all_candidates = tuple(
            reranked_by_identity.get(
                id(candidate),
                replace(
                    candidate,
                    score=-math.inf,
                    valid=False,
                    components={
                        **candidate.components,
                        "preliminary_score": float(candidate.score),
                        "downstream_proxy": {"status": "not_shortlisted"},
                    },
                ),
            )
            for candidate in batch.candidates
        )
        selected_pool = tuple(candidate for candidate in reranked if candidate.valid)
        if not selected_pool:
            raise ValueError("Every downstream partition candidate evaluation failed")
        return replace(batch, candidates=all_candidates), selected_pool

    def _resolve_target_type(self, target: Optional[Any]) -> str:
        if target is None:
            return "none"
        if self.target_type != "auto":
            return self.target_type
        try:
            inferred = type_of_target(np.asarray(target))
        except (TypeError, ValueError):
            return "none"
        if inferred in {"binary", "multiclass"}:
            return "classification"
        if inferred in {"continuous"}:
            return "regression"
        return "none"

    @staticmethod
    def _classification_components(
        labels: np.ndarray,
        target: Optional[Any],
    ) -> Optional[ClassificationPartitionComponents]:
        if target is None:
            return None
        y = np.asarray(target)
        if y.ndim != 1 or y.size != labels.size:
            return None
        try:
            classes, encoded = np.unique(y, return_inverse=True)
        except (TypeError, ValueError):
            return None
        cluster_ids = np.unique(labels)
        class_counts = np.zeros((cluster_ids.size, classes.size), dtype=int)
        cluster_positions = {
            int(cluster_id): position
            for position, cluster_id in enumerate(cluster_ids.tolist())
        }
        rows = np.asarray(
            [cluster_positions[int(cluster_id)] for cluster_id in labels],
            dtype=int,
        )
        np.add.at(class_counts, (rows, encoded), 1)
        return evaluate_classification_partition_components(
            global_class_counts=np.bincount(encoded, minlength=classes.size),
            cluster_class_counts=class_counts,
            class_labels=tuple(str(value) for value in classes.tolist()),
        )

    def _safe_silhouette(
        self, embedding: np.ndarray, labels: np.ndarray
    ) -> Optional[float]:
        unique_labels = np.unique(labels)
        if unique_labels.size < 2 or unique_labels.size >= embedding.shape[0]:
            return None
        sample_size = min(self.selection_sample_size, embedding.shape[0])
        try:
            return float(
                silhouette_score(
                    embedding,
                    labels,
                    sample_size=sample_size
                    if sample_size < embedding.shape[0]
                    else None,
                    random_state=self.random_state,
                )
            )
        except Exception:
            return None

    @staticmethod
    def _target_contrast(labels: np.ndarray, target: Optional[Any]) -> float:
        if target is None:
            return 0.0
        y = np.asarray(target, dtype=float)
        if y.ndim != 1 or y.size != labels.size:
            return 0.0
        global_std = float(np.nanstd(y))
        if not np.isfinite(global_std) or global_std <= 1e-12:
            return 0.0
        cluster_means = []
        weights = []
        for cluster_id in np.unique(labels):
            values = y[labels == cluster_id]
            if values.size == 0:
                continue
            cluster_means.append(float(np.nanmean(values)))
            weights.append(float(values.size / y.size))
        if len(cluster_means) < 2:
            return 0.0
        means = np.asarray(cluster_means, dtype=float)
        weights_arr = np.asarray(weights, dtype=float)
        weighted_mean = float(np.sum(means * weights_arr))
        between = float(np.sqrt(np.sum(weights_arr * (means - weighted_mean) ** 2)))
        return between / global_std

    def _select_candidate(
        self,
        candidates: Sequence[ClusterCandidate],
        embedding: np.ndarray,
        scoring_context: _ClusterScoringContext,
    ) -> _ClusterSelectionDecision:
        valid_candidates = [candidate for candidate in candidates if candidate.valid]
        pool = valid_candidates or list(candidates)
        if self.ensemble_method == "weighted_vote":
            return _ClusterSelectionDecision(
                selected=self._select_by_weighted_vote(pool)
            )
        if self.ensemble_method == "coassociation":
            return self._select_by_coassociation(
                pool,
                embedding,
                scoring_context,
            )
        return _ClusterSelectionDecision(
            selected=max(pool, key=lambda candidate: candidate.score)
        )

    def _select_by_weighted_vote(
        self, candidates: Sequence[ClusterCandidate]
    ) -> ClusterCandidate:
        scores = np.asarray([candidate.score for candidate in candidates], dtype=float)
        scores = np.where(np.isfinite(scores), scores, -1e9)
        weights = np.exp((scores - scores.max()) / self.vote_temperature)
        votes: Dict[int, float] = {}
        for candidate, weight in zip(candidates, weights):
            votes[candidate.n_clusters] = votes.get(candidate.n_clusters, 0.0) + float(
                weight
            )
        selected_k = max(votes.items(), key=lambda item: (item[1], item[0]))[0]
        same_k = [
            candidate for candidate in candidates if candidate.n_clusters == selected_k
        ]
        return max(same_k, key=lambda candidate: candidate.score)

    def _select_by_coassociation(
        self,
        candidates: Sequence[ClusterCandidate],
        embedding: np.ndarray,
        scoring_context: _ClusterScoringContext,
    ) -> _ClusterSelectionDecision:
        ordered = tuple(
            sorted(
                candidates,
                key=lambda candidate: (
                    candidate.algorithm,
                    candidate.n_clusters,
                    -candidate.score,
                ),
            )
        )
        plan = build_cluster_consensus_plan(
            algorithms=[candidate.algorithm for candidate in ordered],
            n_clusters=[candidate.n_clusters for candidate in ordered],
            scores=[candidate.score for candidate in ordered],
            vote_temperature=self.vote_temperature,
        )
        membership = build_weighted_membership_embedding(
            [candidate.labels for candidate in ordered],
            plan,
        )
        estimator = self._make_kmeans(plan.selected_n_clusters)
        labels = estimator.fit_predict(membership)
        consensus_candidate = self._make_candidate(
            "coassociation_consensus",
            plan.selected_n_clusters,
            labels,
            embedding,
            scoring_context,
            estimator,
        )
        same_k_sources = tuple(
            candidate
            for candidate in ordered
            if candidate.n_clusters == plan.selected_n_clusters
        )
        best_source = max(same_k_sources, key=lambda candidate: candidate.score)
        selected = consensus_candidate
        if not consensus_candidate.valid and same_k_sources:
            valid_sources = tuple(
                candidate for candidate in same_k_sources if candidate.valid
            )
            fallback_pool = valid_sources or same_k_sources
            fallback = max(fallback_pool, key=lambda candidate: candidate.score)
            if fallback.valid or fallback.score > consensus_candidate.score:
                selected = fallback
        runtime = _ClusterConsensusRuntime(
            plan=plan,
            candidates=(consensus_candidate,),
            representation_shape=tuple(map(int, membership.shape)),
            representation_nnz=int(membership.nnz),
            fallback_to_source_candidate=(selected is not consensus_candidate),
            best_source_score=float(best_source.score),
            score_delta_vs_best_source=float(
                consensus_candidate.score - best_source.score
            ),
        )
        return _ClusterSelectionDecision(selected=selected, consensus=runtime)

    def _build_diagnostics(
        self,
        batch: _ClusterCandidateBatch,
        decision: _ClusterSelectionDecision,
        scoring_context: _ClusterScoringContext,
    ) -> Dict[str, Any]:
        selected = decision.selected
        consensus = decision.consensus
        return {
            "selected_algorithm": selected.algorithm,
            "selected_n_clusters": int(selected.n_clusters),
            "cluster_algorithms": list(self.algorithms),
            "cluster_selection_metric": self.selection_metric,
            "cluster_ensemble_method": self.ensemble_method,
            "max_cluster_imbalance_ratio": float(self.max_cluster_imbalance_ratio),
            "min_cluster_fraction": float(self.min_cluster_fraction),
            "imbalance_penalty_weight": float(self.imbalance_penalty_weight),
            "tiny_cluster_penalty_weight": float(self.tiny_cluster_penalty_weight),
            "target_contrast_weight": float(self.target_contrast_weight),
            "hard_constraint_penalty": 1.0,
            "cluster_target_type": self.target_type,
            "resolved_cluster_target_type": (
                scoring_context.resolved_target_type
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
            "validation_proxy_plan": (
                scoring_context.validation_proxy.plan.to_dict()
                if scoring_context.validation_proxy is not None
                else None
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
            "include_single_partition_candidate": bool(
                self.include_single_partition_candidate
            ),
            "downstream_proxy_shortlist_size": int(
                self.downstream_proxy_shortlist_size
            ),
            "downstream_complexity_penalty_weight": float(
                self.downstream_complexity_penalty_weight
            ),
            "candidate_plan": batch.plan.to_dict(),
            "candidate_failures": [failure.to_dict() for failure in batch.failures],
            "selected_candidate": self._candidate_diagnostic(selected),
            "candidates": [self._candidate_diagnostic(candidate) for candidate in batch.candidates],
            "consensus": (
                {
                    "plan": consensus.plan.to_dict(),
                    "representation_shape": list(consensus.representation_shape),
                    "representation_nnz": int(consensus.representation_nnz),
                    "fallback_to_source_candidate": bool(
                        consensus.fallback_to_source_candidate
                    ),
                    "best_source_score": float(consensus.best_source_score),
                    "score_delta_vs_best_source": float(
                        consensus.score_delta_vs_best_source
                    ),
                    "candidates": [
                        self._candidate_diagnostic(candidate)
                        for candidate in consensus.candidates
                    ],
                }
                if consensus is not None
                else None
            ),
        }

    @staticmethod
    def _candidate_diagnostic(candidate: ClusterCandidate) -> Dict[str, Any]:
        return {
            "algorithm": candidate.algorithm,
            "n_clusters": int(candidate.n_clusters),
            "score": float(candidate.score),
            "valid": bool(candidate.valid),
            "components": candidate.components,
        }

    def _make_kmeans(self, n_clusters: int) -> KMeans:
        try:
            return KMeans(
                n_clusters=n_clusters, random_state=self.random_state, n_init="auto"
            )
        except TypeError:
            return KMeans(
                n_clusters=n_clusters, random_state=self.random_state, n_init=10
            )

    @staticmethod
    def _normalize_algorithm(name: str) -> str:
        return normalize_cluster_algorithm(name)

    @staticmethod
    def _validate_choice(name: str, value: str, choices: Sequence[str]) -> str:
        if value not in choices:
            allowed = ", ".join(choices)
            raise ValueError(f"{name} must be one of: {allowed}")
        return value

    @staticmethod
    def _validate_nonnegative_float(name: str, value: float) -> float:
        normalized = float(value)
        if not np.isfinite(normalized) or normalized < 0:
            raise ValueError(f"{name} must be a non-negative finite value")
        return normalized

    @staticmethod
    def _validate_positive_float(name: str, value: float) -> float:
        normalized = float(value)
        if not np.isfinite(normalized) or normalized <= 0:
            raise ValueError(f"{name} must be a positive finite value")
        return normalized

    @staticmethod
    def _validate_fraction(name: str, value: float) -> float:
        normalized = float(value)
        if not np.isfinite(normalized) or not 0 < normalized < 0.5:
            raise ValueError(f"{name} must be in (0, 0.5)")
        return normalized

    @staticmethod
    def _validate_unit_fraction(name: str, value: float) -> float:
        normalized = float(value)
        if not np.isfinite(normalized) or not 0 < normalized <= 1:
            raise ValueError(f"{name} must be in (0, 1]")
        return normalized

    @staticmethod
    def _validate_zero_one_fraction(name: str, value: float) -> float:
        normalized = float(value)
        if not np.isfinite(normalized) or not 0 <= normalized <= 1:
            raise ValueError(f"{name} must be in [0, 1]")
        return normalized

    @staticmethod
    def _normalize_labels(labels: np.ndarray) -> np.ndarray:
        labels = np.asarray(labels)
        unique = sorted(np.unique(labels).tolist())
        remap = {old: new for new, old in enumerate(unique)}
        return np.asarray([remap[label] for label in labels], dtype=int)

    @staticmethod
    def _centers_from_labels(embedding: np.ndarray, labels: np.ndarray) -> np.ndarray:
        centers = []
        for cluster_id in sorted(np.unique(labels).tolist()):
            centers.append(np.mean(embedding[labels == cluster_id], axis=0))
        return np.asarray(centers, dtype=float)
