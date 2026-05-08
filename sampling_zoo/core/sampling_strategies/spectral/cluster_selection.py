"""Cluster candidate generation and selection for spectral samplers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture

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
class ClusterSelectionResult:
    labels: np.ndarray
    centers: np.ndarray
    estimator: Any
    selected_algorithm: str
    selected_n_clusters: int
    candidates: Tuple[ClusterCandidate, ...]
    diagnostics: Dict[str, Any]


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
        vote_temperature: float = 0.05,
        random_state: Optional[int] = 42,
        show_progress: bool = True,
    ) -> None:
        self.algorithms = tuple(self._normalize_algorithm(name) for name in algorithms)
        self.selection_metric = self._validate_choice(
            "cluster_selection_metric",
            selection_metric,
            ("silhouette", "balanced_silhouette"),
        )
        self.ensemble_method = self._validate_choice(
            "cluster_ensemble_method",
            ensemble_method,
            ("best_score", "weighted_vote"),
        )
        self.min_partitions = max(1, int(min_partitions))
        self.max_partitions = None if max_partitions is None else max(1, int(max_partitions))
        self.min_auto_partition_size = max(1, int(min_auto_partition_size))
        self.selection_sample_size = max(1, int(selection_sample_size))
        self.max_cluster_imbalance_ratio = float(max_cluster_imbalance_ratio)
        self.min_cluster_fraction = float(min_cluster_fraction)
        self.imbalance_penalty_weight = float(imbalance_penalty_weight)
        self.tiny_cluster_penalty_weight = float(tiny_cluster_penalty_weight)
        self.target_contrast_weight = float(target_contrast_weight)
        self.vote_temperature = max(float(vote_temperature), 1e-6)
        self.random_state = random_state
        self.show_progress = bool(show_progress)

    def select(
        self,
        embedding: np.ndarray,
        target: Optional[Any] = None,
    ) -> ClusterSelectionResult:
        embedding = np.asarray(embedding, dtype=float)
        if embedding.ndim != 2:
            raise ValueError("embedding must be a 2D matrix")
        candidates = self._build_candidates(embedding, target)
        if not candidates:
            raise RuntimeError("No cluster candidates were generated")

        selected = self._select_candidate(candidates)
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
            candidates=tuple(candidates),
            diagnostics=self._build_diagnostics(candidates, selected),
        )

    def _build_candidates(self, embedding: np.ndarray, target: Optional[Any]) -> List[ClusterCandidate]:
        candidates: List[ClusterCandidate] = []
        counts = self._candidate_counts(embedding.shape[0])
        total = self._candidate_fit_count(counts)
        with progress_bar(
            enabled=self.show_progress,
            desc="Spectral cluster candidates",
            total=total,
        ) as bar:
            for algorithm in self.algorithms:
                if algorithm == "hdbscan":
                    self._set_progress_postfix(bar, algorithm=algorithm, n_clusters=None)
                    candidate = self._fit_hdbscan_candidate(embedding, target)
                    if candidate is not None:
                        candidates.append(candidate)
                    bar.update(1)
                    continue
                for n_clusters in counts:
                    self._set_progress_postfix(bar, algorithm=algorithm, n_clusters=n_clusters)
                    candidate = self._fit_count_based_candidate(algorithm, embedding, n_clusters, target)
                    if candidate is not None:
                        candidates.append(candidate)
                    bar.update(1)
        return candidates

    def _candidate_fit_count(self, counts: Sequence[int]) -> int:
        total = 0
        for algorithm in self.algorithms:
            total += 1 if algorithm == "hdbscan" else len(counts)
        return max(total, 1)

    @staticmethod
    def _set_progress_postfix(bar: Any, *, algorithm: str, n_clusters: Optional[int]) -> None:
        postfix = {"algorithm": algorithm}
        if n_clusters is not None:
            postfix["k"] = n_clusters
        try:
            bar.set_postfix(postfix)
        except Exception:
            return

    def _candidate_counts(self, n_samples: int) -> List[int]:
        if n_samples <= 1:
            return [1]
        upper = self.max_partitions if self.max_partitions is not None else self.min_partitions
        upper = max(self.min_partitions, int(upper))
        upper = min(upper, n_samples - 1 if n_samples > 2 else n_samples)
        lower = min(max(2, self.min_partitions), upper)
        candidates = list(range(lower, upper + 1))
        size_filtered = [
            candidate
            for candidate in candidates
            if n_samples / max(candidate, 1) >= self.min_auto_partition_size
        ]
        return size_filtered or candidates or [min(self.min_partitions, n_samples)]

    def _fit_count_based_candidate(
        self,
        algorithm: str,
        embedding: np.ndarray,
        n_clusters: int,
        target: Optional[Any],
    ) -> Optional[ClusterCandidate]:
        try:
            estimator = self._make_count_based_estimator(algorithm, n_clusters)
            if algorithm == "gmm":
                labels = estimator.fit_predict(embedding)
            else:
                labels = estimator.fit_predict(embedding)
        except Exception:
            return None
        return self._make_candidate(algorithm, int(n_clusters), labels, embedding, target, estimator)

    def _make_count_based_estimator(self, algorithm: str, n_clusters: int) -> Any:
        if algorithm == "kmeans":
            return self._make_kmeans(n_clusters)
        if algorithm == "bisecting_kmeans":
            if BisectingKMeans is None:
                raise RuntimeError("BisectingKMeans is not available in this sklearn version")
            return BisectingKMeans(n_clusters=n_clusters, random_state=self.random_state)
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
        target: Optional[Any],
    ) -> Optional[ClusterCandidate]:
        min_cluster_size = max(2, int(math.ceil(self.min_cluster_fraction * embedding.shape[0])))
        estimator = None
        labels = None
        try:
            if SklearnHDBSCAN is not None:
                estimator = SklearnHDBSCAN(min_cluster_size=min_cluster_size)
                labels = estimator.fit_predict(embedding)
            elif hdbscan_package is not None:
                estimator = hdbscan_package.HDBSCAN(min_cluster_size=min_cluster_size)
                labels = estimator.fit_predict(embedding)
        except Exception:
            return None
        if labels is None:
            return None
        normalized = self._normalize_labels(labels)
        n_clusters = int(np.unique(normalized).size)
        if n_clusters < 1:
            return None
        return self._make_candidate("hdbscan", n_clusters, normalized, embedding, target, estimator)

    def _make_candidate(
        self,
        algorithm: str,
        n_clusters: int,
        labels: np.ndarray,
        embedding: np.ndarray,
        target: Optional[Any],
        estimator: Any,
    ) -> ClusterCandidate:
        labels = self._normalize_labels(labels)
        centers = self._centers_from_labels(embedding, labels)
        components = self._score_components(embedding, labels, target)
        score = self._candidate_score(components)
        return ClusterCandidate(
            algorithm=algorithm,
            n_clusters=int(np.unique(labels).size),
            labels=labels,
            centers=centers,
            estimator=estimator,
            score=score,
            valid=bool(components["valid"]),
            components=components,
        )

    def _score_components(
        self,
        embedding: np.ndarray,
        labels: np.ndarray,
        target: Optional[Any],
    ) -> Dict[str, Any]:
        counts = np.bincount(labels, minlength=int(labels.max()) + 1)
        n_samples = int(labels.size)
        min_count = int(counts.min()) if counts.size else 0
        max_count = int(counts.max()) if counts.size else 0
        min_fraction = min_count / max(n_samples, 1)
        imbalance_ratio = max_count / max(min_count, 1)
        silhouette = self._safe_silhouette(embedding, labels)
        tiny_mass = float(counts[counts / max(n_samples, 1) < self.min_cluster_fraction].sum() / max(n_samples, 1))
        target_contrast = self._target_contrast(labels, target)
        valid = (
            imbalance_ratio <= self.max_cluster_imbalance_ratio
            and min_fraction >= self.min_cluster_fraction
        )
        return {
            "silhouette": silhouette,
            "imbalance_ratio": float(imbalance_ratio),
            "min_cluster_fraction": float(min_fraction),
            "tiny_cluster_mass": tiny_mass,
            "target_contrast": target_contrast,
            "valid": bool(valid),
            "counts": counts.astype(int).tolist(),
        }

    def _candidate_score(self, components: Dict[str, Any]) -> float:
        silhouette = components["silhouette"]
        if silhouette is None:
            silhouette = -1.0
        if self.selection_metric == "silhouette":
            return float(silhouette)
        imbalance_ratio = max(float(components["imbalance_ratio"]), 1.0)
        imbalance_penalty = self.imbalance_penalty_weight * math.log(imbalance_ratio)
        tiny_penalty = self.tiny_cluster_penalty_weight * float(components["tiny_cluster_mass"])
        target_bonus = self.target_contrast_weight * float(components["target_contrast"] or 0.0)
        hard_constraint_penalty = 0.0 if components["valid"] else 1.0
        return float(silhouette - imbalance_penalty - tiny_penalty + target_bonus - hard_constraint_penalty)

    def _safe_silhouette(self, embedding: np.ndarray, labels: np.ndarray) -> Optional[float]:
        unique_labels = np.unique(labels)
        if unique_labels.size < 2 or unique_labels.size >= embedding.shape[0]:
            return None
        sample_size = min(self.selection_sample_size, embedding.shape[0])
        try:
            return float(
                silhouette_score(
                    embedding,
                    labels,
                    sample_size=sample_size if sample_size < embedding.shape[0] else None,
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

    def _select_candidate(self, candidates: Sequence[ClusterCandidate]) -> ClusterCandidate:
        valid_candidates = [candidate for candidate in candidates if candidate.valid]
        pool = valid_candidates or list(candidates)
        if self.ensemble_method == "weighted_vote":
            return self._select_by_weighted_vote(pool)
        return max(pool, key=lambda candidate: candidate.score)

    def _select_by_weighted_vote(self, candidates: Sequence[ClusterCandidate]) -> ClusterCandidate:
        scores = np.asarray([candidate.score for candidate in candidates], dtype=float)
        scores = np.where(np.isfinite(scores), scores, -1e9)
        weights = np.exp((scores - scores.max()) / self.vote_temperature)
        votes: Dict[int, float] = {}
        for candidate, weight in zip(candidates, weights):
            votes[candidate.n_clusters] = votes.get(candidate.n_clusters, 0.0) + float(weight)
        selected_k = max(votes.items(), key=lambda item: (item[1], item[0]))[0]
        same_k = [candidate for candidate in candidates if candidate.n_clusters == selected_k]
        return max(same_k, key=lambda candidate: candidate.score)

    def _build_diagnostics(
        self,
        candidates: Sequence[ClusterCandidate],
        selected: ClusterCandidate,
    ) -> Dict[str, Any]:
        return {
            "selected_algorithm": selected.algorithm,
            "selected_n_clusters": int(selected.n_clusters),
            "cluster_algorithms": list(self.algorithms),
            "cluster_selection_metric": self.selection_metric,
            "cluster_ensemble_method": self.ensemble_method,
            "max_cluster_imbalance_ratio": float(self.max_cluster_imbalance_ratio),
            "min_cluster_fraction": float(self.min_cluster_fraction),
            "candidates": [
                {
                    "algorithm": candidate.algorithm,
                    "n_clusters": int(candidate.n_clusters),
                    "score": float(candidate.score),
                    "valid": bool(candidate.valid),
                    "components": candidate.components,
                }
                for candidate in candidates
            ],
        }

    def _make_kmeans(self, n_clusters: int) -> KMeans:
        try:
            return KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init="auto")
        except TypeError:
            return KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)

    @staticmethod
    def _normalize_algorithm(name: str) -> str:
        normalized = str(name).strip().lower()
        aliases = {
            "bisecting-kmeans": "bisecting_kmeans",
            "bisecting": "bisecting_kmeans",
            "gaussian_mixture": "gmm",
        }
        normalized = aliases.get(normalized, normalized)
        if normalized not in {"kmeans", "bisecting_kmeans", "gmm", "hdbscan"}:
            raise ValueError(f"Unsupported cluster algorithm: {name}")
        return normalized

    @staticmethod
    def _validate_choice(name: str, value: str, choices: Sequence[str]) -> str:
        if value not in choices:
            allowed = ", ".join(choices)
            raise ValueError(f"{name} must be one of: {allowed}")
        return value

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
