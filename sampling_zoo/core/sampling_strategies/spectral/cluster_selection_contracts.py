"""Typed pure-core contracts for spectral cluster selection."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, Optional, Sequence, Tuple


class ClusterCandidateKind(str, Enum):
    COUNT_BASED = "count_based"
    DENSITY_BASED = "density_based"


class ClusterCandidateRejectionReason(str, Enum):
    MIN_AVERAGE_PARTITION_SIZE = "min_average_partition_size"


class ClusterCandidateFitFailureCode(str, Enum):
    ADAPTER_UNAVAILABLE = "adapter_unavailable"
    FIT_FAILED = "fit_failed"


class ClusterConstraintViolation(str, Enum):
    MAX_IMBALANCE_RATIO = "max_imbalance_ratio"
    MIN_CLUSTER_FRACTION = "min_cluster_fraction"


@dataclass(frozen=True)
class ClusterConsensusSource:
    """One weighted partition contributing to a consensus representation."""

    key: str
    algorithm: str
    n_clusters: int
    score: float
    weight: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "algorithm": self.algorithm,
            "n_clusters": int(self.n_clusters),
            "score": float(self.score),
            "weight": float(self.weight),
        }


@dataclass(frozen=True)
class ClusterConsensusPlan:
    """Deterministic weights and target K for scalable co-association consensus."""

    method: str
    selected_n_clusters: int
    vote_temperature: float
    sources: Tuple[ClusterConsensusSource, ...]
    votes_by_n_clusters: Tuple[Tuple[int, float], ...]
    representation_columns: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "selected_n_clusters": int(self.selected_n_clusters),
            "vote_temperature": float(self.vote_temperature),
            "sources": [source.to_dict() for source in self.sources],
            "votes_by_n_clusters": {
                str(n_clusters): float(weight)
                for n_clusters, weight in self.votes_by_n_clusters
            },
            "representation_columns": int(self.representation_columns),
        }


@dataclass(frozen=True)
class ClusterCandidateRequest:
    """One effectful clustering adapter invocation planned by the pure core."""

    algorithm: str
    kind: ClusterCandidateKind
    n_clusters: Optional[int] = None

    @property
    def key(self) -> str:
        suffix = "auto" if self.n_clusters is None else str(self.n_clusters)
        return f"{self.algorithm}|k={suffix}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "algorithm": self.algorithm,
            "kind": self.kind.value,
            "n_clusters": self.n_clusters,
        }


@dataclass(frozen=True)
class ClusterCandidateRejection:
    """A count rejected by a planning guard before an adapter is invoked."""

    n_clusters: int
    estimated_average_partition_size: float
    reason: ClusterCandidateRejectionReason

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_clusters": int(self.n_clusters),
            "estimated_average_partition_size": float(
                self.estimated_average_partition_size
            ),
            "reason": self.reason.value,
        }


@dataclass(frozen=True)
class ClusterCandidatePlan:
    """Deterministic candidate grid with guard provenance and runtime requests."""

    n_samples: int
    min_partitions: int
    max_partitions: int
    min_auto_partition_size: int
    requested_count_candidates: Tuple[int, ...]
    eligible_count_candidates: Tuple[int, ...]
    size_guard_rejections: Tuple[ClusterCandidateRejection, ...]
    size_guard_fallback_applied: bool
    requests: Tuple[ClusterCandidateRequest, ...]

    @property
    def total_fit_count(self) -> int:
        return len(self.requests)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_samples": int(self.n_samples),
            "min_partitions": int(self.min_partitions),
            "max_partitions": int(self.max_partitions),
            "min_auto_partition_size": int(self.min_auto_partition_size),
            "requested_count_candidates": list(self.requested_count_candidates),
            "eligible_count_candidates": list(self.eligible_count_candidates),
            "size_guard_rejections": [
                rejection.to_dict() for rejection in self.size_guard_rejections
            ],
            "size_guard_fallback_applied": bool(self.size_guard_fallback_applied),
            "requests": [request.to_dict() for request in self.requests],
            "total_fit_count": int(self.total_fit_count),
        }


@dataclass(frozen=True)
class ClusterCandidateFitFailure:
    """Expected failure of one optional or data-dependent cluster adapter."""

    request: ClusterCandidateRequest
    code: ClusterCandidateFitFailureCode
    error_type: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "request": self.request.to_dict(),
            "code": self.code.value,
            "error_type": self.error_type,
            "message": self.message,
        }


@dataclass(frozen=True)
class ClusterScoreComponents:
    """Balanced-objective inputs and explicit hard-constraint violations."""

    silhouette: Optional[float]
    imbalance_ratio: float
    min_cluster_fraction: float
    tiny_cluster_mass: float
    target_contrast: float
    counts: Tuple[int, ...]
    violations: Tuple[ClusterConstraintViolation, ...]

    @property
    def valid(self) -> bool:
        return not self.violations

    def to_dict(self) -> dict[str, Any]:
        return {
            "silhouette": self.silhouette,
            "imbalance_ratio": float(self.imbalance_ratio),
            "min_cluster_fraction": float(self.min_cluster_fraction),
            "tiny_cluster_mass": float(self.tiny_cluster_mass),
            "target_contrast": float(self.target_contrast),
            "valid": bool(self.valid),
            "counts": list(self.counts),
            "constraint_violations": [violation.value for violation in self.violations],
        }


class ClusterSelectionUnavailableError(RuntimeError):
    """Raised when every planned clustering adapter invocation failed."""

    def __init__(
        self,
        plan: ClusterCandidatePlan,
        failures: Sequence[ClusterCandidateFitFailure],
    ) -> None:
        self.plan = plan
        self.failures = tuple(failures)
        codes = sorted({failure.code.value for failure in self.failures})
        detail = ", ".join(codes) if codes else "no candidate results"
        super().__init__(
            "No cluster candidates were generated from the planned requests: "
            f"{detail}"
        )


def normalize_cluster_algorithm(name: str) -> str:
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


def build_cluster_candidate_plan(
    *,
    n_samples: int,
    algorithms: Sequence[str],
    min_partitions: int,
    max_partitions: Optional[int],
    min_auto_partition_size: int,
) -> ClusterCandidatePlan:
    """Build the exact current candidate grid while retaining guard provenance."""

    n_samples = int(n_samples)
    if n_samples < 0:
        raise ValueError("n_samples must be non-negative")
    normalized_algorithms = tuple(
        normalize_cluster_algorithm(algorithm) for algorithm in algorithms
    )
    if not normalized_algorithms:
        raise ValueError("algorithms must contain at least one value")
    min_partitions = max(1, int(min_partitions))
    requested_max = min_partitions if max_partitions is None else int(max_partitions)
    requested_max = max(min_partitions, requested_max)
    min_auto_partition_size = max(1, int(min_auto_partition_size))

    requested_counts = _requested_candidate_counts(
        n_samples=n_samples,
        min_partitions=min_partitions,
        max_partitions=requested_max,
    )
    rejections = tuple(
        ClusterCandidateRejection(
            n_clusters=count,
            estimated_average_partition_size=(n_samples / max(count, 1)),
            reason=(ClusterCandidateRejectionReason.MIN_AVERAGE_PARTITION_SIZE),
        )
        for count in requested_counts
        if n_samples / max(count, 1) < min_auto_partition_size
    )
    rejected_counts = {rejection.n_clusters for rejection in rejections}
    size_eligible = tuple(
        count for count in requested_counts if count not in rejected_counts
    )
    fallback_applied = bool(requested_counts and not size_eligible)
    eligible_counts = requested_counts if fallback_applied else size_eligible
    requests = tuple(
        request
        for algorithm in normalized_algorithms
        for request in _requests_for_algorithm(algorithm, eligible_counts)
    )
    effective_max = max(requested_counts) if requested_counts else requested_max
    return ClusterCandidatePlan(
        n_samples=n_samples,
        min_partitions=min_partitions,
        max_partitions=int(effective_max),
        min_auto_partition_size=min_auto_partition_size,
        requested_count_candidates=requested_counts,
        eligible_count_candidates=eligible_counts,
        size_guard_rejections=rejections,
        size_guard_fallback_applied=fallback_applied,
        requests=requests,
    )


def evaluate_cluster_score_components(
    *,
    counts: Sequence[int],
    silhouette: Optional[float],
    target_contrast: float,
    max_cluster_imbalance_ratio: float,
    min_cluster_fraction: float,
) -> ClusterScoreComponents:
    normalized_counts = tuple(int(count) for count in counts)
    if not normalized_counts or any(count < 0 for count in normalized_counts):
        raise ValueError("counts must contain non-negative cluster sizes")
    n_samples = sum(normalized_counts)
    if n_samples < 1:
        raise ValueError("counts must contain at least one sample")
    min_count = min(normalized_counts)
    max_count = max(normalized_counts)
    observed_min_fraction = min_count / n_samples
    imbalance_ratio = max_count / max(min_count, 1)
    tiny_mass = (
        sum(
            count
            for count in normalized_counts
            if count / n_samples < float(min_cluster_fraction)
        )
        / n_samples
    )
    violations = []
    if imbalance_ratio > float(max_cluster_imbalance_ratio):
        violations.append(ClusterConstraintViolation.MAX_IMBALANCE_RATIO)
    if observed_min_fraction < float(min_cluster_fraction):
        violations.append(ClusterConstraintViolation.MIN_CLUSTER_FRACTION)
    return ClusterScoreComponents(
        silhouette=None if silhouette is None else float(silhouette),
        imbalance_ratio=float(imbalance_ratio),
        min_cluster_fraction=float(observed_min_fraction),
        tiny_cluster_mass=float(tiny_mass),
        target_contrast=float(target_contrast),
        counts=normalized_counts,
        violations=tuple(violations),
    )


def score_cluster_components(
    components: ClusterScoreComponents,
    *,
    selection_metric: str,
    imbalance_penalty_weight: float,
    tiny_cluster_penalty_weight: float,
    target_contrast_weight: float,
    hard_constraint_penalty: float = 1.0,
) -> float:
    silhouette = -1.0 if components.silhouette is None else components.silhouette
    if selection_metric == "silhouette":
        return float(silhouette)
    if selection_metric != "balanced_silhouette":
        raise ValueError("selection_metric must be silhouette or balanced_silhouette")
    imbalance_penalty = float(imbalance_penalty_weight) * math.log(
        max(components.imbalance_ratio, 1.0)
    )
    tiny_penalty = float(tiny_cluster_penalty_weight) * (components.tiny_cluster_mass)
    target_bonus = float(target_contrast_weight) * components.target_contrast
    constraint_penalty = 0.0 if components.valid else hard_constraint_penalty
    return float(
        silhouette
        - imbalance_penalty
        - tiny_penalty
        + target_bonus
        - constraint_penalty
    )


def candidate_fit_failure(
    request: ClusterCandidateRequest,
    error: BaseException,
    *,
    unavailable: bool = False,
) -> ClusterCandidateFitFailure:
    return ClusterCandidateFitFailure(
        request=request,
        code=(
            ClusterCandidateFitFailureCode.ADAPTER_UNAVAILABLE
            if unavailable
            else ClusterCandidateFitFailureCode.FIT_FAILED
        ),
        error_type=error.__class__.__name__,
        message=str(error),
    )


def build_cluster_consensus_plan(
    *,
    algorithms: Sequence[str],
    n_clusters: Sequence[int],
    scores: Sequence[float],
    vote_temperature: float,
) -> ClusterConsensusPlan:
    """Plan weighted partition consensus while preserving the legacy K vote."""

    normalized_algorithms = tuple(str(value) for value in algorithms)
    normalized_counts = tuple(int(value) for value in n_clusters)
    normalized_scores = tuple(float(value) for value in scores)
    source_count = len(normalized_algorithms)
    if source_count < 1:
        raise ValueError("consensus requires at least one cluster candidate")
    if len(normalized_counts) != source_count or len(normalized_scores) != source_count:
        raise ValueError("algorithms, n_clusters, and scores must have equal lengths")
    if any(value < 1 for value in normalized_counts):
        raise ValueError("consensus source cluster counts must be positive")
    temperature = float(vote_temperature)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("vote_temperature must be a positive finite value")

    finite_scores = tuple(
        score if math.isfinite(score) else -1e9 for score in normalized_scores
    )
    max_score = max(finite_scores)
    raw_weights = tuple(
        math.exp(max((score - max_score) / temperature, -745.0))
        for score in finite_scores
    )
    weight_sum = sum(raw_weights)
    if not math.isfinite(weight_sum) or weight_sum <= 0:
        normalized_weights = tuple(1.0 / source_count for _ in raw_weights)
    else:
        normalized_weights = tuple(weight / weight_sum for weight in raw_weights)

    sources = tuple(
        ClusterConsensusSource(
            key=f"{algorithm}|k={count}|candidate={index}",
            algorithm=algorithm,
            n_clusters=count,
            score=score,
            weight=weight,
        )
        for index, (algorithm, count, score, weight) in enumerate(
            zip(
                normalized_algorithms,
                normalized_counts,
                finite_scores,
                normalized_weights,
            )
        )
    )
    votes: dict[int, float] = {}
    for source in sources:
        votes[source.n_clusters] = votes.get(source.n_clusters, 0.0) + source.weight
    selected_n_clusters = max(votes.items(), key=lambda item: (item[1], item[0]))[0]
    return ClusterConsensusPlan(
        method="weighted_membership_kmeans",
        selected_n_clusters=int(selected_n_clusters),
        vote_temperature=temperature,
        sources=sources,
        votes_by_n_clusters=tuple(sorted(votes.items())),
        representation_columns=sum(source.n_clusters for source in sources),
    )


def _requested_candidate_counts(
    *,
    n_samples: int,
    min_partitions: int,
    max_partitions: int,
) -> Tuple[int, ...]:
    if n_samples <= 1:
        return (1,)
    upper = min(
        max(min_partitions, max_partitions),
        n_samples - 1 if n_samples > 2 else n_samples,
    )
    lower = min(max(2, min_partitions), upper)
    return tuple(range(lower, upper + 1))


def _requests_for_algorithm(
    algorithm: str,
    eligible_counts: Sequence[int],
) -> Tuple[ClusterCandidateRequest, ...]:
    if algorithm == "hdbscan":
        return (
            ClusterCandidateRequest(
                algorithm=algorithm,
                kind=ClusterCandidateKind.DENSITY_BASED,
            ),
        )
    return tuple(
        ClusterCandidateRequest(
            algorithm=algorithm,
            kind=ClusterCandidateKind.COUNT_BASED,
            n_clusters=int(count),
        )
        for count in eligible_counts
    )
