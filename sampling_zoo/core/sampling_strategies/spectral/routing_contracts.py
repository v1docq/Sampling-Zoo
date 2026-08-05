"""Typed contracts for partition geometry and spectral routing."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence

import numpy as np


class _StrEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class PartitionRepresentation(_StrEnum):
    SOURCE_CENTROID = "source_centroid"
    SAMPLED_CENTROID = "sampled_centroid"
    MEDOID = "medoid"


class RoutingMetric(_StrEnum):
    SQUARED_EUCLIDEAN = "squared_euclidean"
    MEDIAN_SCALED_EUCLIDEAN = "median_scaled_euclidean"
    DIAG_SHRINKAGE_MAHALANOBIS = "diag_shrinkage_mahalanobis"
    FULL_SHRINKAGE_MAHALANOBIS = "full_shrinkage_mahalanobis"
    COSINE = "cosine"


class RoutingKernel(_StrEnum):
    SOFTMAX = "softmax"
    GMM_POSTERIOR = "gmm_posterior"


class RoutingValueKind(_StrEnum):
    DISTANCE = "distance"
    LOG_DENSITY = "log_density"


def _coerce_enum(enum_type: type[_StrEnum], value: Any, name: str) -> _StrEnum:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(str(value).strip().lower())
    except ValueError as exc:
        choices = ", ".join(member.value for member in enum_type)
        raise ValueError(f"{name} must be one of: {choices}") from exc


def _readonly_array(value: Any, *, ndim: Optional[int] = None) -> np.ndarray:
    array = np.array(value, dtype=float, copy=True)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"Expected a {ndim}D array, got shape {array.shape}")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class PartitionGeometrySpec:
    """Validated policy for representing partitions and converting proximity to weights."""

    representation: PartitionRepresentation | str = PartitionRepresentation.SOURCE_CENTROID
    metric: RoutingMetric | str = RoutingMetric.SQUARED_EUCLIDEAN
    kernel: RoutingKernel | str = RoutingKernel.SOFTMAX
    temperature: float = 1.0
    covariance_shrinkage: float = 0.10
    uniform_shrinkage: float = 0.05
    min_scale: float = 1e-8
    min_prior: float = 1e-8

    def __post_init__(self) -> None:
        representation = _coerce_enum(
            PartitionRepresentation,
            self.representation,
            "routing_representation",
        )
        metric = _coerce_enum(RoutingMetric, self.metric, "routing_metric")
        kernel = _coerce_enum(RoutingKernel, self.kernel, "routing_kernel")
        object.__setattr__(self, "representation", representation)
        object.__setattr__(self, "metric", metric)
        object.__setattr__(self, "kernel", kernel)

        if float(self.temperature) <= 0:
            raise ValueError("routing_temperature must be positive")
        if not 0.0 <= float(self.covariance_shrinkage) <= 1.0:
            raise ValueError("routing_covariance_shrinkage must be in [0, 1]")
        if not 0.0 <= float(self.uniform_shrinkage) <= 1.0:
            raise ValueError("routing_shrinkage must be in [0, 1]")
        if float(self.min_scale) <= 0:
            raise ValueError("routing_min_scale must be positive")
        if float(self.min_prior) <= 0:
            raise ValueError("routing_min_prior must be positive")
        if kernel == RoutingKernel.GMM_POSTERIOR and metric not in {
            RoutingMetric.DIAG_SHRINKAGE_MAHALANOBIS,
            RoutingMetric.FULL_SHRINKAGE_MAHALANOBIS,
        }:
            raise ValueError(
                "gmm_posterior requires diag_shrinkage_mahalanobis or "
                "full_shrinkage_mahalanobis"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "representation": str(self.representation),
            "metric": str(self.metric),
            "kernel": str(self.kernel),
            "temperature": float(self.temperature),
            "covariance_shrinkage": float(self.covariance_shrinkage),
            "uniform_shrinkage": float(self.uniform_shrinkage),
            "min_scale": float(self.min_scale),
            "min_prior": float(self.min_prior),
        }


@dataclass(frozen=True)
class PartitionGeometryContract:
    """Fitted geometry aligned one-to-one with active partition names."""

    spec: PartitionGeometrySpec
    partition_names: tuple[str, ...]
    cluster_ids: tuple[int, ...]
    centers: np.ndarray
    scales: np.ndarray
    priors: np.ndarray
    precisions: Optional[np.ndarray] = None
    log_determinants: Optional[np.ndarray] = None
    source_counts: tuple[int, ...] = ()
    representation_counts: tuple[int, ...] = ()
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        names = tuple(str(name) for name in self.partition_names)
        cluster_ids = tuple(int(value) for value in self.cluster_ids)
        centers = _readonly_array(self.centers, ndim=2)
        scales = _readonly_array(self.scales, ndim=1)
        priors = _readonly_array(self.priors, ndim=1)
        n_partitions = len(names)
        if n_partitions < 1:
            raise ValueError("Partition geometry requires at least one partition")
        if len(set(names)) != n_partitions:
            raise ValueError("partition_names must be unique")
        if len(cluster_ids) != n_partitions:
            raise ValueError("cluster_ids must align with partition_names")
        if centers.shape[0] != n_partitions:
            raise ValueError("centers must align with partition_names")
        if scales.shape != (n_partitions,) or priors.shape != (n_partitions,):
            raise ValueError("scales and priors must align with partition_names")
        if not np.all(np.isfinite(centers)):
            raise ValueError("centers must be finite")
        if np.any(scales <= 0) or not np.all(np.isfinite(scales)):
            raise ValueError("scales must be positive and finite")
        if np.any(priors <= 0) or not np.all(np.isfinite(priors)):
            raise ValueError("priors must be positive and finite")
        priors = priors / np.sum(priors)
        priors.setflags(write=False)

        precisions = None
        if self.precisions is not None:
            precisions = _readonly_array(self.precisions, ndim=3)
            expected = (n_partitions, centers.shape[1], centers.shape[1])
            if precisions.shape != expected:
                raise ValueError(f"precisions must have shape {expected}")

        log_determinants = None
        if self.log_determinants is not None:
            log_determinants = _readonly_array(self.log_determinants, ndim=1)
            if log_determinants.shape != (n_partitions,):
                raise ValueError("log_determinants must align with partition_names")

        object.__setattr__(self, "partition_names", names)
        object.__setattr__(self, "cluster_ids", cluster_ids)
        object.__setattr__(self, "centers", centers)
        object.__setattr__(self, "scales", scales)
        object.__setattr__(self, "priors", priors)
        object.__setattr__(self, "precisions", precisions)
        object.__setattr__(self, "log_determinants", log_determinants)
        object.__setattr__(self, "source_counts", tuple(int(v) for v in self.source_counts))
        object.__setattr__(
            self,
            "representation_counts",
            tuple(int(v) for v in self.representation_counts),
        )
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))

    @property
    def n_partitions(self) -> int:
        return len(self.partition_names)

    @property
    def embedding_dim(self) -> int:
        return int(self.centers.shape[1])

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec": self.spec.to_dict(),
            "partition_names": list(self.partition_names),
            "cluster_ids": list(self.cluster_ids),
            "source_counts": list(self.source_counts),
            "representation_counts": list(self.representation_counts),
            "scales": self.scales.tolist(),
            "priors": self.priors.tolist(),
            "embedding_dim": self.embedding_dim,
            "has_precision": self.precisions is not None,
            "diagnostics": dict(self.diagnostics),
        }


@dataclass(frozen=True)
class RoutingDistanceContract:
    """Backend output before row-wise probability normalization."""

    partition_names: tuple[str, ...]
    values: np.ndarray
    kind: RoutingValueKind | str
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = _readonly_array(self.values, ndim=2)
        kind = _coerce_enum(RoutingValueKind, self.kind, "routing_value_kind")
        if values.shape[1] != len(self.partition_names):
            raise ValueError("Routing values must align with partition_names")
        object.__setattr__(self, "partition_names", tuple(self.partition_names))
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))


@dataclass(frozen=True)
class RoutingWeightContract:
    """Normalized row-wise routing weights aligned with partition names."""

    partition_names: tuple[str, ...]
    weights: np.ndarray
    temperature: float
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        weights = _readonly_array(self.weights, ndim=2)
        if weights.shape[1] != len(self.partition_names):
            raise ValueError("Routing weights must align with partition_names")
        if np.any(weights < 0) or not np.all(np.isfinite(weights)):
            raise ValueError("Routing weights must be finite and non-negative")
        if weights.shape[0] and not np.allclose(np.sum(weights, axis=1), 1.0, atol=1e-6):
            raise ValueError("Routing weights must sum to one for every row")
        object.__setattr__(self, "partition_names", tuple(self.partition_names))
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))


def normalize_partition_names(names: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(str(name) for name in names)
    if not normalized:
        raise ValueError("At least one partition name is required")
    if len(set(normalized)) != len(normalized):
        raise ValueError("Partition names must be unique")
    return normalized
