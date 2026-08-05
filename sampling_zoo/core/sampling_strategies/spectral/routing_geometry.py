"""Pure construction and evaluation of spectral partition geometry."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Optional

import numpy as np

from .routing_contracts import (
    PartitionGeometryContract,
    PartitionGeometrySpec,
    PartitionRepresentation,
    RoutingDistanceContract,
    RoutingKernel,
    RoutingMetric,
    RoutingValueKind,
    RoutingWeightContract,
    normalize_partition_names,
)


class PartitionGeometryBuilder:
    """Build immutable centers, scales, priors, and covariance geometry."""

    def build(
        self,
        *,
        embedding: np.ndarray,
        cluster_labels: np.ndarray,
        partitions: Mapping[str, np.ndarray],
        partition_to_cluster: Mapping[str, int],
        spec: PartitionGeometrySpec,
    ) -> PartitionGeometryContract:
        embedding = np.asarray(embedding, dtype=float)
        labels = np.asarray(cluster_labels)
        if embedding.ndim != 2 or embedding.shape[0] == 0:
            raise ValueError("embedding must be a non-empty 2D matrix")
        if labels.ndim != 1 or labels.shape[0] != embedding.shape[0]:
            raise ValueError("cluster_labels must align with embedding rows")

        names = normalize_partition_names(tuple(partitions))
        cluster_ids = tuple(int(partition_to_cluster[name]) for name in names)
        centers = []
        scales = []
        priors = []
        precisions = []
        log_determinants = []
        covariance_condition_numbers = []
        source_counts = []
        representation_counts = []
        needs_covariance = spec.metric in {
            RoutingMetric.DIAG_SHRINKAGE_MAHALANOBIS,
            RoutingMetric.FULL_SHRINKAGE_MAHALANOBIS,
        }

        for name, cluster_id in zip(names, cluster_ids):
            source_idx = np.flatnonzero(labels == cluster_id)
            sampled_idx = np.asarray(partitions[name], dtype=int)
            if source_idx.size == 0:
                raise ValueError(f"Partition {name!r} has no source rows")
            if sampled_idx.size == 0:
                raise ValueError(f"Partition {name!r} has no sampled rows")
            if np.any(sampled_idx < 0) or np.any(sampled_idx >= embedding.shape[0]):
                raise ValueError(f"Partition {name!r} contains out-of-range row indices")

            source_points = embedding[source_idx]
            representation_points = self._representation_points(
                embedding=embedding,
                source_idx=source_idx,
                sampled_idx=sampled_idx,
                representation=spec.representation,
            )
            center = self._center(representation_points, spec.representation)
            centered = representation_points - center
            squared_radius = np.sum(centered * centered, axis=1)
            scale = max(float(np.median(squared_radius)), float(spec.min_scale))

            centers.append(center)
            scales.append(scale)
            priors.append(max(float(source_idx.size), float(spec.min_prior)))
            source_counts.append(int(source_idx.size))
            representation_counts.append(int(representation_points.shape[0]))

            if needs_covariance:
                covariance = self._regularized_covariance(
                    representation_points,
                    shrinkage=float(spec.covariance_shrinkage),
                    min_scale=float(spec.min_scale),
                    diagonal=spec.metric == RoutingMetric.DIAG_SHRINKAGE_MAHALANOBIS,
                )
                sign, logdet = np.linalg.slogdet(covariance)
                if sign <= 0 or not np.isfinite(logdet):
                    raise ValueError(f"Partition {name!r} produced an invalid covariance")
                precisions.append(np.linalg.pinv(covariance, hermitian=True))
                log_determinants.append(float(logdet))
                covariance_condition_numbers.append(float(np.linalg.cond(covariance)))

        prior_values = np.asarray(priors, dtype=float)
        prior_values = prior_values / np.sum(prior_values)
        return PartitionGeometryContract(
            spec=spec,
            partition_names=names,
            cluster_ids=cluster_ids,
            centers=np.vstack(centers),
            scales=np.asarray(scales, dtype=float),
            priors=prior_values,
            precisions=np.stack(precisions) if precisions else None,
            log_determinants=np.asarray(log_determinants) if log_determinants else None,
            source_counts=tuple(source_counts),
            representation_counts=tuple(representation_counts),
            diagnostics={
                "source_count_total": int(sum(source_counts)),
                "representation_count_total": int(sum(representation_counts)),
                "scale_min": float(np.min(scales)),
                "scale_max": float(np.max(scales)),
                "prior_min": float(np.min(prior_values)),
                "prior_max": float(np.max(prior_values)),
                "covariance_shrinkage": float(spec.covariance_shrinkage),
                "covariance_condition_numbers": covariance_condition_numbers,
                "covariance_condition_number_median": (
                    float(np.median(covariance_condition_numbers))
                    if covariance_condition_numbers
                    else None
                ),
                "covariance_condition_number_max": (
                    float(np.max(covariance_condition_numbers))
                    if covariance_condition_numbers
                    else None
                ),
            },
        )

    @staticmethod
    def _representation_points(
        *,
        embedding: np.ndarray,
        source_idx: np.ndarray,
        sampled_idx: np.ndarray,
        representation: PartitionRepresentation,
    ) -> np.ndarray:
        if representation == PartitionRepresentation.SAMPLED_CENTROID:
            return embedding[sampled_idx]
        return embedding[source_idx]

    @staticmethod
    def _center(points: np.ndarray, representation: PartitionRepresentation) -> np.ndarray:
        centroid = np.mean(points, axis=0)
        if representation != PartitionRepresentation.MEDOID:
            return centroid
        distances = np.sum((points - centroid) ** 2, axis=1)
        return np.array(points[int(np.argmin(distances))], copy=True)

    @staticmethod
    def _regularized_covariance(
        points: np.ndarray,
        *,
        shrinkage: float,
        min_scale: float,
        diagonal: bool,
    ) -> np.ndarray:
        n_rows, n_features = points.shape
        if n_rows <= 1:
            covariance = np.zeros((n_features, n_features), dtype=float)
        else:
            covariance = np.asarray(np.cov(points, rowvar=False, bias=False), dtype=float)
            covariance = np.atleast_2d(covariance)
        if covariance.shape != (n_features, n_features):
            covariance = np.zeros((n_features, n_features), dtype=float)
        if not np.all(np.isfinite(covariance)):
            covariance = np.zeros((n_features, n_features), dtype=float)

        average_variance = float(np.trace(covariance) / max(n_features, 1))
        target_scale = max(average_variance, min_scale)
        target = np.eye(n_features, dtype=float) * target_scale
        covariance = (1.0 - shrinkage) * covariance + shrinkage * target
        if diagonal:
            covariance = np.diag(np.diag(covariance))
        covariance = covariance + np.eye(n_features, dtype=float) * min_scale
        return covariance


def route_partition_geometry(
    *,
    backend: Any,
    embedding: np.ndarray,
    geometry: PartitionGeometryContract,
    temperature: Optional[float] = None,
) -> tuple[RoutingDistanceContract, RoutingWeightContract]:
    """Evaluate fitted geometry through a numerical backend and normalize row weights."""

    effective_temperature = (
        float(geometry.spec.temperature) if temperature is None else float(temperature)
    )
    if effective_temperature <= 0:
        raise ValueError("routing temperature must be positive")
    values = backend.compute_routing_values(
        embedding,
        geometry.centers,
        metric=str(geometry.spec.metric),
        kernel=str(geometry.spec.kernel),
        scales=geometry.scales,
        precisions=geometry.precisions,
        log_determinants=geometry.log_determinants,
        priors=geometry.priors,
    )
    kind = (
        RoutingValueKind.LOG_DENSITY
        if geometry.spec.kernel == RoutingKernel.GMM_POSTERIOR
        else RoutingValueKind.DISTANCE
    )
    distance_contract = RoutingDistanceContract(
        partition_names=geometry.partition_names,
        values=values,
        kind=kind,
        diagnostics=_routing_value_diagnostics(values, kind),
    )
    weights = backend.normalize_routing_values(
        values,
        kind=str(kind),
        temperature=effective_temperature,
    )
    shrinkage = float(geometry.spec.uniform_shrinkage)
    if shrinkage > 0:
        weights = (1.0 - shrinkage) * weights + shrinkage / weights.shape[1]
        weights = weights / np.sum(weights, axis=1, keepdims=True)
    weight_contract = RoutingWeightContract(
        partition_names=geometry.partition_names,
        weights=weights,
        temperature=effective_temperature,
        diagnostics=_routing_weight_diagnostics(weights),
    )
    return distance_contract, weight_contract


def with_temperature(spec: PartitionGeometrySpec, temperature: float) -> PartitionGeometrySpec:
    """Return a validated copy used by routing replay calibration."""

    return replace(spec, temperature=float(temperature))


def _routing_value_diagnostics(values: np.ndarray, kind: RoutingValueKind) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    concentration = None
    if values.shape[0] and values.shape[1] > 1:
        ordered = np.sort(values, axis=1)
        if kind == RoutingValueKind.DISTANCE:
            best, second = ordered[:, 0], ordered[:, 1]
            separation = second - best
        else:
            best, second = ordered[:, -1], ordered[:, -2]
            separation = best - second
        scale = np.maximum(np.abs(best) + np.abs(second), 1e-12)
        concentration = float(np.mean(separation / scale))
    return {
        "kind": str(kind),
        "value_min": float(np.min(values)) if values.size else None,
        "value_max": float(np.max(values)) if values.size else None,
        "value_mean": float(np.mean(values)) if values.size else None,
        "finite_fraction": float(np.mean(np.isfinite(values))) if values.size else 1.0,
        "mean_relative_top2_separation": concentration,
    }


def _routing_weight_diagnostics(weights: np.ndarray) -> dict[str, Any]:
    weights = np.asarray(weights, dtype=float)
    if weights.shape[0] == 0:
        return {
            "n_rows": 0,
            "mean_max_probability": None,
            "mean_entropy": None,
            "effective_expert_count": None,
            "hard_assignment_counts": [0] * weights.shape[1],
            "dead_expert_count": int(weights.shape[1]),
        }
    entropy = -np.sum(weights * np.log(np.maximum(weights, 1e-12)), axis=1)
    ordered_weights = np.sort(weights, axis=1)
    top1_margin = (
        ordered_weights[:, -1] - ordered_weights[:, -2]
        if weights.shape[1] > 1
        else ordered_weights[:, -1]
    )
    assignments = np.argmax(weights, axis=1)
    counts = np.bincount(assignments, minlength=weights.shape[1])
    soft_mass = np.sum(weights, axis=0)
    return {
        "n_rows": int(weights.shape[0]),
        "mean_max_probability": float(np.mean(np.max(weights, axis=1))),
        "mean_top1_margin": float(np.mean(top1_margin)),
        "median_top1_margin": float(np.median(top1_margin)),
        "mean_entropy": float(np.mean(entropy)),
        "effective_expert_count": float(np.exp(np.mean(entropy))),
        "hard_assignment_counts": counts.astype(int).tolist(),
        "soft_assignment_mass": soft_mass.tolist(),
        "dead_expert_count": int(np.sum(soft_mass <= 1e-8)),
        "hard_assignment_dead_expert_count": int(np.sum(counts == 0)),
    }
