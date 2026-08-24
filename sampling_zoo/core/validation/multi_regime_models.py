"""Pure multi-regime controls and cluster-recovery metrics for RMT validation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics.cluster import contingency_matrix

from .spiked_models import NoiseDistribution


class RegimeProfile(str, Enum):
    BALANCED = "balanced"
    IMBALANCED_4_TO_1 = "imbalanced_4_to_1"

    @classmethod
    def parse(cls, value: str | "RegimeProfile") -> "RegimeProfile":
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).strip().lower())
        except ValueError as exc:
            allowed = ", ".join(profile.value for profile in cls)
            raise ValueError(
                f"Unknown regime_profile {value!r}. Expected one of: {allowed}"
            ) from exc


class PartitionProbePolicy(str, Enum):
    """Partition policies compared without changing production defaults."""

    FIXED_ORACLE = "fixed_oracle"
    AUTO_PRODUCTION = "auto_production"
    AUTO_UNRESTRICTED = "auto_unrestricted"

    @classmethod
    def parse(
        cls,
        value: str | "PartitionProbePolicy",
    ) -> "PartitionProbePolicy":
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).strip().lower())
        except ValueError as exc:
            allowed = ", ".join(policy.value for policy in cls)
            raise ValueError(
                f"Unknown partition_policy {value!r}. Expected one of: {allowed}"
            ) from exc


@dataclass(frozen=True)
class MultiRegimeDataConfig:
    """Specification of a mixture of affine low-rank regimes."""

    n_samples: int = 512
    n_features: int = 64
    n_regimes: int = 3
    local_rank: int = 2
    snr: float = 1.0
    regime_profile: RegimeProfile | str = RegimeProfile.BALANCED
    noise_distribution: NoiseDistribution | str = NoiseDistribution.GAUSSIAN
    regime_separation: float = 2.0
    local_signal_scale: float = 0.50
    student_t_df: float = 5.0
    random_state: int = 42

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "regime_profile",
            RegimeProfile.parse(self.regime_profile),
        )
        object.__setattr__(
            self,
            "noise_distribution",
            NoiseDistribution.parse(self.noise_distribution),
        )
        if int(self.n_regimes) < 2:
            raise ValueError("n_regimes must be at least 2")
        if int(self.local_rank) < 1:
            raise ValueError("local_rank must be positive")
        required_directions = int(self.n_regimes) * (int(self.local_rank) + 1)
        if int(self.n_features) < required_directions:
            raise ValueError("n_features must be at least n_regimes * (local_rank + 1)")
        if int(self.n_samples) < int(self.n_regimes) * (int(self.local_rank) + 2):
            raise ValueError("n_samples is too small for centered local regime factors")
        if float(self.snr) <= 0.0:
            raise ValueError("snr must be positive")
        if float(self.regime_separation) <= 0.0:
            raise ValueError("regime_separation must be positive")
        if float(self.local_signal_scale) <= 0.0:
            raise ValueError("local_signal_scale must be positive")
        if (
            self.noise_distribution is NoiseDistribution.STUDENT_T
            and float(self.student_t_df) <= 2.0
        ):
            raise ValueError("student_t_df must exceed 2")

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_samples": int(self.n_samples),
            "n_features": int(self.n_features),
            "n_regimes": int(self.n_regimes),
            "local_rank": int(self.local_rank),
            "snr": float(self.snr),
            "regime_profile": self.regime_profile.value,
            "noise_distribution": self.noise_distribution.value,
            "regime_separation": float(self.regime_separation),
            "local_signal_scale": float(self.local_signal_scale),
            "student_t_df": float(self.student_t_df),
            "random_state": int(self.random_state),
        }


@dataclass(frozen=True)
class MultiRegimeDataset:
    X: np.ndarray
    signal: np.ndarray
    noise: np.ndarray
    regime_labels: np.ndarray
    regime_sizes: tuple[int, ...]
    true_left_subspace: np.ndarray
    signal_singular_values: np.ndarray
    empirical_snr: float
    standardized_empirical_snr: float
    config: MultiRegimeDataConfig

    @property
    def dataset_name(self) -> str:
        return (
            f"multi_regime_{self.config.noise_distribution.value}"
            f"_{self.config.regime_profile.value}_k{self.config.n_regimes}"
        )

    def summary(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset_name,
            **self.config.to_dict(),
            "true_signal_rank": int(self.true_left_subspace.shape[1]),
            "regime_sizes": list(self.regime_sizes),
            "true_regime_imbalance_ratio": float(
                max(self.regime_sizes) / min(self.regime_sizes)
            ),
            "empirical_snr": float(self.empirical_snr),
            "standardized_empirical_snr": float(self.standardized_empirical_snr),
        }


@dataclass(frozen=True)
class ClusterRecoveryMetrics:
    true_n_clusters: int
    predicted_n_clusters: int
    cluster_count_signed_error: int
    cluster_count_absolute_error: int
    cluster_count_exact: bool
    adjusted_rand_index: float
    normalized_mutual_information: float
    aligned_accuracy: float
    purity: float
    mean_true_regime_recall: float
    min_true_regime_recall: float
    predicted_imbalance_ratio: float
    predicted_min_cluster_fraction: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "true_n_clusters": int(self.true_n_clusters),
            "predicted_n_clusters": int(self.predicted_n_clusters),
            "cluster_count_signed_error": int(self.cluster_count_signed_error),
            "cluster_count_absolute_error": int(self.cluster_count_absolute_error),
            "cluster_count_exact": bool(self.cluster_count_exact),
            "adjusted_rand_index": float(self.adjusted_rand_index),
            "normalized_mutual_information": float(self.normalized_mutual_information),
            "aligned_accuracy": float(self.aligned_accuracy),
            "purity": float(self.purity),
            "mean_true_regime_recall": float(self.mean_true_regime_recall),
            "min_true_regime_recall": float(self.min_true_regime_recall),
            "predicted_imbalance_ratio": float(self.predicted_imbalance_ratio),
            "predicted_min_cluster_fraction": float(
                self.predicted_min_cluster_fraction
            ),
        }


@dataclass(frozen=True)
class MultiRegimeValidationGridPoint:
    noise_distribution: NoiseDistribution
    regime_profile: RegimeProfile
    snr: float
    n_regimes: int
    seed: int
    backend: str
    view_strategy: str
    partition_policy: PartitionProbePolicy

    @property
    def key(self) -> str:
        return (
            f"{self.noise_distribution.value}|{self.regime_profile.value}"
            f"|snr={self.snr:.12g}|k={self.n_regimes}|seed={self.seed}"
            f"|backend={self.backend}|view={self.view_strategy}"
            f"|partition={self.partition_policy.value}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "noise_distribution": self.noise_distribution.value,
            "regime_profile": self.regime_profile.value,
            "snr": float(self.snr),
            "n_regimes": int(self.n_regimes),
            "seed": int(self.seed),
            "backend": self.backend,
            "view_strategy": self.view_strategy,
            "partition_policy": self.partition_policy.value,
            "grid_key": self.key,
        }


def build_multi_regime_validation_grid(
    *,
    snr_values: Sequence[float],
    n_regimes_values: Sequence[int],
    seeds: Sequence[int],
    noise_distributions: Sequence[str | NoiseDistribution],
    regime_profiles: Sequence[str | RegimeProfile],
    backends: Sequence[str],
    view_strategies: Sequence[str],
    partition_policies: Sequence[str | PartitionProbePolicy],
) -> tuple[MultiRegimeValidationGridPoint, ...]:
    dimensions = (
        snr_values,
        n_regimes_values,
        seeds,
        noise_distributions,
        regime_profiles,
        backends,
        view_strategies,
        partition_policies,
    )
    if any(len(dimension) == 0 for dimension in dimensions):
        raise ValueError("Multi-regime grid dimensions must not be empty")
    if any(float(snr) <= 0.0 for snr in snr_values):
        raise ValueError("snr_values must be positive")
    if any(int(value) < 2 for value in n_regimes_values):
        raise ValueError("n_regimes_values must be at least 2")
    normalized_backends = tuple(str(value).strip().lower() for value in backends)
    normalized_views = tuple(str(value).strip().lower() for value in view_strategies)
    unknown_backends = sorted(set(normalized_backends) - {"numpy", "torch"})
    unknown_views = sorted(set(normalized_views) - {"gaussian", "subsample"})
    if unknown_backends:
        raise ValueError(f"Unsupported backends: {unknown_backends}")
    if unknown_views:
        raise ValueError(f"Unsupported view_strategies: {unknown_views}")

    points = tuple(
        MultiRegimeValidationGridPoint(
            noise_distribution=NoiseDistribution.parse(noise),
            regime_profile=RegimeProfile.parse(profile),
            snr=float(snr),
            n_regimes=int(n_regimes),
            seed=int(seed),
            backend=backend,
            view_strategy=view,
            partition_policy=PartitionProbePolicy.parse(policy),
        )
        for noise in noise_distributions
        for profile in regime_profiles
        for n_regimes in n_regimes_values
        for view in normalized_views
        for backend in normalized_backends
        for seed in seeds
        for snr in snr_values
        for policy in partition_policies
    )
    keys = [point.key for point in points]
    if len(keys) != len(set(keys)):
        raise ValueError("Multi-regime grid contains duplicate leaf runs")
    return points


def generate_multi_regime_dataset(
    config: MultiRegimeDataConfig,
) -> MultiRegimeDataset:
    rng = np.random.default_rng(config.random_state)
    sizes = _regime_sizes(config)
    labels = np.repeat(np.arange(config.n_regimes), sizes)
    labels = labels[rng.permutation(config.n_samples)]

    direction_count = config.n_regimes * (config.local_rank + 1)
    directions, _ = np.linalg.qr(
        rng.normal(size=(config.n_features, direction_count)),
        mode="reduced",
    )
    centroids = directions[:, : config.n_regimes].T
    local_offset = config.n_regimes
    signal = np.zeros((config.n_samples, config.n_features), dtype=np.float64)
    for regime in range(config.n_regimes):
        indices = np.flatnonzero(labels == regime)
        start = local_offset + regime * config.local_rank
        loading = directions[:, start : start + config.local_rank]
        scores = rng.normal(size=(indices.size, config.local_rank))
        scores -= np.mean(scores, axis=0, keepdims=True)
        signal[indices] = config.regime_separation * centroids[
            regime
        ] + config.local_signal_scale * (scores @ loading.T)
    signal -= np.mean(signal, axis=0, keepdims=True)

    noise = _generate_noise(config, rng)
    noise -= np.mean(noise, axis=0, keepdims=True)
    target_noise_norm = np.linalg.norm(signal) / math.sqrt(config.snr)
    noise = _normalize_frobenius(noise, target_noise_norm)
    X = signal + noise
    left, singular_values, _ = np.linalg.svd(signal, full_matrices=False)
    tolerance = max(signal.shape) * np.finfo(np.float64).eps * singular_values[0]
    signal_rank = int(np.sum(singular_values > tolerance))

    return MultiRegimeDataset(
        X=_readonly(X),
        signal=_readonly(signal),
        noise=_readonly(noise),
        regime_labels=_readonly_int(labels),
        regime_sizes=tuple(map(int, sizes)),
        true_left_subspace=_readonly(left[:, :signal_rank]),
        signal_singular_values=_readonly(singular_values[:signal_rank]),
        empirical_snr=float(_frobenius_snr(signal, noise)),
        standardized_empirical_snr=float(_standardized_frobenius_snr(signal, noise)),
        config=config,
    )


def evaluate_cluster_recovery(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
) -> ClusterRecoveryMetrics:
    true_values = _validate_labels(true_labels, "true_labels")
    predicted_values = _validate_labels(predicted_labels, "predicted_labels")
    if true_values.shape != predicted_values.shape:
        raise ValueError("true_labels and predicted_labels must have equal shape")
    table = contingency_matrix(true_values, predicted_values, sparse=False)
    row_indices, column_indices = linear_sum_assignment(-table)
    matched = table[row_indices, column_indices]
    true_counts = np.sum(table, axis=1)
    recalls = np.zeros(table.shape[0], dtype=np.float64)
    recalls[row_indices] = matched / np.maximum(true_counts[row_indices], 1)
    predicted_counts = np.sum(table, axis=0)
    predicted_n = int(predicted_counts.size)
    true_n = int(true_counts.size)
    return ClusterRecoveryMetrics(
        true_n_clusters=true_n,
        predicted_n_clusters=predicted_n,
        cluster_count_signed_error=predicted_n - true_n,
        cluster_count_absolute_error=abs(predicted_n - true_n),
        cluster_count_exact=predicted_n == true_n,
        adjusted_rand_index=float(adjusted_rand_score(true_values, predicted_values)),
        normalized_mutual_information=float(
            normalized_mutual_info_score(true_values, predicted_values)
        ),
        aligned_accuracy=float(np.sum(matched) / true_values.size),
        purity=float(np.sum(np.max(table, axis=0)) / true_values.size),
        mean_true_regime_recall=float(np.mean(recalls)),
        min_true_regime_recall=float(np.min(recalls)),
        predicted_imbalance_ratio=float(
            np.max(predicted_counts) / max(int(np.min(predicted_counts)), 1)
        ),
        predicted_min_cluster_fraction=float(
            np.min(predicted_counts) / true_values.size
        ),
    )


def _regime_sizes(config: MultiRegimeDataConfig) -> np.ndarray:
    if config.regime_profile is RegimeProfile.BALANCED:
        weights = np.ones(config.n_regimes, dtype=np.float64)
    else:
        weights = np.geomspace(1.0, 0.25, config.n_regimes)
    weights /= np.sum(weights)
    raw = weights * config.n_samples
    sizes = np.floor(raw).astype(int)
    remainder = config.n_samples - int(np.sum(sizes))
    order = np.argsort(-(raw - sizes), kind="stable")
    sizes[order[:remainder]] += 1
    if np.any(sizes < config.local_rank + 2):
        raise ValueError("Regime profile produces too few rows for local factors")
    return sizes


def _generate_noise(
    config: MultiRegimeDataConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    shape = (config.n_samples, config.n_features)
    if config.noise_distribution is NoiseDistribution.GAUSSIAN:
        return rng.normal(size=shape)
    values = rng.standard_t(config.student_t_df, size=shape)
    return values / math.sqrt(config.student_t_df / (config.student_t_df - 2.0))


def _normalize_frobenius(values: np.ndarray, target_norm: float) -> np.ndarray:
    norm = float(np.linalg.norm(values))
    if not np.isfinite(norm) or norm <= np.finfo(np.float64).eps:
        raise ValueError("Cannot normalize a zero or non-finite noise realization")
    return np.asarray(values, dtype=np.float64) * (float(target_norm) / norm)


def _frobenius_snr(signal: np.ndarray, noise: np.ndarray) -> float:
    return float(np.sum(signal * signal) / np.sum(noise * noise))


def _standardized_frobenius_snr(
    signal: np.ndarray,
    noise: np.ndarray,
) -> float:
    scales = np.std(signal + noise, axis=0)
    scales = np.where(scales > np.finfo(np.float64).eps, scales, 1.0)
    return _frobenius_snr(signal / scales, noise / scales)


def _validate_labels(values: np.ndarray, name: str) -> np.ndarray:
    labels = np.asarray(values)
    if labels.ndim != 1 or labels.size == 0:
        raise ValueError(f"{name} must be a non-empty 1D array")
    if not np.all(np.isfinite(labels.astype(float))):
        raise ValueError(f"{name} must contain finite values")
    return labels


def _readonly(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    result.setflags(write=False)
    return result


def _readonly_int(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.int64).copy()
    result.setflags(write=False)
    return result
