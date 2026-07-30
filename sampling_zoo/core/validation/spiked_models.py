"""Pure contracts, generators, and recovery metrics for spiked RMT controls."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, Mapping, Optional, Sequence

import numpy as np


class NoiseDistribution(str, Enum):
    """Noise families used by the first RMT synthetic validation slice."""

    GAUSSIAN = "gaussian"
    STUDENT_T = "student_t"

    @classmethod
    def parse(cls, value: str | "NoiseDistribution") -> "NoiseDistribution":
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).strip().lower())
        except ValueError as exc:
            allowed = ", ".join(item.value for item in cls)
            raise ValueError(
                f"Unknown noise_distribution {value!r}. Expected one of: {allowed}"
            ) from exc


@dataclass(frozen=True)
class SpikedDataConfig:
    """Validated specification of one controlled low-rank matrix."""

    n_samples: int = 512
    n_features: int = 64
    true_rank: int = 3
    snr: float = 1.0
    noise_distribution: NoiseDistribution | str = NoiseDistribution.GAUSSIAN
    min_spike_ratio: float = 0.50
    student_t_df: float = 5.0
    random_state: int = 42

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "noise_distribution",
            NoiseDistribution.parse(self.noise_distribution),
        )
        if int(self.n_samples) < 2:
            raise ValueError("n_samples must be at least 2")
        if int(self.n_features) < 1:
            raise ValueError("n_features must be positive")
        if int(self.true_rank) < 0:
            raise ValueError("true_rank must be non-negative")
        max_rank = min(int(self.n_samples) - 1, int(self.n_features))
        if int(self.true_rank) > max_rank:
            raise ValueError(
                "true_rank must not exceed min(n_samples - 1, n_features)"
            )
        if int(self.true_rank) == 0 and float(self.snr) != 0.0:
            raise ValueError("snr must be 0 when true_rank is 0")
        if int(self.true_rank) > 0 and float(self.snr) <= 0.0:
            raise ValueError("snr must be positive when true_rank is positive")
        if not 0.0 < float(self.min_spike_ratio) <= 1.0:
            raise ValueError("min_spike_ratio must be in (0, 1]")
        if (
            self.noise_distribution is NoiseDistribution.STUDENT_T
            and float(self.student_t_df) <= 2.0
        ):
            raise ValueError(
                "student_t_df must exceed 2 so the noise variance is finite"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_samples": int(self.n_samples),
            "n_features": int(self.n_features),
            "true_rank": int(self.true_rank),
            "snr": float(self.snr),
            "noise_distribution": self.noise_distribution.value,
            "min_spike_ratio": float(self.min_spike_ratio),
            "student_t_df": float(self.student_t_df),
            "random_state": int(self.random_state),
        }


@dataclass(frozen=True)
class SpikedDataset:
    """One generated matrix plus the latent signal needed for recovery checks."""

    X: np.ndarray
    signal: np.ndarray
    noise: np.ndarray
    true_left_subspace: np.ndarray
    true_right_subspace: np.ndarray
    signal_singular_values: np.ndarray
    empirical_snr: float
    standardized_empirical_snr: float
    config: SpikedDataConfig

    @property
    def dataset_name(self) -> str:
        return (
            f"spiked_{self.config.noise_distribution.value}"
            f"_rank_{self.config.true_rank}"
        )

    def summary(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset_name,
            **self.config.to_dict(),
            "empirical_snr": float(self.empirical_snr),
            "standardized_empirical_snr": float(
                self.standardized_empirical_snr
            ),
            "signal_frobenius_norm": float(np.linalg.norm(self.signal)),
            "noise_frobenius_norm": float(np.linalg.norm(self.noise)),
        }


@dataclass(frozen=True)
class RankRecoveryMetrics:
    """Count-level comparison between a detected and a known signal rank."""

    true_rank: int
    estimated_rank: int
    signed_error: int
    absolute_error: int
    precision: float
    recall: float
    f1: float
    exact: bool
    false_positive: bool

    def to_dict(self, prefix: str = "") -> dict[str, Any]:
        stem = f"{prefix}_" if prefix else ""
        return {
            f"{stem}estimated_rank": int(self.estimated_rank),
            f"{stem}rank_signed_error": int(self.signed_error),
            f"{stem}rank_absolute_error": int(self.absolute_error),
            f"{stem}rank_precision": float(self.precision),
            f"{stem}rank_recall": float(self.recall),
            f"{stem}rank_f1": float(self.f1),
            f"{stem}rank_exact": bool(self.exact),
            f"{stem}rank_false_positive": bool(self.false_positive),
        }


@dataclass(frozen=True)
class SubspaceRecoveryMetrics:
    """Rotation-invariant recovery for true and estimated sample subspaces."""

    true_rank: int
    estimated_rank: int
    overlap: float
    recall: float
    precision: float
    f1: float
    missed_subspace_distance: float
    mean_principal_angle_degrees: Optional[float]
    max_principal_angle_degrees: Optional[float]
    min_canonical_correlation: Optional[float]

    def to_dict(self, prefix: str = "selected_subspace") -> dict[str, Any]:
        return {
            f"{prefix}_estimated_rank": int(self.estimated_rank),
            f"{prefix}_overlap": float(self.overlap),
            f"{prefix}_recall": float(self.recall),
            f"{prefix}_precision": float(self.precision),
            f"{prefix}_f1": float(self.f1),
            f"{prefix}_missed_distance": float(self.missed_subspace_distance),
            f"{prefix}_mean_principal_angle_degrees": (
                self.mean_principal_angle_degrees
            ),
            f"{prefix}_max_principal_angle_degrees": (
                self.max_principal_angle_degrees
            ),
            f"{prefix}_min_canonical_correlation": (
                self.min_canonical_correlation
            ),
        }


@dataclass(frozen=True)
class SpikedRecoveryEvaluation:
    """Recovery views for all rank diagnostics emitted by the RMT sampler."""

    true_rank: int
    explained_variance: RankRecoveryMetrics
    null_edge: Optional[RankRecoveryMetrics]
    view_stability: Optional[RankRecoveryMetrics]
    subspace_stability: Optional[RankRecoveryMetrics]
    selected_subspace: Optional[SubspaceRecoveryMetrics]

    def to_flat_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "true_rank": int(self.true_rank),
            **self.explained_variance.to_dict("explained"),
        }
        for prefix, metrics in (
            ("null_edge", self.null_edge),
            ("view_stability", self.view_stability),
            ("subspace_stability", self.subspace_stability),
        ):
            if metrics is None:
                result.update(_empty_rank_metrics(prefix))
            else:
                result.update(metrics.to_dict(prefix))
        if self.selected_subspace is None:
            result.update(_empty_subspace_metrics())
        else:
            result.update(self.selected_subspace.to_dict())
        return result


@dataclass(frozen=True)
class SpikedValidationGridPoint:
    """One deterministic leaf run of the synthetic validation experiment."""

    noise_distribution: NoiseDistribution
    snr: float
    true_rank: int
    seed: int
    backend: str
    view_strategy: str

    @property
    def key(self) -> str:
        return (
            f"{self.noise_distribution.value}|snr={self.snr:.12g}"
            f"|rank={self.true_rank}|seed={self.seed}"
            f"|backend={self.backend}|view={self.view_strategy}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "noise_distribution": self.noise_distribution.value,
            "snr": float(self.snr),
            "true_rank": int(self.true_rank),
            "seed": int(self.seed),
            "backend": self.backend,
            "view_strategy": self.view_strategy,
            "grid_key": self.key,
        }


def build_spiked_validation_grid(
    *,
    snr_values: Sequence[float],
    seeds: Sequence[int],
    noise_distributions: Sequence[str | NoiseDistribution],
    backends: Sequence[str],
    view_strategies: Sequence[str],
    true_rank: int,
    include_null_control: bool = True,
) -> tuple[SpikedValidationGridPoint, ...]:
    """Build a stable Cartesian grid with optional rank-zero controls."""

    normalized_noise = tuple(
        NoiseDistribution.parse(value) for value in noise_distributions
    )
    normalized_backends = tuple(str(value).strip().lower() for value in backends)
    normalized_views = tuple(
        str(value).strip().lower() for value in view_strategies
    )
    if not snr_values:
        raise ValueError("snr_values must not be empty")
    if not seeds:
        raise ValueError("seeds must not be empty")
    if not normalized_noise or not normalized_backends or not normalized_views:
        raise ValueError(
            "noise_distributions, backends, and view_strategies must not be empty"
        )
    if any(float(value) <= 0.0 for value in snr_values):
        raise ValueError("signal snr_values must be positive")
    unknown_backends = sorted(set(normalized_backends) - {"numpy", "torch"})
    if unknown_backends:
        raise ValueError(f"Unsupported backends: {unknown_backends}")
    unknown_views = sorted(set(normalized_views) - {"gaussian", "subsample"})
    if unknown_views:
        raise ValueError(f"Unsupported view_strategies: {unknown_views}")
    if int(true_rank) < 1:
        raise ValueError("true_rank must be positive for the signal grid")

    points: list[SpikedValidationGridPoint] = []
    for noise in normalized_noise:
        for view_strategy in normalized_views:
            for backend in normalized_backends:
                for seed in map(int, seeds):
                    if include_null_control:
                        points.append(
                            SpikedValidationGridPoint(
                                noise_distribution=noise,
                                snr=0.0,
                                true_rank=0,
                                seed=seed,
                                backend=backend,
                                view_strategy=view_strategy,
                            )
                        )
                    points.extend(
                        SpikedValidationGridPoint(
                            noise_distribution=noise,
                            snr=float(snr),
                            true_rank=int(true_rank),
                            seed=seed,
                            backend=backend,
                            view_strategy=view_strategy,
                        )
                        for snr in map(float, snr_values)
                    )
    keys = [point.key for point in points]
    if len(keys) != len(set(keys)):
        raise ValueError("Synthetic validation grid contains duplicate leaf runs")
    return tuple(points)


def generate_spiked_dataset(config: SpikedDataConfig) -> SpikedDataset:
    """Generate X = signal + noise with exact empirical Frobenius SNR."""

    rng = np.random.default_rng(config.random_state)
    noise = _generate_noise(config, rng)
    noise = _center_columns(noise)

    if config.true_rank == 0:
        noise = _normalize_frobenius(
            noise,
            math.sqrt(config.n_samples * config.n_features),
        )
        signal = np.zeros_like(noise)
        left = np.empty((config.n_samples, 0), dtype=np.float64)
        right = np.empty((config.n_features, 0), dtype=np.float64)
        singular_values = np.empty(0, dtype=np.float64)
        empirical_snr = 0.0
        standardized_empirical_snr = 0.0
    else:
        left = _orthonormal_basis(
            rng,
            n_rows=config.n_samples,
            rank=config.true_rank,
            center_columns=True,
        )
        right = _orthonormal_basis(
            rng,
            n_rows=config.n_features,
            rank=config.true_rank,
            center_columns=False,
        )
        singular_values = np.linspace(
            1.0,
            config.min_spike_ratio,
            config.true_rank,
            dtype=np.float64,
        )
        signal = (left * singular_values.reshape(1, -1)) @ right.T
        target_noise_norm = np.linalg.norm(signal) / math.sqrt(config.snr)
        noise = _normalize_frobenius(noise, target_noise_norm)
        empirical_snr = _frobenius_snr(signal, noise)
        standardized_empirical_snr = _standardized_frobenius_snr(
            signal,
            noise,
        )

    X = signal + noise
    return SpikedDataset(
        X=_readonly(X),
        signal=_readonly(signal),
        noise=_readonly(noise),
        true_left_subspace=_readonly(left),
        true_right_subspace=_readonly(right),
        signal_singular_values=_readonly(singular_values),
        empirical_snr=float(empirical_snr),
        standardized_empirical_snr=float(standardized_empirical_snr),
        config=config,
    )


def evaluate_rank_recovery(
    true_rank: int,
    estimated_rank: int,
) -> RankRecoveryMetrics:
    """Treat rank estimates as nested component sets and compare counts."""

    true_rank = int(true_rank)
    estimated_rank = int(estimated_rank)
    if true_rank < 0 or estimated_rank < 0:
        raise ValueError("Ranks must be non-negative")
    exact = estimated_rank == true_rank
    if true_rank == 0:
        precision = recall = f1 = 1.0 if exact else 0.0
    else:
        overlap = min(true_rank, estimated_rank)
        precision = overlap / estimated_rank if estimated_rank else 0.0
        recall = overlap / true_rank
        f1 = _harmonic_mean(precision, recall)
    return RankRecoveryMetrics(
        true_rank=true_rank,
        estimated_rank=estimated_rank,
        signed_error=estimated_rank - true_rank,
        absolute_error=abs(estimated_rank - true_rank),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        exact=exact,
        false_positive=true_rank == 0 and estimated_rank > 0,
    )


def evaluate_subspace_recovery(
    true_basis: np.ndarray,
    estimated_basis: np.ndarray,
) -> Optional[SubspaceRecoveryMetrics]:
    """Compare unequal-dimensional subspaces without matching basis vectors."""

    true_values = _validate_basis(true_basis, "true_basis")
    estimated_values = _validate_basis(estimated_basis, "estimated_basis")
    if true_values.shape[0] != estimated_values.shape[0]:
        raise ValueError("Subspace bases must have the same number of rows")
    true_rank = int(true_values.shape[1])
    estimated_rank = int(estimated_values.shape[1])
    if true_rank == 0:
        return None
    if estimated_rank == 0:
        return SubspaceRecoveryMetrics(
            true_rank=true_rank,
            estimated_rank=0,
            overlap=0.0,
            recall=0.0,
            precision=0.0,
            f1=0.0,
            missed_subspace_distance=1.0,
            mean_principal_angle_degrees=None,
            max_principal_angle_degrees=None,
            min_canonical_correlation=None,
        )

    true_q, _ = np.linalg.qr(true_values, mode="reduced")
    estimated_q, _ = np.linalg.qr(estimated_values, mode="reduced")
    correlations = np.linalg.svd(
        true_q.T @ estimated_q,
        compute_uv=False,
    )
    correlations = np.clip(correlations, 0.0, 1.0)
    overlap = float(np.sum(correlations * correlations))
    recall = min(1.0, overlap / true_rank)
    precision = min(1.0, overlap / estimated_rank)
    angles = np.degrees(np.arccos(correlations))
    return SubspaceRecoveryMetrics(
        true_rank=true_rank,
        estimated_rank=estimated_rank,
        overlap=overlap,
        recall=float(recall),
        precision=float(precision),
        f1=float(_harmonic_mean(precision, recall)),
        missed_subspace_distance=float(
            math.sqrt(max(0.0, 1.0 - recall))
        ),
        mean_principal_angle_degrees=float(np.mean(angles)),
        max_principal_angle_degrees=float(np.max(angles)),
        min_canonical_correlation=float(np.min(correlations)),
    )


def evaluate_spiked_recovery(
    *,
    true_left_subspace: np.ndarray,
    estimated_left_subspace: np.ndarray,
    sampler_diagnostics: Mapping[str, Any],
) -> SpikedRecoveryEvaluation:
    """Map sampler diagnostics and its fitted basis to recovery contracts."""

    true_rank = int(true_left_subspace.shape[1])

    def optional_rank(name: str) -> Optional[RankRecoveryMetrics]:
        value = sampler_diagnostics.get(name)
        if value is None:
            return None
        return evaluate_rank_recovery(true_rank, int(value))

    selected_rank = sampler_diagnostics.get("rank_by_explained_variance")
    if selected_rank is None:
        selected_rank = estimated_left_subspace.shape[1]
    return SpikedRecoveryEvaluation(
        true_rank=true_rank,
        explained_variance=evaluate_rank_recovery(
            true_rank,
            int(selected_rank),
        ),
        null_edge=optional_rank("rank_by_null_edge"),
        view_stability=optional_rank("rank_by_stability"),
        subspace_stability=optional_rank("rank_by_subspace_stability"),
        selected_subspace=evaluate_subspace_recovery(
            true_left_subspace,
            estimated_left_subspace,
        ),
    )


def _generate_noise(
    config: SpikedDataConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    shape = (config.n_samples, config.n_features)
    if config.noise_distribution is NoiseDistribution.GAUSSIAN:
        return rng.normal(size=shape)
    values = rng.standard_t(config.student_t_df, size=shape)
    return values / math.sqrt(config.student_t_df / (config.student_t_df - 2.0))


def _orthonormal_basis(
    rng: np.random.Generator,
    *,
    n_rows: int,
    rank: int,
    center_columns: bool,
) -> np.ndarray:
    raw = rng.normal(size=(n_rows, rank))
    if center_columns:
        raw = _center_columns(raw)
    basis, upper = np.linalg.qr(raw, mode="reduced")
    signs = np.sign(np.diag(upper))
    signs[signs == 0.0] = 1.0
    return basis * signs.reshape(1, -1)


def _center_columns(values: np.ndarray) -> np.ndarray:
    return values - np.mean(values, axis=0, keepdims=True)


def _normalize_frobenius(values: np.ndarray, target_norm: float) -> np.ndarray:
    norm = float(np.linalg.norm(values))
    if not np.isfinite(norm) or norm <= np.finfo(np.float64).eps:
        raise ValueError("Cannot normalize a zero or non-finite noise realization")
    return np.asarray(values, dtype=np.float64) * (float(target_norm) / norm)


def _frobenius_snr(signal: np.ndarray, noise: np.ndarray) -> float:
    noise_energy = float(np.sum(noise * noise))
    if noise_energy <= 0.0:
        return math.inf
    return float(np.sum(signal * signal) / noise_energy)


def _standardized_frobenius_snr(
    signal: np.ndarray,
    noise: np.ndarray,
) -> float:
    """Return the effective SNR after the sampler's per-feature scaling."""

    scales = np.std(signal + noise, axis=0)
    scales = np.where(
        scales > np.finfo(np.float64).eps,
        scales,
        1.0,
    )
    return _frobenius_snr(
        signal / scales.reshape(1, -1),
        noise / scales.reshape(1, -1),
    )


def _validate_basis(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D matrix")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _harmonic_mean(left: float, right: float) -> float:
    total = left + right
    return 0.0 if total <= 0.0 else 2.0 * left * right / total


def _readonly(values: np.ndarray) -> np.ndarray:
    copy = np.asarray(values, dtype=np.float64).copy()
    copy.setflags(write=False)
    return copy


def _empty_rank_metrics(prefix: str) -> dict[str, Any]:
    return {
        f"{prefix}_estimated_rank": None,
        f"{prefix}_rank_signed_error": None,
        f"{prefix}_rank_absolute_error": None,
        f"{prefix}_rank_precision": None,
        f"{prefix}_rank_recall": None,
        f"{prefix}_rank_f1": None,
        f"{prefix}_rank_exact": None,
        f"{prefix}_rank_false_positive": None,
    }


def _empty_subspace_metrics() -> dict[str, Any]:
    return {
        "selected_subspace_estimated_rank": None,
        "selected_subspace_overlap": None,
        "selected_subspace_recall": None,
        "selected_subspace_precision": None,
        "selected_subspace_f1": None,
        "selected_subspace_missed_distance": None,
        "selected_subspace_mean_principal_angle_degrees": None,
        "selected_subspace_max_principal_angle_degrees": None,
        "selected_subspace_min_canonical_correlation": None,
    }
