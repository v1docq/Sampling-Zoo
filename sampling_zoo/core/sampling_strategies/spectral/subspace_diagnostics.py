from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Optional, Protocol, Sequence, Tuple

import numpy as np


class SubspaceDiagnosticStatus(str, Enum):
    DISABLED = "disabled"
    OK = "ok"
    PARTIAL = "partial"
    FAILED = "failed"


@dataclass(frozen=True)
class SpectralSubspaceDiagnosticConfig:
    """Validated thresholds for view-resampled subspace stability."""

    enabled: bool = False
    n_resamples: int = 16
    quantile: float = 0.90
    max_principal_angle_degrees: float = 15.0
    max_normalized_projection_distance: float = 0.25
    max_rank: Optional[int] = 64
    random_state: Optional[int] = 42

    @classmethod
    def from_values(
        cls,
        *,
        enabled: bool,
        n_resamples: int,
        quantile: float,
        max_principal_angle_degrees: float,
        max_normalized_projection_distance: float,
        max_rank: Optional[int],
        random_state: Optional[int],
    ) -> "SpectralSubspaceDiagnosticConfig":
        n_resamples = int(n_resamples)
        if n_resamples < 2:
            raise ValueError("subspace_resamples must be at least 2")
        quantile = float(quantile)
        if not 0.0 < quantile < 1.0:
            raise ValueError("subspace_quantile must be in (0, 1)")
        angle = float(max_principal_angle_degrees)
        if not 0.0 <= angle <= 90.0:
            raise ValueError(
                "subspace_max_principal_angle_degrees must be in [0, 90]"
            )
        distance = float(max_normalized_projection_distance)
        if not 0.0 <= distance <= 1.0:
            raise ValueError(
                "subspace_max_normalized_projection_distance must be in [0, 1]"
            )
        if max_rank is not None:
            max_rank = int(max_rank)
            if max_rank < 1:
                raise ValueError("subspace_max_rank must be positive or None")
        return cls(
            enabled=bool(enabled),
            n_resamples=n_resamples,
            quantile=quantile,
            max_principal_angle_degrees=angle,
            max_normalized_projection_distance=distance,
            max_rank=max_rank,
            random_state=random_state,
        )

    @property
    def work_units(self) -> int:
        return self.n_resamples if self.enabled else 0


class SubspaceComparisonLike(Protocol):
    rank: int
    principal_angles_degrees: Tuple[float, ...]
    mean_principal_angle_degrees: float
    max_principal_angle_degrees: float
    projection_distance: float
    normalized_projection_distance: float
    min_canonical_correlation: float


@dataclass(frozen=True)
class SubspaceReplicateFailure:
    replicate: int
    code: str
    message: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "replicate": int(self.replicate),
            "code": self.code,
            "message": self.message,
        }


@dataclass(frozen=True)
class SubspaceReplicateDiagnostic:
    replicate: int
    comparison_rank: int
    principal_angles_degrees: Tuple[float, ...]
    mean_principal_angle_degrees: float
    max_principal_angle_degrees: float
    projection_distance: float
    normalized_projection_distance: float
    min_canonical_correlation: float

    @classmethod
    def from_comparison(
        cls,
        replicate: int,
        comparison: SubspaceComparisonLike,
    ) -> "SubspaceReplicateDiagnostic":
        return cls(
            replicate=int(replicate),
            comparison_rank=int(comparison.rank),
            principal_angles_degrees=tuple(
                map(float, comparison.principal_angles_degrees)
            ),
            mean_principal_angle_degrees=float(
                comparison.mean_principal_angle_degrees
            ),
            max_principal_angle_degrees=float(
                comparison.max_principal_angle_degrees
            ),
            projection_distance=float(comparison.projection_distance),
            normalized_projection_distance=float(
                comparison.normalized_projection_distance
            ),
            min_canonical_correlation=float(
                comparison.min_canonical_correlation
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "replicate": int(self.replicate),
            "comparison_rank": int(self.comparison_rank),
            "principal_angles_degrees": list(self.principal_angles_degrees),
            "mean_principal_angle_degrees": self.mean_principal_angle_degrees,
            "max_principal_angle_degrees": self.max_principal_angle_degrees,
            "projection_distance": self.projection_distance,
            "normalized_projection_distance": (
                self.normalized_projection_distance
            ),
            "min_canonical_correlation": self.min_canonical_correlation,
        }


@dataclass(frozen=True)
class SubspaceRankDiagnostic:
    rank: int
    max_angle_mean_degrees: float
    max_angle_quantile_degrees: float
    normalized_projection_distance_mean: float
    normalized_projection_distance_quantile: float
    min_canonical_correlation_mean: float
    stability_frequency: float
    stable: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rank": int(self.rank),
            "max_angle_mean_degrees": self.max_angle_mean_degrees,
            "max_angle_quantile_degrees": self.max_angle_quantile_degrees,
            "normalized_projection_distance_mean": (
                self.normalized_projection_distance_mean
            ),
            "normalized_projection_distance_quantile": (
                self.normalized_projection_distance_quantile
            ),
            "min_canonical_correlation_mean": (
                self.min_canonical_correlation_mean
            ),
            "stability_frequency": self.stability_frequency,
            "stable": bool(self.stable),
        }


@dataclass(frozen=True)
class SpectralSubspaceDiagnosticResult:
    status: SubspaceDiagnosticStatus
    comparison_rank: int
    rank_source: str
    requested_resamples: int
    successful_resamples: int
    rank_diagnostics: Tuple[SubspaceRankDiagnostic, ...]
    replicate_diagnostics: Tuple[SubspaceReplicateDiagnostic, ...]
    failures: Tuple[SubspaceReplicateFailure, ...]

    @classmethod
    def disabled(cls) -> "SpectralSubspaceDiagnosticResult":
        return cls(
            status=SubspaceDiagnosticStatus.DISABLED,
            comparison_rank=0,
            rank_source="selected_rank",
            requested_resamples=0,
            successful_resamples=0,
            rank_diagnostics=(),
            replicate_diagnostics=(),
            failures=(),
        )

    @property
    def rank_by_subspace_stability(self) -> Optional[int]:
        if self.status in {
            SubspaceDiagnosticStatus.DISABLED,
            SubspaceDiagnosticStatus.FAILED,
        }:
            return None
        stable_ranks = [
            diagnostic.rank
            for diagnostic in self.rank_diagnostics
            if diagnostic.stable
        ]
        return max(stable_ranks, default=0)

    @property
    def comparison_rank_diagnostic(self) -> Optional[SubspaceRankDiagnostic]:
        return next(
            (
                diagnostic
                for diagnostic in self.rank_diagnostics
                if diagnostic.rank == self.comparison_rank
            ),
            None,
        )

    def to_dict(self) -> Dict[str, Any]:
        full_rank = self.comparison_rank_diagnostic
        return {
            "status": self.status.value,
            "comparison_rank": int(self.comparison_rank),
            "rank_source": self.rank_source,
            "requested_resamples": int(self.requested_resamples),
            "successful_resamples": int(self.successful_resamples),
            "rank_by_subspace_stability": self.rank_by_subspace_stability,
            "max_angle_quantile_degrees": (
                full_rank.max_angle_quantile_degrees
                if full_rank is not None
                else None
            ),
            "normalized_projection_distance_quantile": (
                full_rank.normalized_projection_distance_quantile
                if full_rank is not None
                else None
            ),
            "stability_frequency": (
                full_rank.stability_frequency
                if full_rank is not None
                else None
            ),
            "rank_diagnostics": [
                diagnostic.to_dict()
                for diagnostic in self.rank_diagnostics
            ],
            "replicate_diagnostics": [
                diagnostic.to_dict()
                for diagnostic in self.replicate_diagnostics
            ],
            "failures": [failure.to_dict() for failure in self.failures],
        }


BasisEvaluator = Callable[[np.random.Generator], np.ndarray]
SubspaceComparator = Callable[
    [np.ndarray, np.ndarray, int],
    Sequence[SubspaceComparisonLike],
]
ProgressCallback = Callable[[], None]


class SpectralSubspaceDiagnostic:
    """Thin effect shell over view resampling and pure stability summaries."""

    def __init__(self, config: SpectralSubspaceDiagnosticConfig) -> None:
        self.config = config

    def evaluate(
        self,
        *,
        reference_basis: np.ndarray,
        basis_evaluator: BasisEvaluator,
        subspace_comparator: SubspaceComparator,
        on_progress: Optional[ProgressCallback] = None,
    ) -> SpectralSubspaceDiagnosticResult:
        if not self.config.enabled:
            return SpectralSubspaceDiagnosticResult.disabled()

        reference = self._validate_basis(reference_basis, "reference_basis")
        available_rank = int(reference.shape[1])
        comparison_rank = self.resolve_comparison_rank(available_rank)
        reference = reference[:, :comparison_rank]
        curves, replicates, failures = self._evaluate_resamples(
            reference_basis=reference,
            comparison_rank=comparison_rank,
            basis_evaluator=basis_evaluator,
            subspace_comparator=subspace_comparator,
            on_progress=on_progress,
        )
        rank_diagnostics = summarize_subspace_curves(
            curves=curves,
            comparison_rank=comparison_rank,
            config=self.config,
        )
        return SpectralSubspaceDiagnosticResult(
            status=self._result_status(len(curves)),
            comparison_rank=comparison_rank,
            rank_source=(
                "selected_rank"
                if comparison_rank == available_rank
                else "selected_rank_capped"
            ),
            requested_resamples=self.config.n_resamples,
            successful_resamples=len(curves),
            rank_diagnostics=rank_diagnostics,
            replicate_diagnostics=tuple(replicates),
            failures=tuple(failures),
        )

    def resolve_comparison_rank(self, available_rank: int) -> int:
        available_rank = int(available_rank)
        if available_rank < 1:
            raise ValueError("available_rank must be positive")
        if self.config.max_rank is None:
            return available_rank
        return min(available_rank, self.config.max_rank)

    def _evaluate_resamples(
        self,
        *,
        reference_basis: np.ndarray,
        comparison_rank: int,
        basis_evaluator: BasisEvaluator,
        subspace_comparator: SubspaceComparator,
        on_progress: Optional[ProgressCallback],
    ) -> Tuple[
        list[Tuple[SubspaceComparisonLike, ...]],
        list[SubspaceReplicateDiagnostic],
        list[SubspaceReplicateFailure],
    ]:
        curves: list[Tuple[SubspaceComparisonLike, ...]] = []
        replicates: list[SubspaceReplicateDiagnostic] = []
        failures: list[SubspaceReplicateFailure] = []
        for replicate in range(self.config.n_resamples):
            rng = self._replicate_rng(replicate)
            try:
                candidate = self._validate_basis(
                    basis_evaluator(rng),
                    "candidate_basis",
                )
                curve = tuple(
                    subspace_comparator(
                        reference_basis,
                        candidate,
                        comparison_rank,
                    )
                )
                self._validate_comparison_curve(curve, comparison_rank)
                curves.append(curve)
                replicates.append(
                    SubspaceReplicateDiagnostic.from_comparison(
                        replicate,
                        curve[-1],
                    )
                )
            except Exception as exc:
                failures.append(
                    SubspaceReplicateFailure(
                        replicate=replicate,
                        code=type(exc).__name__,
                        message=str(exc),
                    )
                )
            finally:
                if on_progress is not None:
                    on_progress()
        return curves, replicates, failures

    def _result_status(self, successful_resamples: int) -> SubspaceDiagnosticStatus:
        if successful_resamples == 0:
            return SubspaceDiagnosticStatus.FAILED
        if successful_resamples < self.config.n_resamples:
            return SubspaceDiagnosticStatus.PARTIAL
        return SubspaceDiagnosticStatus.OK

    def _replicate_rng(self, replicate: int) -> np.random.Generator:
        if self.config.random_state is None:
            seed = np.random.SeedSequence()
        else:
            seed = np.random.SeedSequence(
                [int(self.config.random_state), 71, int(replicate)]
            )
        return np.random.default_rng(seed)

    @staticmethod
    def _validate_basis(basis: np.ndarray, name: str) -> np.ndarray:
        values = np.asarray(basis)
        if values.ndim != 2 or min(values.shape) < 1:
            raise ValueError(f"{name} must be a non-empty 2D matrix")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must contain only finite values")
        return values

    @staticmethod
    def _validate_comparison_curve(
        curve: Tuple[SubspaceComparisonLike, ...],
        comparison_rank: int,
    ) -> None:
        expected_ranks = tuple(range(1, comparison_rank + 1))
        actual_ranks = tuple(int(comparison.rank) for comparison in curve)
        if actual_ranks != expected_ranks:
            raise ValueError(
                "subspace comparator must return ordered ranks 1..comparison_rank"
            )
        metrics = [
            metric
            for comparison in curve
            for metric in (
                comparison.max_principal_angle_degrees,
                comparison.normalized_projection_distance,
                comparison.min_canonical_correlation,
            )
        ]
        if not np.all(np.isfinite(metrics)):
            raise ValueError("subspace comparator returned non-finite metrics")


def summarize_subspace_curves(
    *,
    curves: Sequence[Sequence[SubspaceComparisonLike]],
    comparison_rank: int,
    config: SpectralSubspaceDiagnosticConfig,
) -> Tuple[SubspaceRankDiagnostic, ...]:
    """Aggregate per-resample prefix comparisons into rank-level diagnostics."""

    if not curves:
        return ()
    summaries = []
    for rank_index in range(comparison_rank):
        comparisons = [curve[rank_index] for curve in curves]
        max_angles = np.asarray(
            [comparison.max_principal_angle_degrees for comparison in comparisons],
            dtype=np.float64,
        )
        distances = np.asarray(
            [
                comparison.normalized_projection_distance
                for comparison in comparisons
            ],
            dtype=np.float64,
        )
        correlations = np.asarray(
            [comparison.min_canonical_correlation for comparison in comparisons],
            dtype=np.float64,
        )
        angle_quantile = float(np.quantile(max_angles, config.quantile))
        distance_quantile = float(np.quantile(distances, config.quantile))
        stable_mask = (
            (max_angles <= config.max_principal_angle_degrees)
            & (
                distances
                <= config.max_normalized_projection_distance
            )
        )
        summaries.append(
            SubspaceRankDiagnostic(
                rank=rank_index + 1,
                max_angle_mean_degrees=float(np.mean(max_angles)),
                max_angle_quantile_degrees=angle_quantile,
                normalized_projection_distance_mean=float(np.mean(distances)),
                normalized_projection_distance_quantile=distance_quantile,
                min_canonical_correlation_mean=float(np.mean(correlations)),
                stability_frequency=float(np.mean(stable_mask)),
                stable=bool(
                    angle_quantile
                    <= config.max_principal_angle_degrees
                    and distance_quantile
                    <= config.max_normalized_projection_distance
                ),
            )
        )
    return tuple(summaries)
