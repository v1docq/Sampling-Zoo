from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np


class NullModelPolicy(str, Enum):
    """Reference distributions supported by the spectral diagnostic."""

    FEATURE_PERMUTATION = "feature_permutation"
    MOMENT_MATCHED_GAUSSIAN = "moment_matched_gaussian"
    VIEW_RESAMPLING = "view_resampling"


class NullDiagnosticStatus(str, Enum):
    DISABLED = "disabled"
    OK = "ok"
    PARTIAL = "partial"
    FAILED = "failed"


_POLICY_SEEDS = {
    NullModelPolicy.FEATURE_PERMUTATION: 11,
    NullModelPolicy.MOMENT_MATCHED_GAUSSIAN: 23,
    NullModelPolicy.VIEW_RESAMPLING: 37,
}


@dataclass(frozen=True)
class SpectralNullDiagnosticConfig:
    """Validated configuration for empirical spectral reference distributions."""

    enabled: bool = False
    policies: Tuple[NullModelPolicy, ...] = (
        NullModelPolicy.FEATURE_PERMUTATION,
        NullModelPolicy.MOMENT_MATCHED_GAUSSIAN,
        NullModelPolicy.VIEW_RESAMPLING,
    )
    n_resamples: int = 16
    quantile: float = 0.95
    min_selection_frequency: float = 0.80
    primary_policy: NullModelPolicy = NullModelPolicy.FEATURE_PERMUTATION
    random_state: Optional[int] = 42

    @classmethod
    def from_values(
        cls,
        *,
        enabled: bool,
        policies: Sequence[Union[str, NullModelPolicy]],
        n_resamples: int,
        quantile: float,
        min_selection_frequency: float,
        primary_policy: Union[str, NullModelPolicy],
        random_state: Optional[int],
    ) -> "SpectralNullDiagnosticConfig":
        raw_policies = (
            (policies,)
            if isinstance(policies, (str, NullModelPolicy))
            else policies
        )
        normalized_policies = tuple(cls._normalize_policy(value) for value in raw_policies)
        if not normalized_policies:
            raise ValueError("null_model_policies must contain at least one policy")
        if len(set(normalized_policies)) != len(normalized_policies):
            raise ValueError("null_model_policies must not contain duplicates")

        normalized_primary = cls._normalize_policy(primary_policy)
        if normalized_primary not in normalized_policies:
            raise ValueError("null_primary_policy must be included in null_model_policies")
        if normalized_primary is NullModelPolicy.VIEW_RESAMPLING:
            raise ValueError(
                "null_primary_policy must be feature_permutation or "
                "moment_matched_gaussian; view_resampling is a stability reference"
            )

        n_resamples = int(n_resamples)
        if n_resamples < 2:
            raise ValueError("null_resamples must be at least 2")
        quantile = float(quantile)
        if not 0.0 < quantile < 1.0:
            raise ValueError("null_quantile must be in (0, 1)")
        min_selection_frequency = float(min_selection_frequency)
        if not 0.0 < min_selection_frequency <= 1.0:
            raise ValueError("null_min_selection_frequency must be in (0, 1]")

        return cls(
            enabled=bool(enabled),
            policies=normalized_policies,
            n_resamples=n_resamples,
            quantile=quantile,
            min_selection_frequency=min_selection_frequency,
            primary_policy=normalized_primary,
            random_state=random_state,
        )

    @staticmethod
    def _normalize_policy(value: Union[str, NullModelPolicy]) -> NullModelPolicy:
        if isinstance(value, NullModelPolicy):
            return value
        try:
            return NullModelPolicy(str(value).strip().lower())
        except ValueError as exc:
            allowed = ", ".join(policy.value for policy in NullModelPolicy)
            raise ValueError(f"Unknown spectral null policy {value!r}. Expected one of: {allowed}") from exc

    @property
    def work_units(self) -> int:
        return len(self.policies) * self.n_resamples if self.enabled else 0


@dataclass(frozen=True)
class NullReplicateFailure:
    policy: NullModelPolicy
    replicate: int
    code: str
    message: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy": self.policy.value,
            "replicate": int(self.replicate),
            "code": self.code,
            "message": self.message,
        }


@dataclass(frozen=True)
class NullPolicyDiagnostic:
    policy: NullModelPolicy
    reference_kind: str
    status: NullDiagnosticStatus
    requested_resamples: int
    successful_resamples: int
    singular_value_quantiles: Tuple[float, ...]
    empirical_bulk_edge: Optional[float]
    outlier_count: Optional[int]
    stable_outlier_count: Optional[int]
    max_outlier_excess: Optional[float]
    outlier_excess: Tuple[float, ...]
    selection_frequency: Tuple[float, ...]
    empirical_p_values: Tuple[float, ...]
    failures: Tuple[NullReplicateFailure, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy": self.policy.value,
            "reference_kind": self.reference_kind,
            "status": self.status.value,
            "requested_resamples": int(self.requested_resamples),
            "successful_resamples": int(self.successful_resamples),
            "singular_value_quantiles": list(self.singular_value_quantiles),
            "empirical_bulk_edge": self.empirical_bulk_edge,
            "outlier_count": self.outlier_count,
            "stable_outlier_count": self.stable_outlier_count,
            "max_outlier_excess": self.max_outlier_excess,
            "outlier_excess": list(self.outlier_excess),
            "selection_frequency": list(self.selection_frequency),
            "empirical_p_values": list(self.empirical_p_values),
            "failures": [failure.to_dict() for failure in self.failures],
        }


@dataclass(frozen=True)
class SpectralNullDiagnosticResult:
    status: NullDiagnosticStatus
    primary_policy: NullModelPolicy
    observed_singular_values: Tuple[float, ...]
    policy_results: Tuple[NullPolicyDiagnostic, ...]

    @classmethod
    def disabled(
        cls,
        primary_policy: NullModelPolicy = NullModelPolicy.FEATURE_PERMUTATION,
    ) -> "SpectralNullDiagnosticResult":
        return cls(
            status=NullDiagnosticStatus.DISABLED,
            primary_policy=primary_policy,
            observed_singular_values=(),
            policy_results=(),
        )

    def policy_result(self, policy: NullModelPolicy) -> Optional[NullPolicyDiagnostic]:
        return next((result for result in self.policy_results if result.policy is policy), None)

    @property
    def primary_result(self) -> Optional[NullPolicyDiagnostic]:
        return self.policy_result(self.primary_policy)

    @property
    def empirical_bulk_edge(self) -> Optional[float]:
        result = self.primary_result
        return result.empirical_bulk_edge if result is not None else None

    @property
    def rank_by_null_edge(self) -> Optional[int]:
        result = self.primary_result
        return result.outlier_count if result is not None else None

    @property
    def rank_by_stability(self) -> Optional[int]:
        result = self.policy_result(NullModelPolicy.VIEW_RESAMPLING)
        return result.stable_outlier_count if result is not None else None

    @property
    def max_outlier_excess(self) -> Optional[float]:
        result = self.primary_result
        return result.max_outlier_excess if result is not None else None

    @property
    def successful_resamples(self) -> int:
        return sum(result.successful_resamples for result in self.policy_results)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "primary_policy": self.primary_policy.value,
            "observed_singular_values": list(self.observed_singular_values),
            "empirical_bulk_edge": self.empirical_bulk_edge,
            "rank_by_null_edge": self.rank_by_null_edge,
            "rank_by_stability": self.rank_by_stability,
            "max_outlier_excess": self.max_outlier_excess,
            "successful_resamples": int(self.successful_resamples),
            "policies": {
                result.policy.value: result.to_dict()
                for result in self.policy_results
            },
        }


SpectrumEvaluator = Callable[[np.ndarray, np.random.Generator, bool], np.ndarray]
ProgressCallback = Callable[[], None]


def feature_permutation_surrogate(
    X: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Independently permute rows within each encoded feature."""

    values = np.asarray(X)
    surrogate = np.empty_like(values)
    for column in range(values.shape[1]):
        surrogate[:, column] = values[rng.permutation(values.shape[0]), column]
    return surrogate


def moment_matched_gaussian_surrogate(
    X: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a Gaussian surrogate with the same column moments as X."""

    values = np.asarray(X, dtype=np.float64)
    means = np.mean(values, axis=0)
    scales = np.std(values, axis=0)
    surrogate = np.broadcast_to(means, values.shape).copy()
    variable = scales > 0.0
    if np.any(variable) and values.shape[0] > 1:
        noise = rng.normal(size=(values.shape[0], int(np.sum(variable))))
        noise -= np.mean(noise, axis=0, keepdims=True)
        noise_std = np.std(noise, axis=0, keepdims=True)
        noise /= np.maximum(noise_std, np.finfo(np.float64).eps)
        surrogate[:, variable] += noise * scales[variable]
    return surrogate


def _policy_status(
    successful_resamples: int,
    requested_resamples: int,
) -> NullDiagnosticStatus:
    if successful_resamples == 0:
        return NullDiagnosticStatus.FAILED
    if successful_resamples < requested_resamples:
        return NullDiagnosticStatus.PARTIAL
    return NullDiagnosticStatus.OK


def summarize_reference_spectra(
    *,
    observed_singular_values: np.ndarray,
    reference_spectra: Sequence[np.ndarray],
    policy: NullModelPolicy,
    config: SpectralNullDiagnosticConfig,
    failures: Sequence[NullReplicateFailure] = (),
    comparison_edge: Optional[float] = None,
) -> NullPolicyDiagnostic:
    """Reduce reference spectra into null-edge or stability diagnostics."""

    observed = np.asarray(observed_singular_values, dtype=np.float64)
    successful = len(reference_spectra)
    status = _policy_status(successful, config.n_resamples)
    reference_kind = (
        "stochastic_reference"
        if policy is NullModelPolicy.VIEW_RESAMPLING
        else "null_reference"
    )
    if successful == 0:
        return NullPolicyDiagnostic(
            policy=policy,
            reference_kind=reference_kind,
            status=status,
            requested_resamples=config.n_resamples,
            successful_resamples=0,
            singular_value_quantiles=(),
            empirical_bulk_edge=None,
            outlier_count=None,
            stable_outlier_count=None,
            max_outlier_excess=None,
            outlier_excess=(),
            selection_frequency=(),
            empirical_p_values=(),
            failures=tuple(failures),
        )

    width = min(
        observed.size,
        min(np.asarray(spectrum).size for spectrum in reference_spectra),
    )
    observed = observed[:width]
    spectra = np.vstack([
        np.asarray(spectrum, dtype=np.float64)[:width]
        for spectrum in reference_spectra
    ])
    component_quantiles = np.quantile(spectra, config.quantile, axis=0)

    if policy is NullModelPolicy.VIEW_RESAMPLING:
        edge = comparison_edge
        if edge is None:
            outlier_count = None
            stable_outlier_count = None
            excess = np.asarray([], dtype=np.float64)
            selection_frequency = np.asarray([], dtype=np.float64)
            p_values = np.asarray([], dtype=np.float64)
        else:
            excess = np.maximum(observed - edge, 0.0)
            outlier_count = int(np.sum(excess > 0.0))
            selection_frequency = np.mean(spectra > edge, axis=0)
            stable_outlier_count = int(
                np.sum(
                    (excess > 0.0)
                    & (selection_frequency >= config.min_selection_frequency)
                )
            )
            p_values = np.asarray([], dtype=np.float64)
    else:
        edge_samples = spectra[:, 0]
        edge = float(np.quantile(edge_samples, config.quantile))
        excess = np.maximum(observed - edge, 0.0)
        outlier_count = int(np.sum(excess > 0.0))
        selection_frequency = np.mean(
            observed.reshape(1, -1) > edge_samples.reshape(-1, 1),
            axis=0,
        )
        stable_outlier_count = int(
            np.sum(
                (excess > 0.0)
                & (selection_frequency >= config.min_selection_frequency)
            )
        )
        p_values = (
            1.0
            + np.sum(
                edge_samples.reshape(-1, 1) >= observed.reshape(1, -1),
                axis=0,
            )
        ) / (successful + 1.0)

    return NullPolicyDiagnostic(
        policy=policy,
        reference_kind=reference_kind,
        status=status,
        requested_resamples=config.n_resamples,
        successful_resamples=successful,
        singular_value_quantiles=tuple(map(float, component_quantiles)),
        empirical_bulk_edge=None if edge is None else float(edge),
        outlier_count=outlier_count,
        stable_outlier_count=stable_outlier_count,
        max_outlier_excess=float(np.max(excess)) if excess.size else None,
        outlier_excess=tuple(map(float, excess)),
        selection_frequency=tuple(map(float, selection_frequency)),
        empirical_p_values=tuple(map(float, p_values)),
        failures=tuple(failures),
    )


class SpectralNullDiagnostic:
    """Effect shell that evaluates spectra and returns immutable diagnostics."""

    def __init__(self, config: SpectralNullDiagnosticConfig) -> None:
        self.config = config

    def evaluate(
        self,
        encoded_features: np.ndarray,
        observed_singular_values: np.ndarray,
        spectrum_evaluator: SpectrumEvaluator,
        on_progress: Optional[ProgressCallback] = None,
    ) -> SpectralNullDiagnosticResult:
        if not self.config.enabled:
            return SpectralNullDiagnosticResult.disabled(self.config.primary_policy)

        spectra_by_policy, failures_by_policy = self._collect_reference_spectra(
            encoded_features=np.asarray(encoded_features),
            spectrum_evaluator=spectrum_evaluator,
            on_progress=on_progress,
        )
        policy_results = self._build_policy_results(
            observed_singular_values=observed_singular_values,
            spectra_by_policy=spectra_by_policy,
            failures_by_policy=failures_by_policy,
        )
        return SpectralNullDiagnosticResult(
            status=self._resolve_result_status(policy_results),
            primary_policy=self.config.primary_policy,
            observed_singular_values=tuple(
                map(float, np.asarray(observed_singular_values, dtype=np.float64))
            ),
            policy_results=policy_results,
        )

    def _collect_reference_spectra(
        self,
        *,
        encoded_features: np.ndarray,
        spectrum_evaluator: SpectrumEvaluator,
        on_progress: Optional[ProgressCallback],
    ) -> Tuple[
        Dict[NullModelPolicy, list[np.ndarray]],
        Dict[NullModelPolicy, list[NullReplicateFailure]],
    ]:
        spectra_by_policy: Dict[NullModelPolicy, list[np.ndarray]] = {
            policy: [] for policy in self.config.policies
        }
        failures_by_policy: Dict[NullModelPolicy, list[NullReplicateFailure]] = {
            policy: [] for policy in self.config.policies
        }

        for policy in self.config.policies:
            for replicate in range(self.config.n_resamples):
                rng = self._replicate_rng(policy, replicate)
                try:
                    reference_X, resample_views = self._reference_input(
                        encoded_features,
                        policy,
                        rng,
                    )
                    spectrum = np.asarray(
                        spectrum_evaluator(reference_X, rng, resample_views),
                        dtype=np.float64,
                    )
                    spectra_by_policy[policy].append(self._validate_spectrum(spectrum))
                except Exception as exc:
                    failures_by_policy[policy].append(
                        NullReplicateFailure(
                            policy=policy,
                            replicate=replicate,
                            code=type(exc).__name__,
                            message=str(exc),
                        )
                    )
                finally:
                    if on_progress is not None:
                        on_progress()
        return spectra_by_policy, failures_by_policy

    def _build_policy_results(
        self,
        *,
        observed_singular_values: np.ndarray,
        spectra_by_policy: Dict[NullModelPolicy, list[np.ndarray]],
        failures_by_policy: Dict[NullModelPolicy, list[NullReplicateFailure]],
    ) -> Tuple[NullPolicyDiagnostic, ...]:
        primary_result = summarize_reference_spectra(
            observed_singular_values=observed_singular_values,
            reference_spectra=spectra_by_policy[self.config.primary_policy],
            policy=self.config.primary_policy,
            config=self.config,
            failures=failures_by_policy[self.config.primary_policy],
        )
        primary_edge = primary_result.empirical_bulk_edge
        policy_results: list[NullPolicyDiagnostic] = []
        for policy in self.config.policies:
            if policy is self.config.primary_policy:
                result = primary_result
            else:
                result = summarize_reference_spectra(
                    observed_singular_values=observed_singular_values,
                    reference_spectra=spectra_by_policy[policy],
                    policy=policy,
                    config=self.config,
                    failures=failures_by_policy[policy],
                    comparison_edge=(
                        primary_edge
                        if policy is NullModelPolicy.VIEW_RESAMPLING
                        else None
                    ),
                )
            policy_results.append(result)
        return tuple(policy_results)

    def _resolve_result_status(
        self,
        policy_results: Tuple[NullPolicyDiagnostic, ...],
    ) -> NullDiagnosticStatus:
        primary_result = next(
            result
            for result in policy_results
            if result.policy is self.config.primary_policy
        )
        statuses = {result.status for result in policy_results}
        if primary_result.status is NullDiagnosticStatus.FAILED:
            return NullDiagnosticStatus.FAILED
        if statuses == {NullDiagnosticStatus.OK}:
            return NullDiagnosticStatus.OK
        return NullDiagnosticStatus.PARTIAL

    @staticmethod
    def _validate_spectrum(spectrum: np.ndarray) -> np.ndarray:
        if spectrum.ndim != 1 or spectrum.size == 0:
            raise ValueError("spectrum evaluator must return a non-empty 1D array")
        if not np.all(np.isfinite(spectrum)):
            raise ValueError("spectrum evaluator returned non-finite values")
        return spectrum

    def _replicate_rng(
        self,
        policy: NullModelPolicy,
        replicate: int,
    ) -> np.random.Generator:
        if self.config.random_state is None:
            seed = np.random.SeedSequence()
        else:
            seed = np.random.SeedSequence(
                [int(self.config.random_state), _POLICY_SEEDS[policy], int(replicate)]
            )
        return np.random.default_rng(seed)

    @staticmethod
    def _reference_input(
        X: np.ndarray,
        policy: NullModelPolicy,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, bool]:
        if policy is NullModelPolicy.FEATURE_PERMUTATION:
            return feature_permutation_surrogate(X, rng), False
        if policy is NullModelPolicy.MOMENT_MATCHED_GAUSSIAN:
            return moment_matched_gaussian_surrogate(X, rng), False
        return X, True
