from __future__ import annotations

import numpy as np

from sampling_zoo.core.sampling_strategies.spectral.null_diagnostics import (
    NullDiagnosticStatus,
    NullModelPolicy,
    SpectralNullDiagnostic,
    SpectralNullDiagnosticConfig,
    feature_permutation_surrogate,
    moment_matched_gaussian_surrogate,
    summarize_reference_spectra,
)


def _config(
    *,
    policies: tuple[str, ...] = (
        "feature_permutation",
        "moment_matched_gaussian",
        "view_resampling",
    ),
    n_resamples: int = 4,
) -> SpectralNullDiagnosticConfig:
    return SpectralNullDiagnosticConfig.from_values(
        enabled=True,
        policies=policies,
        n_resamples=n_resamples,
        quantile=0.95,
        min_selection_frequency=0.75,
        primary_policy="feature_permutation",
        random_state=42,
    )


def test_feature_permutation_preserves_each_column_multiset() -> None:
    X = np.arange(60, dtype=np.float64).reshape(12, 5)

    first = feature_permutation_surrogate(X, np.random.default_rng(7))
    second = feature_permutation_surrogate(X, np.random.default_rng(7))

    assert np.array_equal(first, second)
    for column in range(X.shape[1]):
        assert np.array_equal(np.sort(first[:, column]), np.sort(X[:, column]))


def test_moment_matched_gaussian_preserves_column_moments_and_constants() -> None:
    rng = np.random.default_rng(9)
    X = rng.normal(loc=[2.0, -3.0], scale=[4.0, 0.5], size=(200, 2))
    X = np.column_stack([X, np.full(X.shape[0], 11.0)])

    surrogate = moment_matched_gaussian_surrogate(X, np.random.default_rng(13))

    assert np.allclose(np.mean(surrogate, axis=0), np.mean(X, axis=0))
    assert np.allclose(np.std(surrogate, axis=0), np.std(X, axis=0))
    assert np.all(surrogate[:, 2] == 11.0)


def test_reference_summary_detects_stable_spikes_above_empirical_edge() -> None:
    config = _config(policies=("feature_permutation",), n_resamples=4)
    references = [
        np.asarray([3.0, 2.0, 1.0]),
        np.asarray([3.2, 2.1, 0.9]),
        np.asarray([2.8, 1.9, 1.1]),
        np.asarray([3.1, 2.0, 1.0]),
    ]

    result = summarize_reference_spectra(
        observed_singular_values=np.asarray([9.0, 5.0, 1.0]),
        reference_spectra=references,
        policy=NullModelPolicy.FEATURE_PERMUTATION,
        config=config,
    )

    assert result.status is NullDiagnosticStatus.OK
    assert result.empirical_bulk_edge is not None
    assert 3.0 < result.empirical_bulk_edge < 3.2
    assert result.outlier_count == 2
    assert result.stable_outlier_count == 2
    assert result.selection_frequency[:2] == (1.0, 1.0)


def test_diagnostic_is_deterministic_and_distinguishes_reference_kinds() -> None:
    X = np.random.default_rng(17).normal(size=(40, 5))
    observed = np.linalg.svd(X, compute_uv=False)[:3]
    config = _config(n_resamples=3)

    def evaluate(
        reference_X: np.ndarray,
        rng: np.random.Generator,
        resample_views: bool,
    ) -> np.ndarray:
        if resample_views:
            projection = rng.normal(size=(reference_X.shape[1], 4))
            reference_X = reference_X @ projection
        return np.linalg.svd(reference_X, compute_uv=False)[:3]

    first = SpectralNullDiagnostic(config).evaluate(X, observed, evaluate)
    second = SpectralNullDiagnostic(config).evaluate(X, observed, evaluate)

    assert first.to_dict() == second.to_dict()
    assert first.status is NullDiagnosticStatus.OK
    assert first.primary_result is not None
    assert first.primary_result.reference_kind == "null_reference"
    view_result = first.policy_result(NullModelPolicy.VIEW_RESAMPLING)
    assert view_result is not None
    assert view_result.reference_kind == "stochastic_reference"
    assert view_result.stable_outlier_count is not None


def test_replicate_failures_are_typed_and_do_not_abort_diagnostic() -> None:
    X = np.random.default_rng(21).normal(size=(24, 4))
    observed = np.linalg.svd(X, compute_uv=False)[:2]
    config = _config(policies=("feature_permutation",), n_resamples=3)
    calls = 0

    def flaky_evaluator(
        reference_X: np.ndarray,
        rng: np.random.Generator,
        resample_views: bool,
    ) -> np.ndarray:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("synthetic failure")
        return np.linalg.svd(reference_X, compute_uv=False)[:2]

    result = SpectralNullDiagnostic(config).evaluate(X, observed, flaky_evaluator)

    assert result.status is NullDiagnosticStatus.PARTIAL
    assert result.primary_result is not None
    assert result.primary_result.successful_resamples == 2
    assert result.primary_result.failures[0].code == "RuntimeError"
    assert result.primary_result.failures[0].message == "synthetic failure"
