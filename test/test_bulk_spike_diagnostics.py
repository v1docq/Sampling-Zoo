from __future__ import annotations

import numpy as np
import pandas as pd

from sampling_zoo.core.sampling_strategies.spectral.bulk_spike import (
    BulkSpikeBudgetPolicy,
    ComponentSplitStatus,
    build_bulk_spike_training_plan,
    build_spectral_component_split,
    compute_row_spectral_participation,
)
from sampling_zoo.core.sampling_strategies.spectral.null_diagnostics import (
    NullDiagnosticStatus,
    NullModelPolicy,
    SpectralNullDiagnosticConfig,
    SpectralNullDiagnosticResult,
    summarize_reference_spectra,
)
from sampling_zoo.core.sampling_strategies.spectral.rmt_contraction_sampler import (
    RMTContractionTensorSampler,
)


def _stable_split_result() -> SpectralNullDiagnosticResult:
    observed = np.asarray([9.0, 5.0, 1.0])
    config = SpectralNullDiagnosticConfig.from_values(
        enabled=True,
        policies=("feature_permutation", "view_resampling"),
        n_resamples=4,
        quantile=0.95,
        min_selection_frequency=0.75,
        primary_policy="feature_permutation",
        random_state=42,
    )
    primary = summarize_reference_spectra(
        observed_singular_values=observed,
        reference_spectra=(
            np.asarray([3.0, 2.0, 1.0]),
            np.asarray([3.2, 2.1, 0.9]),
            np.asarray([2.8, 1.9, 1.1]),
            np.asarray([3.1, 2.0, 1.0]),
        ),
        policy=NullModelPolicy.FEATURE_PERMUTATION,
        config=config,
    )
    view = summarize_reference_spectra(
        observed_singular_values=observed,
        reference_spectra=(
            np.asarray([8.8, 4.8, 1.0]),
            np.asarray([8.5, 4.6, 1.1]),
            np.asarray([8.7, 4.7, 0.9]),
            np.asarray([8.6, 4.5, 1.0]),
        ),
        policy=NullModelPolicy.VIEW_RESAMPLING,
        config=config,
        comparison_edge=primary.empirical_bulk_edge,
    )
    return SpectralNullDiagnosticResult(
        status=NullDiagnosticStatus.OK,
        primary_policy=NullModelPolicy.FEATURE_PERMUTATION,
        observed_singular_values=tuple(observed),
        policy_results=(primary, view),
    )


def test_component_split_uses_edge_and_stability_frequency() -> None:
    split = build_spectral_component_split(
        np.asarray([9.0, 5.0, 1.0]),
        _stable_split_result(),
        min_selection_frequency=0.75,
    )

    assert split.status is ComponentSplitStatus.OK
    assert split.spike_indices.tolist() == [0, 1]
    assert split.bulk_mask.tolist() == [False, False, True]
    assert split.frequency_source == "view_resampling"


def test_row_participation_separates_localized_spike_energy() -> None:
    split = build_spectral_component_split(
        np.asarray([9.0, 5.0, 1.0]),
        _stable_split_result(),
        min_selection_frequency=0.75,
    )
    basis = np.asarray(
        [
            [0.70, 0.05, 0.05],
            [0.65, 0.10, 0.05],
            [0.05, 0.10, 0.70],
            [0.05, 0.05, 0.65],
        ]
    )

    participation = compute_row_spectral_participation(basis, split)

    assert participation.signalness[:2].min() > 0.95
    assert participation.signalness[2:].max() < 0.80
    assert np.allclose(
        participation.spike_signatures[
            participation.spike_energy > 0.0
        ].sum(axis=1),
        1.0,
    )


def test_row_participation_zeroes_numerically_negligible_spike_energy() -> None:
    split = build_spectral_component_split(
        np.asarray([9.0, 5.0, 1.0]),
        _stable_split_result(),
        min_selection_frequency=0.75,
    )
    basis = np.asarray(
        [
            [1e-12, 1e-12, 0.5],
            [0.5, 0.25, 0.1],
        ]
    )

    participation = compute_row_spectral_participation(
        basis,
        split,
        epsilon=1e-12,
    )

    assert participation.spike_energy[0] == 0.0
    assert np.all(participation.spike_signatures[0] == 0.0)
    assert np.isclose(participation.spike_signatures[1].sum(), 1.0)


def test_bulk_spike_plan_preserves_exact_unique_budget() -> None:
    split = build_spectral_component_split(
        np.asarray([9.0, 5.0, 1.0]),
        _stable_split_result(),
        min_selection_frequency=0.75,
    )
    rng = np.random.default_rng(17)
    basis = rng.normal(scale=0.05, size=(100, 3))
    basis[:20, 0] += 0.8
    basis[20:, 2] += 0.2
    participation = compute_row_spectral_participation(basis, split)

    first = build_bulk_spike_training_plan(
        participation,
        total_budget=25,
        policy=BulkSpikeBudgetPolicy.EXCESS_ENERGY,
        random_state=7,
    )
    second = build_bulk_spike_training_plan(
        participation,
        total_budget=25,
        policy=BulkSpikeBudgetPolicy.EXCESS_ENERGY,
        random_state=7,
    )

    assert first.selected_indices.size == 25
    assert np.unique(first.selected_indices).size == 25
    assert np.array_equal(first.selected_indices, second.selected_indices)
    assert first.spike_selected_indices.size > 0
    assert first.bulk_selected_indices.size > 0


def test_sampler_materializes_component_diagnostics_when_enabled() -> None:
    rng = np.random.default_rng(29)
    X = pd.DataFrame(rng.normal(size=(80, 6)))
    X.iloc[:12, 0] += 8.0
    sampler = RMTContractionTensorSampler(
        n_partitions=2,
        n_views=4,
        initial_rank_fraction=0.5,
        null_diagnostic_enabled=True,
        null_model_policies=("feature_permutation", "view_resampling"),
        null_resamples=2,
        null_min_selection_frequency=0.5,
        component_diagnostic_enabled=True,
        backend="numpy",
        show_progress=False,
        random_state=31,
    )

    sampler.fit(X)

    assert sampler.initial_left_basis_ is not None
    assert sampler.row_spectral_participation_ is not None
    assert sampler.row_spectral_participation_.n_rows == len(X)
    assert "spectral_component_split" in sampler.diagnostics_
    assert "row_spectral_participation" in sampler.diagnostics_
