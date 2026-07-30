from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from sampling_zoo.core.sampling_strategies.spectral.backend.matrix_backend import (
    MatrixRMTBackend,
)
from sampling_zoo.core.sampling_strategies.spectral.backend.tensor_backend import (
    TensorRMTBackend,
)
from sampling_zoo.core.sampling_strategies.spectral.subspace_diagnostics import (
    SpectralSubspaceDiagnostic,
    SpectralSubspaceDiagnosticConfig,
    SubspaceDiagnosticStatus,
)


def _basis_pair(theta: float = 0.6) -> tuple[np.ndarray, np.ndarray]:
    reference = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ]
    )
    rotation = np.asarray(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ]
    )
    return reference, reference @ rotation


def _config(n_resamples: int = 4) -> SpectralSubspaceDiagnosticConfig:
    return SpectralSubspaceDiagnosticConfig.from_values(
        enabled=True,
        n_resamples=n_resamples,
        quantile=0.90,
        max_principal_angle_degrees=5.0,
        max_normalized_projection_distance=0.05,
        max_rank=64,
        random_state=42,
    )


def test_matrix_backend_is_invariant_to_internal_basis_rotation() -> None:
    reference, rotated = _basis_pair()

    curve = MatrixRMTBackend.compare_subspace_prefixes(
        reference,
        rotated,
        max_rank=2,
    )

    assert curve[0].max_principal_angle_degrees > 30.0
    assert curve[0].normalized_projection_distance > 0.5
    assert curve[1].max_principal_angle_degrees == pytest.approx(0.0, abs=1e-6)
    assert curve[1].normalized_projection_distance == pytest.approx(0.0, abs=1e-6)
    assert curve[1].min_canonical_correlation == pytest.approx(1.0)


def test_matrix_backend_reports_unit_distance_for_orthogonal_subspaces() -> None:
    reference = np.eye(4)[:, :2]
    orthogonal = np.eye(4)[:, 2:]

    comparison = MatrixRMTBackend.compare_subspace_prefixes(
        reference,
        orthogonal,
        max_rank=2,
    )[-1]

    assert comparison.principal_angles_degrees == pytest.approx((90.0, 90.0))
    assert comparison.normalized_projection_distance == pytest.approx(1.0)
    assert comparison.min_canonical_correlation == pytest.approx(0.0)


def test_diagnostic_recovers_stable_span_despite_unstable_first_vector() -> None:
    reference, rotated = _basis_pair()
    diagnostic = SpectralSubspaceDiagnostic(_config(n_resamples=3))

    result = diagnostic.evaluate(
        reference_basis=reference,
        basis_evaluator=lambda rng: rotated,
        subspace_comparator=MatrixRMTBackend.compare_subspace_prefixes,
    )

    assert result.status is SubspaceDiagnosticStatus.OK
    assert result.successful_resamples == 3
    assert result.rank_diagnostics[0].stable is False
    assert result.rank_diagnostics[1].stable is True
    assert result.rank_by_subspace_stability == 2
    assert result.comparison_rank_diagnostic is not None
    assert result.comparison_rank_diagnostic.stability_frequency == 1.0


def test_diagnostic_is_deterministic_and_keeps_typed_partial_failures() -> None:
    reference, _ = _basis_pair()

    def run_once():
        calls = 0

        def evaluator(rng: np.random.Generator) -> np.ndarray:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("synthetic basis failure")
            return reference

        return SpectralSubspaceDiagnostic(_config(n_resamples=3)).evaluate(
            reference_basis=reference,
            basis_evaluator=evaluator,
            subspace_comparator=MatrixRMTBackend.compare_subspace_prefixes,
        )

    first = run_once()
    second = run_once()

    assert first.to_dict() == second.to_dict()
    assert first.status is SubspaceDiagnosticStatus.PARTIAL
    assert first.successful_resamples == 2
    assert first.rank_by_subspace_stability == 2
    assert first.failures[0].code == "RuntimeError"
    assert first.failures[0].message == "synthetic basis failure"


def test_diagnostic_caps_only_the_comparison_rank() -> None:
    reference = np.eye(6)
    config = SpectralSubspaceDiagnosticConfig.from_values(
        enabled=True,
        n_resamples=2,
        quantile=0.90,
        max_principal_angle_degrees=5.0,
        max_normalized_projection_distance=0.05,
        max_rank=3,
        random_state=42,
    )

    result = SpectralSubspaceDiagnostic(config).evaluate(
        reference_basis=reference,
        basis_evaluator=lambda rng: reference[:, :3],
        subspace_comparator=MatrixRMTBackend.compare_subspace_prefixes,
    )

    assert result.comparison_rank == 3
    assert result.rank_source == "selected_rank_capped"
    assert result.rank_by_subspace_stability == 3


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_tensor_backend_matches_matrix_subspace_metrics() -> None:
    reference, rotated = _basis_pair(theta=0.4)
    matrix_curve = MatrixRMTBackend.compare_subspace_prefixes(
        reference,
        rotated,
        max_rank=2,
    )
    tensor_curve = TensorRMTBackend(
        device="cpu",
        dtype="float64",
        random_state=42,
    ).compare_subspace_prefixes(
        reference,
        rotated,
        max_rank=2,
    )

    for matrix_result, tensor_result in zip(matrix_curve, tensor_curve):
        assert tensor_result.principal_angles_degrees == pytest.approx(
            matrix_result.principal_angles_degrees,
            abs=1e-6,
        )
        assert tensor_result.normalized_projection_distance == pytest.approx(
            matrix_result.normalized_projection_distance,
            abs=1e-6,
        )
