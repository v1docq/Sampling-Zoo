from __future__ import annotations

import numpy as np
import pytest

from sampling_zoo.core.sampling_strategies.spectral.backend.matrix_backend import (
    MatrixRMTBackend,
)
from sampling_zoo.core.sampling_strategies.spectral.classification_sampling import (
    select_class_aware_partition_indices,
)
from sampling_zoo.core.sampling_strategies.spectral.leverage_sketch import (
    build_exact_budget_sketch_plan,
    dependent_round_exact,
    evaluate_subspace_preservation,
    saturated_inclusion_probabilities,
)


def test_saturated_probabilities_preserve_exact_budget_and_high_leverage_rows() -> None:
    probabilities = saturated_inclusion_probabilities(
        np.asarray([100.0, 8.0, 4.0, 2.0, 1.0, 1.0]),
        target_size=3,
    )

    assert probabilities.sum() == pytest.approx(3.0)
    assert np.all((0.0 <= probabilities) & (probabilities <= 1.0))
    assert probabilities[0] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "policy",
    [
        "uniform",
        "capped_leverage",
        "saturated_leverage",
        "robust_leverage_mixture",
    ],
)
def test_exact_budget_plans_are_unique_and_reproducible(policy: str) -> None:
    scores = np.linspace(0.1, 2.0, 20)
    first = build_exact_budget_sketch_plan(
        np.arange(20),
        target_size=7,
        policy=policy,
        scores=scores,
        random_state=17,
        reweighting="inverse_probability",
    )
    second = build_exact_budget_sketch_plan(
        np.arange(20),
        target_size=7,
        policy=policy,
        scores=scores,
        random_state=17,
        reweighting="inverse_probability",
    )

    assert first.exact_budget
    assert np.array_equal(first.selected_indices, second.selected_indices)
    assert np.unique(first.selected_indices).size == 7
    assert first.inclusion.probabilities.sum() == pytest.approx(7.0)
    assert np.mean(first.training_weights) == pytest.approx(1.0)


def test_dependent_rounding_matches_first_order_marginals_empirically() -> None:
    probabilities = np.asarray([1.0, 0.8, 0.6, 0.4, 0.2, 0.0])
    counts = np.zeros(probabilities.size, dtype=float)
    repetitions = 4000
    for seed in range(repetitions):
        counts[dependent_round_exact(probabilities, random_state=seed)] += 1.0

    assert np.allclose(counts / repetitions, probabilities, atol=0.025)


def test_contract_arrays_are_read_only() -> None:
    plan = build_exact_budget_sketch_plan(
        np.arange(8),
        target_size=3,
        policy="uniform",
        scores=np.ones(8),
        random_state=2,
    )

    with pytest.raises(ValueError):
        plan.inclusion.probabilities[0] = 0.0
    with pytest.raises(ValueError):
        plan.selected_indices[0] = 4


def test_full_budget_has_zero_subspace_distortion() -> None:
    rng = np.random.default_rng(4)
    matrix = rng.normal(size=(30, 6))
    plan = build_exact_budget_sketch_plan(
        np.arange(matrix.shape[0]),
        target_size=matrix.shape[0],
        policy="saturated_leverage",
        scores=np.sum(matrix * matrix, axis=1),
        random_state=3,
        reweighting="inverse_probability",
    )
    diagnostic = evaluate_subspace_preservation(matrix, plan, rank=4)

    assert diagnostic.gram_relative_error < 1e-12
    assert diagnostic.projection_cost_relative_error < 1e-10
    assert diagnostic.max_principal_angle_degrees < 1e-5


def test_class_aware_sketch_preserves_exact_budget_and_weights() -> None:
    target = np.repeat(np.asarray([0, 1, 2]), [10, 8, 6])
    embedding = np.column_stack(
        [np.arange(target.size, dtype=float), np.ones(target.size)]
    )
    plan = select_class_aware_partition_indices(
        np.arange(target.size),
        target=target,
        target_size=12,
        min_samples_per_class=2,
        class_allocation_policy="proportional",
        selection_method="robust_leverage_mixture",
        scores=np.linspace(0.1, 1.0, target.size),
        embedding=embedding,
        random_state=7,
        training_reweighting="inverse_probability",
    )

    assert plan.feasible
    assert plan.selected_indices.size == 12
    assert plan.training_weights.size == 12
    assert len(plan.selected_class_counts) == 3
    assert all(count >= 2 for _label, count in plan.selected_class_counts)


def test_matrix_ridge_leverage_is_finite_and_has_effective_dimension() -> None:
    rng = np.random.default_rng(9)
    matrix = rng.normal(size=(40, 5))
    result = MatrixRMTBackend.compute_ridge_leverage_scores(matrix, "auto")

    assert result.scores.shape == (40,)
    assert np.all(np.isfinite(result.scores))
    assert np.all(result.scores >= 0.0)
    assert 0.0 < result.effective_dimension <= 5.0
    assert result.ridge_lambda > 0.0


def test_tensor_ridge_leverage_matches_matrix_backend_when_torch_is_available() -> None:
    torch = pytest.importorskip("torch")
    from sampling_zoo.core.sampling_strategies.spectral.backend.tensor_backend import (
        TensorRMTBackend,
    )

    rng = np.random.default_rng(11)
    matrix = rng.normal(size=(25, 4))
    ridge_lambda = 0.2
    matrix_result = MatrixRMTBackend.compute_ridge_leverage_scores(
        matrix,
        ridge_lambda,
    )
    tensor_result = TensorRMTBackend(
        device="cpu",
        dtype="float64",
    ).compute_ridge_leverage_scores(matrix, ridge_lambda)

    assert torch is not None
    assert np.allclose(tensor_result.scores, matrix_result.scores, atol=1e-9)
    assert tensor_result.effective_dimension == pytest.approx(
        matrix_result.effective_dimension,
        abs=1e-9,
    )
