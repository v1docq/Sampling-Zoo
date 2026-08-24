from __future__ import annotations

import numpy as np

from sampling_zoo.core.validation.leverage_shap import (
    CoalitionSamplingPolicy,
    build_coalition_sampling_plan,
    estimate_projected_shapley,
    exact_shapley_values,
)


def _interaction_game(coalitions: np.ndarray) -> np.ndarray:
    values = coalitions.astype(float)
    return (
        values @ np.asarray([1.0, -2.0, 0.5, 3.0, -1.5, 0.25])
        + 4.0 * values[:, 0] * values[:, 3]
        - 2.0 * values[:, 1] * values[:, 4]
    )


def test_exact_shapley_splits_pairwise_interactions_equally() -> None:
    values = exact_shapley_values(_interaction_game, 6)

    assert np.allclose(
        values,
        np.asarray([3.0, -3.0, 0.5, 5.0, -2.5, 0.25]),
    )


def test_coalition_plans_are_exact_unique_paired_and_deterministic() -> None:
    for policy in CoalitionSamplingPolicy:
        first = build_coalition_sampling_plan(
            8,
            budget=34,
            policy=policy,
            random_state=17,
        )
        second = build_coalition_sampling_plan(
            8,
            budget=34,
            policy=policy,
            random_state=17,
        )
        assert first.effective_budget == 34
        assert first.coalition_matrix.shape == (32, 8)
        assert np.unique(first.coalition_matrix, axis=0).shape[0] == 32
        assert np.array_equal(first.coalition_matrix, second.coalition_matrix)


def test_projected_regression_enforces_efficiency() -> None:
    estimate = estimate_projected_shapley(
        _interaction_game,
        6,
        budget=24,
        policy="leverage",
        random_state=29,
    )

    assert estimate.model_evaluations == 24
    assert estimate.efficiency_residual < 1e-9
    assert np.isclose(
        np.sum(estimate.values),
        np.sum(exact_shapley_values(_interaction_game, 6)),
    )


def test_full_budget_recovers_exact_values_for_both_policies() -> None:
    expected = exact_shapley_values(_interaction_game, 6)
    for policy in CoalitionSamplingPolicy:
        estimate = estimate_projected_shapley(
            _interaction_game,
            6,
            budget=2**6,
            policy=policy,
            random_state=7,
        )
        assert np.allclose(estimate.values, expected, atol=1e-9)
