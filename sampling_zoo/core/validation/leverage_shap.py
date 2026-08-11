"""Projected Shapley regression with exact-budget coalition sampling."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import itertools
import math
from typing import Any, Callable, Sequence

import numpy as np

from sampling_zoo.core.sampling_strategies.spectral.leverage_sketch import (
    dependent_round_exact,
    saturated_inclusion_probabilities,
)


CoalitionGame = Callable[[np.ndarray], np.ndarray]


class CoalitionSamplingPolicy(str, Enum):
    """Sampling distributions compared in the explanation experiment."""

    KERNEL_WEIGHT = "kernel_weight"
    LEVERAGE = "leverage"


def _readonly(values: Any, *, dtype: Any, ndim: int) -> np.ndarray:
    array = np.asarray(values, dtype=dtype).copy()
    if array.ndim != ndim:
        raise ValueError(f"expected an array with ndim={ndim}")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class CoalitionSamplingPlan:
    """Paired, without-replacement coalition plan with first-order marginals."""

    policy: CoalitionSamplingPolicy
    n_players: int
    requested_budget: int
    effective_budget: int
    coalition_matrix: np.ndarray
    inclusion_probabilities: np.ndarray
    regression_weights: np.ndarray

    def __post_init__(self) -> None:
        coalitions = _readonly(self.coalition_matrix, dtype=bool, ndim=2)
        inclusion = _readonly(self.inclusion_probabilities, dtype=float, ndim=1)
        weights = _readonly(self.regression_weights, dtype=float, ndim=1)
        if coalitions.shape != (int(self.effective_budget) - 2, int(self.n_players)):
            raise ValueError("coalition matrix must contain the interior evaluation budget")
        if inclusion.size != coalitions.shape[0] or weights.size != coalitions.shape[0]:
            raise ValueError("coalition probabilities and weights must align")
        if coalitions.size and (
            np.any(np.sum(coalitions, axis=1) == 0)
            or np.any(np.sum(coalitions, axis=1) == int(self.n_players))
        ):
            raise ValueError("interior coalition matrix must exclude boundary coalitions")
        if np.unique(coalitions, axis=0).shape[0] != coalitions.shape[0]:
            raise ValueError("coalitions must be sampled without replacement")
        if inclusion.size and (
            not np.all(np.isfinite(inclusion))
            or np.any(inclusion <= 0.0)
            or np.any(inclusion > 1.0)
            or not np.all(np.isfinite(weights))
            or np.any(weights <= 0.0)
        ):
            raise ValueError("sampling probabilities and weights must be positive")
        complements = {tuple((~row).tolist()) for row in coalitions}
        if any(tuple(row.tolist()) not in complements for row in coalitions):
            raise ValueError("coalition plan must contain every sampled complement")
        object.__setattr__(self, "coalition_matrix", coalitions)
        object.__setattr__(self, "inclusion_probabilities", inclusion)
        object.__setattr__(self, "regression_weights", weights)

    def to_dict(self, *, include_arrays: bool = False) -> dict[str, Any]:
        sizes = np.sum(self.coalition_matrix, axis=1)
        payload: dict[str, Any] = {
            "policy": self.policy.value,
            "n_players": int(self.n_players),
            "requested_budget": int(self.requested_budget),
            "effective_budget": int(self.effective_budget),
            "interior_coalitions": int(self.coalition_matrix.shape[0]),
            "min_coalition_size": int(np.min(sizes)) if sizes.size else None,
            "max_coalition_size": int(np.max(sizes)) if sizes.size else None,
            "min_inclusion_probability": (
                float(np.min(self.inclusion_probabilities))
                if self.inclusion_probabilities.size
                else None
            ),
            "max_inclusion_probability": (
                float(np.max(self.inclusion_probabilities))
                if self.inclusion_probabilities.size
                else None
            ),
        }
        if include_arrays:
            payload.update(
                {
                    "coalition_matrix": self.coalition_matrix.astype(int).tolist(),
                    "inclusion_probabilities": self.inclusion_probabilities.tolist(),
                    "regression_weights": self.regression_weights.tolist(),
                }
            )
        return payload


@dataclass(frozen=True)
class ShapleyRegressionEstimate:
    """Projected-regression estimate and its numerical diagnostics."""

    values: np.ndarray
    empty_value: float
    full_value: float
    model_evaluations: int
    condition_number: float
    efficiency_residual: float
    plan: CoalitionSamplingPlan

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=float).reshape(-1).copy()
        if values.size != self.plan.n_players or not np.all(np.isfinite(values)):
            raise ValueError("Shapley values must be finite and align with players")
        values.setflags(write=False)
        object.__setattr__(self, "values", values)

    def to_dict(self, *, include_values: bool = False) -> dict[str, Any]:
        payload = {
            "empty_value": float(self.empty_value),
            "full_value": float(self.full_value),
            "model_evaluations": int(self.model_evaluations),
            "condition_number": float(self.condition_number),
            "efficiency_residual": float(self.efficiency_residual),
            "plan": self.plan.to_dict(),
        }
        if include_values:
            payload["values"] = self.values.tolist()
        return payload


def _validate_n_players(n_players: int) -> int:
    n = int(n_players)
    if n < 2:
        raise ValueError("n_players must be at least 2")
    if n > 20:
        raise ValueError("exact coalition enumeration is limited to 20 players")
    return n


def enumerate_coalitions(n_players: int) -> np.ndarray:
    """Enumerate every coalition in stable binary order."""

    n = _validate_n_players(n_players)
    integers = np.arange(2**n, dtype=np.uint64)
    bits = ((integers[:, None] >> np.arange(n, dtype=np.uint64)) & 1).astype(bool)
    return bits


def _canonical_coalition_pairs(n_players: int) -> tuple[np.ndarray, np.ndarray]:
    coalitions = enumerate_coalitions(n_players)
    sizes = np.sum(coalitions, axis=1)
    interior = coalitions[(sizes > 0) & (sizes < n_players)]
    representatives = interior[~interior[:, 0]]
    return representatives, np.sum(representatives, axis=1).astype(int)


def _kernel_regression_weight(n_players: int, sizes: np.ndarray) -> np.ndarray:
    combinations = np.asarray(
        [math.comb(n_players, int(size)) for size in sizes],
        dtype=float,
    )
    return 1.0 / (combinations * sizes * (n_players - sizes))


def build_coalition_sampling_plan(
    n_players: int,
    *,
    budget: int,
    policy: str | CoalitionSamplingPolicy,
    random_state: int | np.random.Generator | None = None,
) -> CoalitionSamplingPlan:
    """Build an exact even budget with empty/full coalitions evaluated separately."""

    n = _validate_n_players(n_players)
    resolved = policy if isinstance(policy, CoalitionSamplingPolicy) else CoalitionSamplingPolicy(str(policy))
    maximum = 2**n
    requested = int(budget)
    effective = max(6, min(requested, maximum))
    effective -= effective % 2
    representatives, sizes = _canonical_coalition_pairs(n)
    pair_budget = min((effective - 2) // 2, representatives.shape[0])
    effective = 2 + 2 * pair_budget
    kernel_weights = _kernel_regression_weight(n, sizes)
    if resolved is CoalitionSamplingPolicy.LEVERAGE:
        pair_scores = np.asarray(
            [1.0 / math.comb(n, int(size)) for size in sizes],
            dtype=float,
        )
    else:
        pair_scores = kernel_weights
    pair_probabilities = saturated_inclusion_probabilities(pair_scores, pair_budget)
    rng = (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )
    pair_levels = np.minimum(sizes, n - sizes)
    levels = np.unique(pair_levels)
    expected_counts = np.asarray(
        [float(np.sum(pair_probabilities[pair_levels == level])) for level in levels],
        dtype=float,
    )
    level_counts = np.floor(expected_counts).astype(int)
    fractional = expected_counts - level_counts
    remainder = pair_budget - int(np.sum(level_counts))
    if remainder:
        if not np.isclose(float(np.sum(fractional)), remainder, atol=1e-8):
            raise RuntimeError("coalition-size expectations do not preserve the budget")
        level_counts += np.isin(
            np.arange(levels.size),
            dependent_round_exact(fractional, random_state=rng),
        ).astype(int)
    selected_groups: list[np.ndarray] = []
    for level, count in zip(levels, level_counts):
        candidates = np.flatnonzero(pair_levels == level)
        if count:
            selected_groups.append(
                np.asarray(rng.choice(candidates, int(count), replace=False), dtype=int)
            )
    selected_pairs = np.sort(np.concatenate(selected_groups))
    if selected_pairs.size != pair_budget:
        raise RuntimeError("coalition sampling did not preserve the exact pair budget")
    selected = representatives[selected_pairs]
    coalitions = np.vstack([selected, ~selected])
    selected_probabilities = pair_probabilities[selected_pairs]
    inclusion = np.concatenate([selected_probabilities, selected_probabilities])
    selected_sizes = np.sum(coalitions, axis=1).astype(int)
    regression = _kernel_regression_weight(n, selected_sizes) / inclusion
    return CoalitionSamplingPlan(
        policy=resolved,
        n_players=n,
        requested_budget=requested,
        effective_budget=effective,
        coalition_matrix=coalitions,
        inclusion_probabilities=inclusion,
        regression_weights=regression,
    )


def _evaluate_game(game: CoalitionGame, coalitions: np.ndarray) -> np.ndarray:
    values = np.asarray(game(np.asarray(coalitions, dtype=bool)), dtype=float).reshape(-1)
    if values.size != coalitions.shape[0] or not np.all(np.isfinite(values)):
        raise ValueError("game must return one finite value per coalition")
    return values


def estimate_projected_shapley(
    game: CoalitionGame,
    n_players: int,
    *,
    budget: int,
    policy: str | CoalitionSamplingPolicy = CoalitionSamplingPolicy.LEVERAGE,
    random_state: int | np.random.Generator | None = None,
) -> ShapleyRegressionEstimate:
    """Estimate Shapley values using the exact efficiency projection."""

    plan = build_coalition_sampling_plan(
        n_players,
        budget=budget,
        policy=policy,
        random_state=random_state,
    )
    empty = np.zeros((1, plan.n_players), dtype=bool)
    full = np.ones((1, plan.n_players), dtype=bool)
    empty_value = float(_evaluate_game(game, empty)[0])
    full_value = float(_evaluate_game(game, full)[0])
    coalitions = plan.coalition_matrix.astype(float)
    coalition_values = _evaluate_game(game, plan.coalition_matrix)
    sizes = np.sum(coalitions, axis=1)
    total_effect = full_value - empty_value
    adjusted = coalition_values - empty_value - sizes * total_effect / plan.n_players
    projection = np.eye(plan.n_players) - np.ones(
        (plan.n_players, plan.n_players), dtype=float
    ) / plan.n_players
    weighted_coalitions = coalitions * plan.regression_weights.reshape(-1, 1)
    gram = projection @ coalitions.T @ weighted_coalitions @ projection
    rhs = projection @ coalitions.T @ (plan.regression_weights * adjusted)
    centered = np.linalg.lstsq(gram, rhs, rcond=None)[0]
    values = centered + total_effect / plan.n_players
    condition = float(np.linalg.cond(gram))
    efficiency = float(abs(np.sum(values) - total_effect))
    return ShapleyRegressionEstimate(
        values=values,
        empty_value=empty_value,
        full_value=full_value,
        model_evaluations=plan.effective_budget,
        condition_number=condition,
        efficiency_residual=efficiency,
        plan=plan,
    )


def exact_shapley_values(game: CoalitionGame, n_players: int) -> np.ndarray:
    """Compute exact Shapley values for a small cooperative game."""

    n = _validate_n_players(n_players)
    coalitions = enumerate_coalitions(n)
    values = _evaluate_game(game, coalitions)
    integer_masks = np.arange(2**n, dtype=np.uint64)
    result = np.zeros(n, dtype=float)
    factorial_n = math.factorial(n)
    for player in range(n):
        absent = ~coalitions[:, player]
        indices = integer_masks[absent]
        sizes = np.sum(coalitions[absent], axis=1).astype(int)
        with_player = indices | (np.uint64(1) << np.uint64(player))
        coefficients = np.asarray(
            [
                math.factorial(int(size))
                * math.factorial(n - int(size) - 1)
                / factorial_n
                for size in sizes
            ],
            dtype=float,
        )
        result[player] = float(
            np.sum(coefficients * (values[with_player.astype(int)] - values[indices.astype(int)]))
        )
    return result


def full_kernel_regression_loss(
    game: CoalitionGame,
    shapley_values: Sequence[float] | np.ndarray,
) -> float:
    """Evaluate the exact KernelSHAP weighted objective on all interior coalitions."""

    estimate = np.asarray(shapley_values, dtype=float).reshape(-1)
    n = _validate_n_players(estimate.size)
    coalitions = enumerate_coalitions(n)
    sizes = np.sum(coalitions, axis=1).astype(int)
    interior = (sizes > 0) & (sizes < n)
    coalitions = coalitions[interior].astype(float)
    sizes = sizes[interior]
    values = _evaluate_game(game, coalitions.astype(bool))
    empty = float(_evaluate_game(game, np.zeros((1, n), dtype=bool))[0])
    residual = coalitions @ estimate - (values - empty)
    weights = _kernel_regression_weight(n, sizes)
    return float(np.sum(weights * np.square(residual)))


def make_masked_model_game(
    baseline: Sequence[float] | np.ndarray,
    explicand: Sequence[float] | np.ndarray,
    predict: Callable[[np.ndarray], np.ndarray],
) -> CoalitionGame:
    """Adapt a tabular prediction function to a coalition game."""

    baseline_values = np.asarray(baseline, dtype=float).reshape(-1)
    explicand_values = np.asarray(explicand, dtype=float).reshape(-1)
    if baseline_values.size != explicand_values.size:
        raise ValueError("baseline and explicand must have equal width")

    def game(coalitions: np.ndarray) -> np.ndarray:
        masks = np.asarray(coalitions, dtype=bool)
        inputs = np.where(masks, explicand_values, baseline_values)
        return np.asarray(predict(inputs), dtype=float).reshape(-1)

    return game
