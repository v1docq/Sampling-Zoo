from __future__ import annotations

import numpy as np
import pytest

from sampling_zoo.core.sampling_strategies.spectral.partition_sampling import (
    partition_membership_fingerprint,
    select_partition_indices,
)


def _select_capped(seed: int) -> np.ndarray:
    indices = np.arange(20)
    scores = np.ones(20)
    scores[0] = 1_000_000.0
    return select_partition_indices(
        indices,
        target_size=5,
        selection_method="capped_leverage",
        scores=scores,
        embedding=np.eye(20),
        random_state=seed,
        leverage_cap_quantile=0.90,
    )


def test_capped_leverage_is_deterministic_unique_and_budget_exact() -> None:
    first = _select_capped(17)
    second = _select_capped(17)

    assert np.array_equal(first, second)
    assert first.size == 5
    assert np.unique(first).size == first.size


def test_capped_leverage_prevents_extreme_score_from_always_winning() -> None:
    selected_extreme_count = sum(0 in _select_capped(seed) for seed in range(40))

    assert 0 < selected_extreme_count < 40


@pytest.mark.parametrize("quantile", [0.0, -0.1, 1.1])
def test_capped_leverage_rejects_invalid_cap_quantile(quantile: float) -> None:
    with pytest.raises(ValueError, match="leverage_cap_quantile"):
        select_partition_indices(
            np.arange(5),
            target_size=2,
            selection_method="capped_leverage",
            scores=np.ones(5),
            embedding=np.eye(5),
            random_state=42,
            leverage_cap_quantile=quantile,
        )


def test_partition_fingerprint_is_invariant_to_label_permutation() -> None:
    first = partition_membership_fingerprint([4, 4, 8, 8, 4, 2])
    second = partition_membership_fingerprint([1, 1, 0, 0, 1, 7])

    assert first == second
    assert first != partition_membership_fingerprint([4, 8, 4, 8, 4, 2])
