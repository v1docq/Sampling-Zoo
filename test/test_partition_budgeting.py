from __future__ import annotations

import pytest

from sampling_zoo.core.experiment.budgeting import (
    PartitionBudgetViolation,
    build_partition_budget_plan,
)


def test_budget_plan_allocates_exact_budget_and_preserves_minimums() -> None:
    plan = build_partition_budget_plan(
        {"chunk_0": 700, "chunk_1": 200, "chunk_2": 100},
        total_rows=1000,
        budget_ratio=0.1,
        min_rows_per_partition=20,
        max_imbalance_ratio=5.0,
        min_partition_fraction=0.05,
    )

    assert plan.feasible
    assert plan.selected_size == 100
    assert sum(plan.allocation_map.values()) == plan.requested_budget_size
    assert min(plan.allocation_map.values()) >= 20
    assert all(
        plan.allocation_map[name] <= source_size
        for name, source_size in plan.source_size_map.items()
    )


def test_budget_plan_marks_too_many_partitions_infeasible() -> None:
    plan = build_partition_budget_plan(
        {f"chunk_{index}": 100 for index in range(5)},
        total_rows=500,
        budget_ratio=0.1,
        min_rows_per_partition=16,
    )

    assert not plan.feasible
    assert PartitionBudgetViolation.BUDGET_BELOW_MIN_ROWS in plan.violations
    assert plan.selected_size == 50


def test_budget_plan_is_deterministic_and_never_exceeds_capacities() -> None:
    kwargs = dict(
        partition_sizes={"a": 3, "b": 7, "c": 11},
        total_rows=21,
        budget_ratio=0.6,
        min_rows_per_partition=1,
    )

    first = build_partition_budget_plan(**kwargs)
    second = build_partition_budget_plan(**kwargs)

    assert first == second
    assert first.selected_size == 13
    assert all(
        first.allocation_map[name] <= first.source_size_map[name]
        for name in first.allocation_map
    )


@pytest.mark.parametrize("ratio", [0.0, -0.1, 1.1, float("nan")])
def test_budget_plan_rejects_invalid_ratios(ratio: float) -> None:
    with pytest.raises(ValueError, match="budget_ratio"):
        build_partition_budget_plan(
            {"chunk": 10},
            total_rows=10,
            budget_ratio=ratio,
        )
