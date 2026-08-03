"""Pure budget allocation for partition-based experiments."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, Mapping, Sequence


class PartitionBudgetViolation(str, Enum):
    EMPTY_PARTITIONS = "empty_partitions"
    SOURCE_BELOW_MIN_ROWS = "source_below_min_rows"
    BUDGET_BELOW_MIN_ROWS = "budget_below_min_rows"
    MAX_IMBALANCE_RATIO = "max_imbalance_ratio"
    MIN_PARTITION_FRACTION = "min_partition_fraction"


@dataclass(frozen=True)
class PartitionBudgetPlan:
    """Deterministic allocation shared by selection and runtime sampling."""

    budget_ratio: float
    total_rows: int
    requested_budget_size: int
    source_sizes: tuple[tuple[str, int], ...]
    allocations: tuple[tuple[str, int], ...]
    min_rows_per_partition: int
    max_imbalance_ratio: float | None
    min_partition_fraction: float
    violations: tuple[PartitionBudgetViolation, ...] = ()

    @property
    def feasible(self) -> bool:
        return not self.violations

    @property
    def selected_size(self) -> int:
        return sum(count for _, count in self.allocations)

    @property
    def allocation_map(self) -> dict[str, int]:
        return dict(self.allocations)

    @property
    def source_size_map(self) -> dict[str, int]:
        return dict(self.source_sizes)

    def to_dict(self) -> dict[str, Any]:
        counts = tuple(count for _, count in self.allocations)
        min_count = min(counts) if counts else 0
        max_count = max(counts) if counts else 0
        return {
            "budget_ratio": float(self.budget_ratio),
            "total_rows": int(self.total_rows),
            "requested_budget_size": int(self.requested_budget_size),
            "selected_size": int(self.selected_size),
            "source_sizes": self.source_size_map,
            "allocations": self.allocation_map,
            "min_rows_per_partition": int(self.min_rows_per_partition),
            "max_imbalance_ratio": self.max_imbalance_ratio,
            "min_partition_fraction": float(self.min_partition_fraction),
            "observed_imbalance_ratio": (
                float(max_count / min_count) if min_count > 0 else None
            ),
            "observed_min_partition_fraction": (
                float(min_count / self.selected_size)
                if self.selected_size > 0 and counts
                else None
            ),
            "feasible": bool(self.feasible),
            "violations": [violation.value for violation in self.violations],
        }


def build_partition_budget_plan(
    partition_sizes: Mapping[str, int] | Sequence[int],
    *,
    total_rows: int,
    budget_ratio: float,
    min_rows_per_partition: int = 1,
    max_imbalance_ratio: float | None = None,
    min_partition_fraction: float = 0.0,
) -> PartitionBudgetPlan:
    """Allocate an exact global budget without silently dropping partitions."""

    normalized_sizes = _normalize_partition_sizes(partition_sizes)
    total_rows = int(total_rows)
    if total_rows < 1:
        raise ValueError("total_rows must be positive")
    ratio = float(budget_ratio)
    if not math.isfinite(ratio) or not 0 < ratio <= 1:
        raise ValueError("budget_ratio must be a finite value in (0, 1]")
    min_rows = int(min_rows_per_partition)
    if min_rows < 1:
        raise ValueError("min_rows_per_partition must be positive")
    min_fraction = float(min_partition_fraction)
    if not math.isfinite(min_fraction) or not 0 <= min_fraction <= 1:
        raise ValueError("min_partition_fraction must be in [0, 1]")
    if max_imbalance_ratio is not None:
        max_imbalance_ratio = float(max_imbalance_ratio)
        if not math.isfinite(max_imbalance_ratio) or max_imbalance_ratio < 1:
            raise ValueError("max_imbalance_ratio must be at least 1")

    requested_budget = max(1, min(total_rows, int(round(total_rows * ratio))))
    if not normalized_sizes:
        return PartitionBudgetPlan(
            budget_ratio=ratio,
            total_rows=total_rows,
            requested_budget_size=requested_budget,
            source_sizes=(),
            allocations=(),
            min_rows_per_partition=min_rows,
            max_imbalance_ratio=max_imbalance_ratio,
            min_partition_fraction=min_fraction,
            violations=(PartitionBudgetViolation.EMPTY_PARTITIONS,),
        )

    names = tuple(name for name, _ in normalized_sizes)
    sizes = tuple(size for _, size in normalized_sizes)
    effective_budget = min(requested_budget, sum(sizes))
    source_below_min = any(size < min_rows for size in sizes)
    budget_below_min = effective_budget < min_rows * len(sizes)

    if source_below_min or budget_below_min:
        lower_bounds = tuple(1 for _ in sizes)
    else:
        lower_bounds = tuple(min_rows for _ in sizes)
    allocations = _bounded_proportional_allocation(
        capacities=sizes,
        total=effective_budget,
        lower_bounds=lower_bounds,
    )

    violations: list[PartitionBudgetViolation] = []
    if source_below_min:
        violations.append(PartitionBudgetViolation.SOURCE_BELOW_MIN_ROWS)
    if budget_below_min:
        violations.append(PartitionBudgetViolation.BUDGET_BELOW_MIN_ROWS)
    min_count = min(allocations)
    max_count = max(allocations)
    selected_size = sum(allocations)
    if (
        max_imbalance_ratio is not None
        and min_count > 0
        and max_count / min_count > max_imbalance_ratio
    ):
        violations.append(PartitionBudgetViolation.MAX_IMBALANCE_RATIO)
    if selected_size > 0 and min_count / selected_size < min_fraction:
        violations.append(PartitionBudgetViolation.MIN_PARTITION_FRACTION)

    return PartitionBudgetPlan(
        budget_ratio=ratio,
        total_rows=total_rows,
        requested_budget_size=requested_budget,
        source_sizes=normalized_sizes,
        allocations=tuple(zip(names, allocations)),
        min_rows_per_partition=min_rows,
        max_imbalance_ratio=max_imbalance_ratio,
        min_partition_fraction=min_fraction,
        violations=tuple(violations),
    )


def _normalize_partition_sizes(
    values: Mapping[str, int] | Sequence[int],
) -> tuple[tuple[str, int], ...]:
    items = (
        tuple((str(name), int(size)) for name, size in values.items())
        if isinstance(values, Mapping)
        else tuple((f"chunk_{index}", int(size)) for index, size in enumerate(values))
    )
    if any(size < 1 for _, size in items):
        raise ValueError("partition sizes must be positive")
    return items


def _bounded_proportional_allocation(
    *,
    capacities: Sequence[int],
    total: int,
    lower_bounds: Sequence[int],
) -> tuple[int, ...]:
    capacities = tuple(int(value) for value in capacities)
    lower_bounds = tuple(int(value) for value in lower_bounds)
    if len(capacities) != len(lower_bounds):
        raise ValueError("capacities and lower_bounds must align")
    if not capacities:
        return ()
    total = max(0, min(int(total), sum(capacities)))
    allocations = [min(capacity, lower) for capacity, lower in zip(capacities, lower_bounds)]
    if sum(allocations) > total:
        allocations = [0 for _ in capacities]

    remaining = total - sum(allocations)
    while remaining > 0:
        residual = [capacity - count for capacity, count in zip(capacities, allocations)]
        active = [index for index, value in enumerate(residual) if value > 0]
        if not active:
            break
        residual_total = sum(residual[index] for index in active)
        quotas = {
            index: remaining * residual[index] / residual_total
            for index in active
        }
        floor_additions = {
            index: min(residual[index], int(math.floor(quotas[index])))
            for index in active
        }
        added = sum(floor_additions.values())
        for index, value in floor_additions.items():
            allocations[index] += value
        remaining -= added
        if remaining <= 0:
            break
        order = sorted(
            active,
            key=lambda index: (
                -(quotas[index] - math.floor(quotas[index])),
                -residual[index],
                index,
            ),
        )
        progressed = False
        for index in order:
            if remaining <= 0:
                break
            if allocations[index] >= capacities[index]:
                continue
            allocations[index] += 1
            remaining -= 1
            progressed = True
        if not progressed:
            break
    return tuple(allocations)
