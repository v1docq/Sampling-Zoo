"""Pure class-coverage planning for spectral partition row selection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from sklearn.utils.multiclass import type_of_target

from .leverage_sketch import combine_stratified_sketch_plans
from .partition_sampling import build_partition_sketch_plan
from .sketch_contracts import ExactBudgetSketchPlan


@dataclass(frozen=True)
class ClassCoverageSelectionPlan:
    """Result of exact-budget, class-aware row selection for one partition."""

    selected_indices: np.ndarray
    target_size: int
    min_samples_per_class: int
    allocation_policy: str
    source_class_counts: tuple[tuple[str, int], ...]
    allocated_class_counts: tuple[tuple[str, int], ...]
    selected_class_counts: tuple[tuple[str, int], ...]
    missing_classes: tuple[str, ...]
    feasible: bool
    violations: tuple[str, ...]
    distribution_total_variation: float | None
    selected_probabilities: np.ndarray = field(
        default_factory=lambda: np.asarray([], dtype=float)
    )
    training_weights: np.ndarray = field(
        default_factory=lambda: np.asarray([], dtype=float)
    )
    sketch_plans: tuple[ExactBudgetSketchPlan, ...] = ()
    combined_sketch_plan: ExactBudgetSketchPlan | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_size": int(self.target_size),
            "selected_size": int(self.selected_indices.size),
            "min_samples_per_class": int(self.min_samples_per_class),
            "allocation_policy": self.allocation_policy,
            "source_class_counts": dict(self.source_class_counts),
            "allocated_class_counts": dict(self.allocated_class_counts),
            "selected_class_counts": dict(self.selected_class_counts),
            "missing_classes": list(self.missing_classes),
            "n_source_classes": len(self.source_class_counts),
            "n_selected_classes": len(self.selected_class_counts),
            "single_class_selection": len(self.selected_class_counts) <= 1,
            "feasible": bool(self.feasible),
            "violations": list(self.violations),
            "distribution_total_variation": self.distribution_total_variation,
            "weight_min": (
                float(np.min(self.training_weights))
                if self.training_weights.size
                else None
            ),
            "weight_max": (
                float(np.max(self.training_weights))
                if self.training_weights.size
                else None
            ),
            "sketch_plans": [plan.to_dict() for plan in self.sketch_plans],
            "combined_sketch_plan": (
                self.combined_sketch_plan.to_dict()
                if self.combined_sketch_plan is not None
                else None
            ),
        }


def infer_target_type(target: Any, configured: str = "auto") -> str:
    """Resolve a target kind without coupling the sampler to cluster selection."""

    normalized = str(configured).strip().lower()
    if normalized != "auto":
        return normalized
    if target is None:
        return "none"
    try:
        inferred = type_of_target(np.asarray(target))
    except (TypeError, ValueError):
        return "none"
    if inferred in {"binary", "multiclass"}:
        return "classification"
    if inferred == "continuous":
        return "regression"
    return "none"


def select_class_aware_partition_indices(
    candidate_indices: Sequence[int] | np.ndarray,
    *,
    target: Sequence[Any] | np.ndarray,
    target_size: int,
    min_samples_per_class: int,
    class_allocation_policy: str = "minimum_then_global",
    selection_method: str,
    scores: np.ndarray,
    embedding: np.ndarray,
    random_state: int | np.random.Generator | None = None,
    leverage_cap_quantile: float = 0.95,
    leverage_mixture_alpha: float = 0.25,
    leverage_uniform_floor: float = 1e-12,
    ridge_scores: np.ndarray | None = None,
    ridge_lambda: float | None = None,
    training_reweighting: str = "none",
) -> ClassCoverageSelectionPlan:
    """Select an exact-size subset while preserving every locally observed class."""

    indices = np.asarray(candidate_indices, dtype=int).reshape(-1)
    target_values = np.asarray(target).reshape(-1)
    resolved_size = max(0, min(int(target_size), indices.size))
    minimum = int(min_samples_per_class)
    if minimum < 1:
        raise ValueError("min_samples_per_class must be positive")
    allocation_policy = str(class_allocation_policy).strip().lower()
    if allocation_policy not in {"minimum_then_global", "proportional"}:
        raise ValueError(
            "class_allocation_policy must be one of: "
            "minimum_then_global, proportional"
        )
    if indices.size and target_values.size <= int(np.max(indices)):
        raise ValueError("target must align with candidate indices")

    local_target = target_values[indices]
    classes, encoded = np.unique(local_target, return_inverse=True)
    source_counts = np.bincount(encoded, minlength=classes.size)
    labels = tuple(_class_label(value) for value in classes.tolist())
    source_pairs = tuple(zip(labels, source_counts.astype(int).tolist()))
    required_rows = int(classes.size * minimum)
    violations: list[str] = []
    if resolved_size < required_rows:
        violations.append("budget_below_class_coverage_minimum")
    if np.any(source_counts < minimum):
        violations.append("source_class_below_required_minimum")
    if classes.size < 2:
        violations.append("source_partition_is_single_class")

    if violations:
        return ClassCoverageSelectionPlan(
            selected_indices=np.asarray([], dtype=int),
            target_size=resolved_size,
            min_samples_per_class=minimum,
            allocation_policy=allocation_policy,
            source_class_counts=source_pairs,
            allocated_class_counts=(),
            selected_class_counts=(),
            missing_classes=labels,
            feasible=False,
            violations=tuple(violations),
            distribution_total_variation=None,
        )

    rng = (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )
    if allocation_policy == "proportional":
        allocation = _proportional_class_allocation(
            source_counts,
            target_size=resolved_size,
            minimum=minimum,
        )
        sketch_plans = [
            build_partition_sketch_plan(
                indices[encoded == class_index],
                target_size=int(class_size),
                selection_method=selection_method,
                scores=scores,
                embedding=embedding,
                random_state=rng,
                leverage_cap_quantile=leverage_cap_quantile,
                leverage_mixture_alpha=leverage_mixture_alpha,
                leverage_uniform_floor=leverage_uniform_floor,
                ridge_scores=ridge_scores,
                ridge_lambda=ridge_lambda,
                training_reweighting=training_reweighting,
            )
            for class_index, class_size in enumerate(allocation)
        ]
        selected = np.concatenate(
            [plan.selected_indices for plan in sketch_plans]
        ).astype(int, copy=False)
    else:
        allocation = np.full(classes.size, minimum, dtype=int)
        sketch_plans = [
            build_partition_sketch_plan(
                indices[encoded == class_index],
                target_size=minimum,
                selection_method=selection_method,
                scores=scores,
                embedding=embedding,
                random_state=rng,
                leverage_cap_quantile=leverage_cap_quantile,
                leverage_mixture_alpha=leverage_mixture_alpha,
                leverage_uniform_floor=leverage_uniform_floor,
                ridge_scores=ridge_scores,
                ridge_lambda=ridge_lambda,
                training_reweighting=training_reweighting,
            )
            for class_index in range(classes.size)
        ]
        selected = np.concatenate(
            [plan.selected_indices for plan in sketch_plans]
        ).astype(int, copy=False)
        remaining_size = resolved_size - selected.size
        if remaining_size > 0:
            remaining = indices[~np.isin(indices, selected)]
            remaining_plan = build_partition_sketch_plan(
                remaining,
                target_size=remaining_size,
                selection_method=selection_method,
                scores=scores,
                embedding=embedding,
                random_state=rng,
                leverage_cap_quantile=leverage_cap_quantile,
                leverage_mixture_alpha=leverage_mixture_alpha,
                leverage_uniform_floor=leverage_uniform_floor,
                ridge_scores=ridge_scores,
                ridge_lambda=ridge_lambda,
                training_reweighting=training_reweighting,
            )
            sketch_plans.append(remaining_plan)
            selected = np.concatenate(
                [
                    selected,
                    remaining_plan.selected_indices,
                ]
            )
        selected_target_for_allocation = target_values[selected]
        allocation = np.asarray(
            [
                np.sum(selected_target_for_allocation == value)
                for value in classes
            ],
            dtype=int,
        )

    selected_target = target_values[selected]
    selected_counts = np.asarray(
        [np.sum(selected_target == value) for value in classes],
        dtype=int,
    )
    selected_pairs = tuple(zip(labels, selected_counts.tolist()))
    allocated_pairs = tuple(zip(labels, allocation.astype(int).tolist()))
    missing = tuple(
        label for label, count in selected_pairs if int(count) < minimum
    )
    drift = _distribution_total_variation(source_counts, selected_counts)
    selected_probabilities = np.concatenate(
        [plan.selected_probabilities for plan in sketch_plans]
    )
    training_weights = np.concatenate(
        [plan.training_weights for plan in sketch_plans]
    )
    combined_sketch_plan = combine_stratified_sketch_plans(
        indices,
        sketch_plans,
        scores=scores,
    )
    return ClassCoverageSelectionPlan(
        selected_indices=selected,
        target_size=resolved_size,
        min_samples_per_class=minimum,
        allocation_policy=allocation_policy,
        source_class_counts=source_pairs,
        allocated_class_counts=allocated_pairs,
        selected_class_counts=selected_pairs,
        missing_classes=missing,
        feasible=selected.size == resolved_size and not missing,
        violations=(() if selected.size == resolved_size and not missing else ("selection_invariant_failed",)),
        distribution_total_variation=drift,
        selected_probabilities=selected_probabilities,
        training_weights=training_weights,
        sketch_plans=tuple(sketch_plans),
        combined_sketch_plan=combined_sketch_plan,
    )


def _proportional_class_allocation(
    source_counts: np.ndarray,
    *,
    target_size: int,
    minimum: int,
) -> np.ndarray:
    """Allocate an exact class budget with lower bounds and finite capacities."""

    counts = np.asarray(source_counts, dtype=int).reshape(-1)
    if counts.size == 0:
        return np.asarray([], dtype=int)
    allocation = np.full(counts.size, int(minimum), dtype=int)
    if np.any(allocation > counts) or int(allocation.sum()) > int(target_size):
        raise ValueError("class allocation is infeasible")

    remaining = int(target_size) - int(allocation.sum())
    while remaining > 0:
        capacity = counts - allocation
        active = np.flatnonzero(capacity > 0)
        if active.size == 0:
            raise ValueError("class allocation cannot fill the requested budget")

        ideal = remaining * capacity[active] / float(capacity[active].sum())
        additions = np.minimum(
            np.floor(ideal).astype(int),
            capacity[active],
        )
        allocated_now = int(additions.sum())
        if allocated_now > 0:
            allocation[active] += additions
            remaining -= allocated_now
            continue

        fractional = ideal - np.floor(ideal)
        best_local = int(np.argmax(fractional))
        allocation[int(active[best_local])] += 1
        remaining -= 1
    return allocation


def _class_label(value: Any) -> str:
    if isinstance(value, np.generic):
        value = value.item()
    return str(value)


def _distribution_total_variation(
    source_counts: np.ndarray,
    selected_counts: np.ndarray,
) -> float:
    source = np.asarray(source_counts, dtype=float)
    selected = np.asarray(selected_counts, dtype=float)
    source /= max(float(source.sum()), 1.0)
    selected /= max(float(selected.sum()), 1.0)
    return float(0.5 * np.sum(np.abs(source - selected)))
