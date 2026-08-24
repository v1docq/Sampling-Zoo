"""Exact-budget expert topologies and hierarchical routing for bulk/spike RMT."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence, Tuple
import zlib

import numpy as np

from .bulk_spike import (
    BulkSpikeBudgetPolicy,
    RowSpectralParticipationContract,
    build_bulk_spike_training_plan,
)
from .leverage_sketch import build_exact_budget_sketch_plan
from .routing_contracts import RoutingWeightContract


class BulkSpikeTopologyMode(str, Enum):
    STANDARD = "standard"
    SINGLE_SPIKE = "single_spike"
    MULTI_SPIKE = "multi_spike"
    SELECTED = "selected"
    ORACLE = "oracle"


def _readonly_indices(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=int).reshape(-1).copy()
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class BulkSpikeExpertTopologySpec:
    """Frozen topology, budget, and spike-branch routing policy."""

    name: str
    mode: BulkSpikeTopologyMode | str
    max_experts: int = 5
    min_partition_size: int = 32
    budget_policy: BulkSpikeBudgetPolicy | str = BulkSpikeBudgetPolicy.EXCESS_ENERGY
    fixed_spike_share: float = 0.25
    min_spike_share: float = 0.10
    max_spike_share: float = 0.50
    signal_threshold: float = 0.50
    spike_router: str = "gmm_posterior"
    covariance_shrinkage: float = 0.10

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise ValueError("bulk/spike topology name must be non-empty")
        mode = (
            self.mode
            if isinstance(self.mode, BulkSpikeTopologyMode)
            else BulkSpikeTopologyMode(str(self.mode))
        )
        policy = (
            self.budget_policy
            if isinstance(self.budget_policy, BulkSpikeBudgetPolicy)
            else BulkSpikeBudgetPolicy(str(self.budget_policy))
        )
        if int(self.max_experts) < 1:
            raise ValueError("max_experts must be positive")
        if int(self.min_partition_size) < 1:
            raise ValueError("min_partition_size must be positive")
        if not 0.0 <= float(self.min_spike_share) <= float(self.max_spike_share) <= 1.0:
            raise ValueError("spike shares must satisfy 0 <= min <= max <= 1")
        if not 0.0 <= float(self.fixed_spike_share) <= 1.0:
            raise ValueError("fixed_spike_share must be in [0, 1]")
        if not 0.0 <= float(self.signal_threshold) <= 1.0:
            raise ValueError("signal_threshold must be in [0, 1]")
        if self.spike_router not in {
            "diag_shrinkage_mahalanobis",
            "full_shrinkage_mahalanobis",
            "gmm_posterior",
        }:
            raise ValueError("spike_router must be a shrinkage Mahalanobis or GMM policy")
        if not 0.0 <= float(self.covariance_shrinkage) <= 1.0:
            raise ValueError("covariance_shrinkage must be in [0, 1]")
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "budget_policy", policy)
        object.__setattr__(self, "max_experts", int(self.max_experts))
        object.__setattr__(self, "min_partition_size", int(self.min_partition_size))


@dataclass(frozen=True)
class BulkSpikePartitionContract:
    """Disjoint expert partitions whose union matches one exact row budget."""

    topology_name: str
    mode: BulkSpikeTopologyMode
    partition_names: Tuple[str, ...]
    regimes: Tuple[str, ...]
    row_indices: Tuple[np.ndarray, ...]
    total_budget: int
    max_experts: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        names = tuple(str(value) for value in self.partition_names)
        regimes = tuple(str(value) for value in self.regimes)
        rows = tuple(_readonly_indices(value) for value in self.row_indices)
        if not names or len(set(names)) != len(names):
            raise ValueError("partition_names must be non-empty and unique")
        if len(names) != len(regimes) or len(names) != len(rows):
            raise ValueError("partition names, regimes, and rows must align")
        if len(names) > int(self.max_experts):
            raise ValueError("partition count exceeds max_experts")
        if any(values.size == 0 for values in rows):
            raise ValueError("bulk/spike partitions must not be empty")
        concatenated = np.concatenate(rows)
        if concatenated.size != int(self.total_budget):
            raise ValueError("partition rows must match the exact total budget")
        if np.unique(concatenated).size != concatenated.size:
            raise ValueError("rows cannot repeat within or across experts")
        object.__setattr__(self, "partition_names", names)
        object.__setattr__(self, "regimes", regimes)
        object.__setattr__(self, "row_indices", rows)
        object.__setattr__(self, "total_budget", int(self.total_budget))
        object.__setattr__(self, "max_experts", int(self.max_experts))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def n_experts(self) -> int:
        return len(self.partition_names)

    @property
    def selected_indices(self) -> np.ndarray:
        values = np.concatenate(self.row_indices)
        values.setflags(write=False)
        return values

    def to_dict(self, *, include_indices: bool = False) -> dict[str, Any]:
        payload = {
            "topology_name": self.topology_name,
            "mode": self.mode.value,
            "partition_names": list(self.partition_names),
            "regimes": list(self.regimes),
            "partition_sizes": [int(values.size) for values in self.row_indices],
            "total_budget": self.total_budget,
            "n_experts": self.n_experts,
            "max_experts": self.max_experts,
            "exact_budget": int(self.selected_indices.size) == self.total_budget,
            "unique_rows": int(np.unique(self.selected_indices).size),
            "metadata": dict(self.metadata),
        }
        if include_indices:
            payload["row_indices"] = [values.tolist() for values in self.row_indices]
        return payload


class BulkSpikeBudgetAllocator:
    """Allocate one exact row budget to single- or multi-spike experts."""

    def allocate(
        self,
        spec: BulkSpikeExpertTopologySpec,
        participation: RowSpectralParticipationContract,
        *,
        total_budget: int,
        random_state: int = 42,
    ) -> BulkSpikePartitionContract:
        if spec.mode is BulkSpikeTopologyMode.SINGLE_SPIKE:
            return self._allocate_single(
                spec,
                participation,
                total_budget=total_budget,
                random_state=random_state,
            )
        if spec.mode is BulkSpikeTopologyMode.MULTI_SPIKE:
            return self._allocate_multi(
                spec,
                participation,
                total_budget=total_budget,
                random_state=random_state,
            )
        raise ValueError("budget allocation requires single_spike or multi_spike mode")

    @staticmethod
    def _allocate_single(
        spec: BulkSpikeExpertTopologySpec,
        participation: RowSpectralParticipationContract,
        *,
        total_budget: int,
        random_state: int,
    ) -> BulkSpikePartitionContract:
        plan = build_bulk_spike_training_plan(
            participation,
            total_budget=total_budget,
            policy=spec.budget_policy,
            fixed_spike_share=spec.fixed_spike_share,
            min_spike_share=spec.min_spike_share,
            max_spike_share=spec.max_spike_share,
            signal_threshold=spec.signal_threshold,
            random_state=random_state,
        )
        names = []
        regimes = []
        rows = []
        if plan.bulk_selected_indices.size:
            names.append("bulk")
            regimes.append("bulk")
            rows.append(plan.bulk_selected_indices)
        if plan.spike_selected_indices.size:
            names.append("spike")
            regimes.append("spike")
            rows.append(plan.spike_selected_indices)
        return BulkSpikePartitionContract(
            topology_name=spec.name,
            mode=spec.mode,
            partition_names=tuple(names),
            regimes=tuple(regimes),
            row_indices=tuple(rows),
            total_budget=plan.total_budget,
            max_experts=spec.max_experts,
            metadata=plan.to_dict(),
        )

    def _allocate_multi(
        self,
        spec: BulkSpikeExpertTopologySpec,
        participation: RowSpectralParticipationContract,
        *,
        total_budget: int,
        random_state: int,
    ) -> BulkSpikePartitionContract:
        single_plan = build_bulk_spike_training_plan(
            participation,
            total_budget=total_budget,
            policy=spec.budget_policy,
            fixed_spike_share=spec.fixed_spike_share,
            min_spike_share=spec.min_spike_share,
            max_spike_share=spec.max_spike_share,
            signal_threshold=spec.signal_threshold,
            random_state=random_state,
        )
        spike_budget = int(single_plan.spike_selected_indices.size)
        signatures = participation.spike_signatures
        if spike_budget == 0 or signatures.shape[1] <= 1:
            return self._allocate_single(
                BulkSpikeExpertTopologySpec(
                    **{**spec.__dict__, "mode": BulkSpikeTopologyMode.SINGLE_SPIKE}
                ),
                participation,
                total_budget=total_budget,
                random_state=random_state,
            )

        all_indices = np.arange(participation.n_rows, dtype=int)
        spike_candidates = all_indices[
            participation.signalness >= spec.signal_threshold
        ]
        component_mass = np.sum(signatures[spike_candidates], axis=0)
        nonzero_components = np.flatnonzero(component_mass > 0.0)
        max_by_budget = max(1, spike_budget // spec.min_partition_size)
        max_components = min(
            len(nonzero_components),
            spec.max_experts - int(single_plan.bulk_selected_indices.size > 0),
            max_by_budget,
        )
        if max_components <= 1:
            return self._allocate_single(
                BulkSpikeExpertTopologySpec(
                    **{**spec.__dict__, "mode": BulkSpikeTopologyMode.SINGLE_SPIKE}
                ),
                participation,
                total_budget=total_budget,
                random_state=random_state,
            )
        active_components = nonzero_components[
            np.argsort(component_mass[nonzero_components], kind="stable")[
                -max_components:
            ]
        ]
        active_components = np.sort(active_components)
        assignments = active_components[
            np.argmax(signatures[spike_candidates][:, active_components], axis=1)
        ]
        candidate_groups = [spike_candidates[assignments == component] for component in active_components]
        sufficiently_large = np.asarray(
            [len(values) >= spec.min_partition_size for values in candidate_groups],
            dtype=bool,
        )
        if np.sum(sufficiently_large) <= 1:
            return self._allocate_single(
                BulkSpikeExpertTopologySpec(
                    **{**spec.__dict__, "mode": BulkSpikeTopologyMode.SINGLE_SPIKE}
                ),
                participation,
                total_budget=total_budget,
                random_state=random_state,
            )
        active_components = active_components[sufficiently_large]
        assignments = active_components[
            np.argmax(signatures[spike_candidates][:, active_components], axis=1)
        ]
        candidate_groups = [spike_candidates[assignments == component] for component in active_components]
        capacities = np.asarray([len(values) for values in candidate_groups], dtype=int)
        masses = np.asarray(
            [component_mass[component] for component in active_components],
            dtype=float,
        )
        group_budgets = self._bounded_largest_remainder(
            spike_budget,
            masses,
            capacities,
            minimum=spec.min_partition_size,
        )

        names = []
        regimes = []
        rows = []
        if single_plan.bulk_selected_indices.size:
            names.append("bulk")
            regimes.append("bulk")
            rows.append(single_plan.bulk_selected_indices)
        rng = np.random.default_rng(random_state)
        for component, candidates, budget in zip(
            active_components,
            candidate_groups,
            group_budgets,
        ):
            if budget <= 0:
                continue
            candidate_scores = (
                participation.signalness[candidates]
                * np.maximum(signatures[candidates, component], 1e-12)
            )
            scores = np.zeros(participation.n_rows, dtype=float)
            scores[candidates] = candidate_scores
            plan = build_exact_budget_sketch_plan(
                candidates,
                target_size=int(budget),
                policy="saturated_leverage",
                scores=scores,
                random_state=rng,
            )
            names.append(f"spike_{int(component)}")
            regimes.append(f"spike:{int(component)}")
            rows.append(plan.selected_indices)
        return BulkSpikePartitionContract(
            topology_name=spec.name,
            mode=spec.mode,
            partition_names=tuple(names),
            regimes=tuple(regimes),
            row_indices=tuple(rows),
            total_budget=int(total_budget),
            max_experts=spec.max_experts,
            metadata={
                "requested_spike_share": single_plan.requested_spike_share,
                "resolved_spike_share": single_plan.resolved_spike_share,
                "active_spike_components": active_components.tolist(),
                "spike_component_budgets": group_budgets.tolist(),
            },
        )

    @staticmethod
    def _bounded_largest_remainder(
        total: int,
        masses: np.ndarray,
        capacities: np.ndarray,
        *,
        minimum: int = 0,
    ) -> np.ndarray:
        if total < 0 or total > int(np.sum(capacities)):
            raise ValueError("total must fit within group capacities")
        if total == 0:
            return np.zeros_like(capacities)
        if minimum < 0 or np.any(capacities < minimum):
            raise ValueError("minimum allocation must fit every group capacity")
        if total < len(capacities) * minimum:
            raise ValueError("total is smaller than the required group minima")
        normalized = masses / max(float(np.sum(masses)), np.finfo(float).eps)
        residual_total = int(total - len(capacities) * minimum)
        residual_capacity = capacities - int(minimum)
        ideal = normalized * residual_total
        allocated = int(minimum) + np.minimum(
            np.floor(ideal).astype(int),
            residual_capacity,
        )
        remaining = int(total - np.sum(allocated))
        remainders = ideal - np.floor(ideal)
        while remaining:
            eligible = np.flatnonzero(allocated < capacities)
            if eligible.size == 0:
                raise RuntimeError("unable to allocate exact component budget")
            chosen = sorted(
                eligible.tolist(),
                key=lambda index: (-remainders[index], index),
            )[0]
            allocated[chosen] += 1
            remainders[chosen] -= 1.0
            remaining -= 1
        return allocated


class BulkSpikeTopologyBuilder:
    """Pure facade that builds deployable B0, B1, or B2 partition contracts."""

    def __init__(self, allocator: Optional[BulkSpikeBudgetAllocator] = None) -> None:
        self.allocator = allocator or BulkSpikeBudgetAllocator()

    def build(
        self,
        spec: BulkSpikeExpertTopologySpec,
        participation: RowSpectralParticipationContract,
        *,
        total_budget: int,
        baseline_partitions: Optional[Mapping[str, Sequence[int]]] = None,
        random_state: int = 42,
    ) -> BulkSpikePartitionContract:
        if spec.mode is BulkSpikeTopologyMode.STANDARD:
            if not baseline_partitions:
                raise ValueError("standard topology requires baseline_partitions")
            return BulkSpikePartitionContract(
                topology_name=spec.name,
                mode=spec.mode,
                partition_names=tuple(baseline_partitions),
                regimes=tuple("standard" for _ in baseline_partitions),
                row_indices=tuple(
                    np.asarray(values, dtype=int)
                    for values in baseline_partitions.values()
                ),
                total_budget=total_budget,
                max_experts=spec.max_experts,
            )
        if spec.mode in {
            BulkSpikeTopologyMode.SINGLE_SPIKE,
            BulkSpikeTopologyMode.MULTI_SPIKE,
        }:
            return self.allocator.allocate(
                spec,
                participation,
                total_budget=total_budget,
                random_state=random_state,
            )
        raise ValueError("selected and oracle modes are decisions, not materialized topologies")


class BulkSpikeHierarchicalRouter:
    """Combine a soft bulk/spike gate with conditional spike routing weights."""

    def route(
        self,
        topology: BulkSpikePartitionContract,
        *,
        spike_probability: Sequence[float] | np.ndarray,
        conditional_spike_weights: Optional[np.ndarray] = None,
    ) -> RoutingWeightContract:
        probabilities = np.asarray(spike_probability, dtype=float).reshape(-1)
        if not np.all(np.isfinite(probabilities)):
            raise ValueError("spike_probability must be finite")
        probabilities = np.clip(probabilities, 0.0, 1.0)
        bulk_columns = [
            index for index, regime in enumerate(topology.regimes) if regime == "bulk"
        ]
        spike_columns = [
            index
            for index, regime in enumerate(topology.regimes)
            if regime.startswith("spike")
        ]
        if len(bulk_columns) > 1:
            raise ValueError("hierarchical routing supports at most one bulk expert")
        weights = np.zeros((probabilities.size, topology.n_experts), dtype=float)
        if bulk_columns:
            weights[:, bulk_columns[0]] = 1.0 - probabilities
        elif spike_columns:
            probabilities = np.ones_like(probabilities)
        if spike_columns:
            conditional = self._normalize_conditional(
                conditional_spike_weights,
                n_rows=probabilities.size,
                n_spikes=len(spike_columns),
            )
            weights[:, spike_columns] = probabilities[:, None] * conditional
        elif bulk_columns:
            weights[:, bulk_columns[0]] = 1.0
        else:
            raise ValueError("topology has neither bulk nor spike experts")
        weights /= np.sum(weights, axis=1, keepdims=True)
        entropy = -np.sum(weights * np.log(np.maximum(weights, 1e-12)), axis=1)
        return RoutingWeightContract(
            partition_names=topology.partition_names,
            weights=weights,
            temperature=1.0,
            diagnostics={
                "mean_spike_probability": float(np.mean(probabilities)),
                "mean_routing_entropy": float(np.mean(entropy)),
                "mean_max_probability": float(np.mean(np.max(weights, axis=1))),
            },
        )

    @staticmethod
    def _normalize_conditional(
        values: Optional[np.ndarray],
        *,
        n_rows: int,
        n_spikes: int,
    ) -> np.ndarray:
        if values is None:
            return np.full((n_rows, n_spikes), 1.0 / n_spikes)
        array = np.asarray(values, dtype=float)
        if array.shape != (n_rows, n_spikes):
            raise ValueError("conditional_spike_weights have incompatible shape")
        if np.any(array < 0.0) or not np.all(np.isfinite(array)):
            raise ValueError("conditional_spike_weights must be finite and non-negative")
        sums = np.sum(array, axis=1, keepdims=True)
        return np.divide(
            array,
            sums,
            out=np.full_like(array, 1.0 / n_spikes),
            where=sums > 0.0,
        )


@dataclass(frozen=True)
class BulkSpikeTrainingRequest:
    topology: BulkSpikePartitionContract
    problem_type: str
    model_family: str
    random_state: int

    def __post_init__(self) -> None:
        if self.problem_type not in {"regression", "classification"}:
            raise ValueError("problem_type must be regression or classification")
        if not self.model_family:
            raise ValueError("model_family must be non-empty")


@dataclass(frozen=True)
class BulkSpikeTrainingResult:
    request: BulkSpikeTrainingRequest
    model_names: Tuple[str, ...]
    models: Tuple[Any, ...]
    fit_rows: Tuple[int, ...]
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        names = tuple(self.model_names)
        models = tuple(self.models)
        fit_rows = tuple(int(value) for value in self.fit_rows)
        expected = self.request.topology.n_experts
        if len(names) != expected or len(models) != expected or len(fit_rows) != expected:
            raise ValueError("trained models must align with topology partitions")
        if sum(fit_rows) != self.request.topology.total_budget:
            raise ValueError("fit_rows must preserve the exact topology budget")
        object.__setattr__(self, "model_names", names)
        object.__setattr__(self, "models", models)
        object.__setattr__(self, "fit_rows", fit_rows)
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "topology": self.request.topology.to_dict(),
            "problem_type": self.request.problem_type,
            "model_family": self.request.model_family,
            "model_names": list(self.model_names),
            "fit_rows": list(self.fit_rows),
            "diagnostics": dict(self.diagnostics),
        }


@dataclass(frozen=True)
class BulkSpikeTopologyFoldScore:
    arm_name: str
    fold_id: str
    value: float
    direction: str

    def __post_init__(self) -> None:
        if not self.arm_name or not self.fold_id:
            raise ValueError("arm_name and fold_id must be non-empty")
        if self.direction not in {"lower", "higher"}:
            raise ValueError("direction must be lower or higher")
        if not np.isfinite(float(self.value)):
            raise ValueError("fold score must be finite")


@dataclass(frozen=True)
class BulkSpikeTopologySelectionResult:
    selected_arm: str
    fallback_arm: str
    status: str
    mean_gain: float
    median_gain: float
    confidence_interval: Tuple[float, float]
    positive_fold_fraction: float
    reason: str


class CrossFittedBulkSpikeTopologySelector:
    """Select B1/B2 only with stable held-out gain; otherwise retain B0."""

    def __init__(
        self,
        *,
        fallback_arm: str = "B0_standard_A9",
        candidate_arms: Sequence[str] = ("B1_bulk_single_spike", "B2_bulk_multi_spike"),
        noninferiority_margin: float = 0.005,
        min_positive_fold_fraction: float = 2.0 / 3.0,
        min_mean_gain: float = 0.0,
        min_median_gain: float = 0.0,
        bootstrap_iterations: int = 2_000,
        random_state: int = 42,
    ) -> None:
        self.fallback_arm = str(fallback_arm)
        self.candidate_arms = tuple(candidate_arms)
        self.noninferiority_margin = float(noninferiority_margin)
        self.min_positive_fold_fraction = float(min_positive_fold_fraction)
        self.min_mean_gain = float(min_mean_gain)
        self.min_median_gain = float(min_median_gain)
        self.bootstrap_iterations = int(bootstrap_iterations)
        self.random_state = int(random_state)
        if not self.fallback_arm or not self.candidate_arms:
            raise ValueError("fallback and candidate topology arms must be non-empty")
        if self.bootstrap_iterations < 100:
            raise ValueError("bootstrap_iterations must be at least 100")
        if self.noninferiority_margin < 0.0:
            raise ValueError("noninferiority_margin must be non-negative")
        if not 0.0 < self.min_positive_fold_fraction <= 1.0:
            raise ValueError("min_positive_fold_fraction must be in (0, 1]")

    def select(
        self,
        scores: Sequence[BulkSpikeTopologyFoldScore],
    ) -> BulkSpikeTopologySelectionResult:
        indexed: dict[str, dict[str, BulkSpikeTopologyFoldScore]] = {}
        for score in scores:
            by_fold = indexed.setdefault(score.arm_name, {})
            if score.fold_id in by_fold:
                raise ValueError("duplicate bulk/spike topology fold score")
            by_fold[score.fold_id] = score
        reference = indexed.get(self.fallback_arm, {})
        if len(reference) < 3:
            raise ValueError("fallback topology requires at least three folds")
        candidates = []
        for arm_name in self.candidate_arms:
            candidate = indexed.get(arm_name, {})
            folds = sorted(set(reference) & set(candidate))
            if len(folds) < 3 or set(reference) != set(candidate):
                continue
            gains = np.asarray(
                [self._gain(reference[fold], candidate[fold]) for fold in folds],
                dtype=float,
            )
            interval = self._interval(gains, arm_name)
            mean = float(np.mean(gains))
            median = float(np.median(gains))
            positive = float(np.mean(gains > 0.0))
            if (
                interval[0] >= -self.noninferiority_margin
                and mean >= self.min_mean_gain
                and median >= self.min_median_gain
                and positive >= self.min_positive_fold_fraction
            ):
                candidates.append((arm_name, mean, median, interval, positive))
        if not candidates:
            return BulkSpikeTopologySelectionResult(
                selected_arm=self.fallback_arm,
                fallback_arm=self.fallback_arm,
                status="fallback_to_b0",
                mean_gain=0.0,
                median_gain=0.0,
                confidence_interval=(0.0, 0.0),
                positive_fold_fraction=0.0,
                reason="no_bulk_spike_candidate_passed_cross_fitted_gates",
            )
        selected = sorted(candidates, key=lambda item: (-item[3][0], -item[2], item[0]))[0]
        return BulkSpikeTopologySelectionResult(
            selected_arm=selected[0],
            fallback_arm=self.fallback_arm,
            status="selected",
            mean_gain=selected[1],
            median_gain=selected[2],
            confidence_interval=selected[3],
            positive_fold_fraction=selected[4],
            reason="bulk_spike_candidate_passed_cross_fitted_gates",
        )

    @staticmethod
    def _gain(
        reference: BulkSpikeTopologyFoldScore,
        candidate: BulkSpikeTopologyFoldScore,
    ) -> float:
        if reference.direction != candidate.direction:
            raise ValueError("paired topology score directions must match")
        difference = (
            reference.value - candidate.value
            if reference.direction == "lower"
            else candidate.value - reference.value
        )
        return float(difference / max(abs(reference.value), np.finfo(float).eps))

    def _interval(self, gains: np.ndarray, arm_name: str) -> Tuple[float, float]:
        seed = np.random.SeedSequence(
            [self.random_state, zlib.crc32(arm_name.encode("utf-8"))]
        )
        rng = np.random.default_rng(seed)
        indices = rng.integers(
            0,
            gains.size,
            size=(self.bootstrap_iterations, gains.size),
        )
        means = np.mean(gains[indices], axis=1)
        lower, upper = np.quantile(means, (0.025, 0.975))
        return float(lower), float(upper)
