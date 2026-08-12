"""Pure contracts and plans for component-aware bulk/spike diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Sequence

import numpy as np

from .leverage_sketch import build_exact_budget_sketch_plan
from .null_diagnostics import (
    NullDiagnosticStatus,
    NullModelPolicy,
    SpectralNullDiagnosticResult,
)


class ComponentSplitStatus(str, Enum):
    """Availability of a statistically supported component split."""

    DISABLED = "disabled"
    NULL_DIAGNOSTIC_UNAVAILABLE = "null_diagnostic_unavailable"
    NO_STABLE_SPIKES = "no_stable_spikes"
    OK = "ok"


class BulkSpikeBudgetPolicy(str, Enum):
    """How the fixed row budget is divided between component regimes."""

    EXCESS_ENERGY = "excess_energy"
    FIXED_SHARE = "fixed_share"


def _readonly_vector(values: Any, *, dtype: Any) -> np.ndarray:
    array = np.asarray(values, dtype=dtype).reshape(-1).copy()
    array.setflags(write=False)
    return array


def _readonly_matrix(values: Any, *, dtype: Any) -> np.ndarray:
    array = np.asarray(values, dtype=dtype).copy()
    if array.ndim != 2:
        raise ValueError("expected a two-dimensional array")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class SpectralComponentSplitContract:
    """Empirical-null split of spectral components, never of data rows."""

    status: ComponentSplitStatus
    singular_values: np.ndarray
    empirical_bulk_edge: Optional[float]
    selection_frequencies: np.ndarray
    spike_mask: np.ndarray
    bulk_mask: np.ndarray
    min_selection_frequency: float
    frequency_source: str
    reason: str

    def __post_init__(self) -> None:
        singular = _readonly_vector(self.singular_values, dtype=float)
        frequencies = _readonly_vector(self.selection_frequencies, dtype=float)
        spike = _readonly_vector(self.spike_mask, dtype=bool)
        bulk = _readonly_vector(self.bulk_mask, dtype=bool)
        widths = {singular.size, frequencies.size, spike.size, bulk.size}
        if len(widths) != 1:
            raise ValueError("all component arrays must have equal width")
        if singular.size and (
            not np.all(np.isfinite(singular))
            or np.any(singular < 0.0)
            or not np.all(np.isfinite(frequencies))
            or np.any(frequencies < 0.0)
            or np.any(frequencies > 1.0)
        ):
            raise ValueError("component values must be finite and valid")
        if np.any(spike & bulk) or not np.all(spike | bulk):
            raise ValueError("spike_mask and bulk_mask must form a partition")
        threshold = float(self.min_selection_frequency)
        if not 0.0 < threshold <= 1.0:
            raise ValueError("min_selection_frequency must be in (0, 1]")
        object.__setattr__(self, "singular_values", singular)
        object.__setattr__(self, "selection_frequencies", frequencies)
        object.__setattr__(self, "spike_mask", spike)
        object.__setattr__(self, "bulk_mask", bulk)
        object.__setattr__(self, "min_selection_frequency", threshold)

    @classmethod
    def unavailable(
        cls,
        singular_values: Sequence[float] | np.ndarray,
        *,
        status: ComponentSplitStatus,
        min_selection_frequency: float,
        reason: str,
    ) -> "SpectralComponentSplitContract":
        singular = np.asarray(singular_values, dtype=float).reshape(-1)
        return cls(
            status=status,
            singular_values=singular,
            empirical_bulk_edge=None,
            selection_frequencies=np.zeros(singular.size, dtype=float),
            spike_mask=np.zeros(singular.size, dtype=bool),
            bulk_mask=np.ones(singular.size, dtype=bool),
            min_selection_frequency=min_selection_frequency,
            frequency_source="unavailable",
            reason=reason,
        )

    @property
    def spike_indices(self) -> np.ndarray:
        indices = np.flatnonzero(self.spike_mask)
        indices.setflags(write=False)
        return indices

    @property
    def n_spikes(self) -> int:
        return int(np.sum(self.spike_mask))

    def to_dict(self, *, include_arrays: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "status": self.status.value,
            "empirical_bulk_edge": self.empirical_bulk_edge,
            "n_components": int(self.singular_values.size),
            "n_spikes": self.n_spikes,
            "spike_indices": self.spike_indices.tolist(),
            "min_selection_frequency": float(self.min_selection_frequency),
            "frequency_source": self.frequency_source,
            "reason": self.reason,
        }
        if include_arrays:
            payload.update(
                {
                    "singular_values": self.singular_values.tolist(),
                    "selection_frequencies": self.selection_frequencies.tolist(),
                    "spike_mask": self.spike_mask.tolist(),
                    "bulk_mask": self.bulk_mask.tolist(),
                }
            )
        return payload


@dataclass(frozen=True)
class RowSpectralParticipationContract:
    """Component-wise energy carried by each observed row."""

    leverage: np.ndarray
    spike_energy: np.ndarray
    bulk_energy: np.ndarray
    signalness: np.ndarray
    spike_signatures: np.ndarray

    def __post_init__(self) -> None:
        leverage = _readonly_vector(self.leverage, dtype=float)
        spike_energy = _readonly_vector(self.spike_energy, dtype=float)
        bulk_energy = _readonly_vector(self.bulk_energy, dtype=float)
        signalness = _readonly_vector(self.signalness, dtype=float)
        signatures = _readonly_matrix(self.spike_signatures, dtype=float)
        row_counts = {
            leverage.size,
            spike_energy.size,
            bulk_energy.size,
            signalness.size,
            signatures.shape[0],
        }
        if len(row_counts) != 1:
            raise ValueError("row participation arrays must align")
        for values in (leverage, spike_energy, bulk_energy, signalness, signatures):
            if not np.all(np.isfinite(values)) or np.any(values < -1e-12):
                raise ValueError("row participation values must be finite and non-negative")
        if np.any(signalness > 1.0 + 1e-12):
            raise ValueError("signalness must be in [0, 1]")
        if signatures.shape[1]:
            row_sums = np.sum(signatures, axis=1)
            active = spike_energy > 0.0
            if np.any(active) and not np.allclose(row_sums[active], 1.0):
                raise ValueError("active spike signatures must sum to one")
        object.__setattr__(self, "leverage", leverage)
        object.__setattr__(self, "spike_energy", spike_energy)
        object.__setattr__(self, "bulk_energy", bulk_energy)
        object.__setattr__(self, "signalness", np.clip(signalness, 0.0, 1.0))
        object.__setattr__(self, "spike_signatures", signatures)

    @property
    def n_rows(self) -> int:
        return int(self.signalness.size)

    def to_dict(self, *, include_arrays: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "n_rows": self.n_rows,
            "n_spike_components": int(self.spike_signatures.shape[1]),
            "mean_signalness": (
                float(np.mean(self.signalness)) if self.n_rows else 0.0
            ),
            "median_signalness": (
                float(np.median(self.signalness)) if self.n_rows else 0.0
            ),
            "high_signal_fraction": (
                float(np.mean(self.signalness >= 0.5)) if self.n_rows else 0.0
            ),
            "total_spike_energy": float(np.sum(self.spike_energy)),
            "total_bulk_energy": float(np.sum(self.bulk_energy)),
        }
        if include_arrays:
            payload.update(
                {
                    "leverage": self.leverage.tolist(),
                    "spike_energy": self.spike_energy.tolist(),
                    "bulk_energy": self.bulk_energy.tolist(),
                    "signalness": self.signalness.tolist(),
                    "spike_signatures": self.spike_signatures.tolist(),
                }
            )
        return payload


@dataclass(frozen=True)
class BulkSpikeTrainingPlan:
    """Exact unique-row budget allocated to disjoint bulk and spike regimes."""

    policy: BulkSpikeBudgetPolicy
    total_budget: int
    requested_spike_share: float
    resolved_spike_share: float
    bulk_candidate_indices: np.ndarray
    spike_candidate_indices: np.ndarray
    bulk_selected_indices: np.ndarray
    spike_selected_indices: np.ndarray

    def __post_init__(self) -> None:
        names = (
            "bulk_candidate_indices",
            "spike_candidate_indices",
            "bulk_selected_indices",
            "spike_selected_indices",
        )
        arrays = {
            name: _readonly_vector(getattr(self, name), dtype=int) for name in names
        }
        for name, values in arrays.items():
            if np.unique(values).size != values.size:
                raise ValueError(f"{name} must contain unique row indices")
            object.__setattr__(self, name, values)
        if np.intersect1d(
            arrays["bulk_candidate_indices"], arrays["spike_candidate_indices"]
        ).size:
            raise ValueError("bulk and spike candidates must be disjoint")
        selected = np.concatenate(
            [arrays["bulk_selected_indices"], arrays["spike_selected_indices"]]
        )
        if selected.size != int(self.total_budget):
            raise ValueError("selected rows must match the exact total budget")
        if np.unique(selected).size != selected.size:
            raise ValueError("selected rows must be globally unique")
        if not np.all(
            np.isin(
                arrays["bulk_selected_indices"], arrays["bulk_candidate_indices"]
            )
        ) or not np.all(
            np.isin(
                arrays["spike_selected_indices"], arrays["spike_candidate_indices"]
            )
        ):
            raise ValueError("selected rows must belong to their regime candidates")
        for share in (self.requested_spike_share, self.resolved_spike_share):
            if not 0.0 <= float(share) <= 1.0:
                raise ValueError("spike shares must be in [0, 1]")

    @property
    def selected_indices(self) -> np.ndarray:
        values = np.concatenate(
            [self.bulk_selected_indices, self.spike_selected_indices]
        )
        values.setflags(write=False)
        return values

    def to_dict(self, *, include_arrays: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "policy": self.policy.value,
            "total_budget": int(self.total_budget),
            "requested_spike_share": float(self.requested_spike_share),
            "resolved_spike_share": float(self.resolved_spike_share),
            "bulk_candidate_count": int(self.bulk_candidate_indices.size),
            "spike_candidate_count": int(self.spike_candidate_indices.size),
            "bulk_selected_count": int(self.bulk_selected_indices.size),
            "spike_selected_count": int(self.spike_selected_indices.size),
            "exact_budget": int(self.selected_indices.size) == int(self.total_budget),
        }
        if include_arrays:
            payload.update(
                {
                    "bulk_candidate_indices": self.bulk_candidate_indices.tolist(),
                    "spike_candidate_indices": self.spike_candidate_indices.tolist(),
                    "bulk_selected_indices": self.bulk_selected_indices.tolist(),
                    "spike_selected_indices": self.spike_selected_indices.tolist(),
                }
            )
        return payload


def build_spectral_component_split(
    singular_values: Sequence[float] | np.ndarray,
    null_result: SpectralNullDiagnosticResult,
    *,
    min_selection_frequency: float,
) -> SpectralComponentSplitContract:
    """Build the stable spike mask from an empirical null edge."""

    singular = np.asarray(singular_values, dtype=float).reshape(-1)
    threshold = float(min_selection_frequency)
    if null_result.status is NullDiagnosticStatus.DISABLED:
        return SpectralComponentSplitContract.unavailable(
            singular,
            status=ComponentSplitStatus.DISABLED,
            min_selection_frequency=threshold,
            reason="spectral null diagnostics are disabled",
        )
    primary = null_result.primary_result
    if primary is None or primary.empirical_bulk_edge is None:
        return SpectralComponentSplitContract.unavailable(
            singular,
            status=ComponentSplitStatus.NULL_DIAGNOSTIC_UNAVAILABLE,
            min_selection_frequency=threshold,
            reason="empirical bulk edge is unavailable",
        )
    view_result = null_result.policy_result(NullModelPolicy.VIEW_RESAMPLING)
    frequency_source = null_result.primary_policy.value
    frequencies = np.asarray(primary.selection_frequency, dtype=float)
    if view_result is not None and len(view_result.selection_frequency) > 0:
        frequencies = np.asarray(view_result.selection_frequency, dtype=float)
        frequency_source = NullModelPolicy.VIEW_RESAMPLING.value
    width = min(singular.size, frequencies.size)
    aligned_frequencies = np.zeros(singular.size, dtype=float)
    aligned_frequencies[:width] = frequencies[:width]
    edge = float(primary.empirical_bulk_edge)
    spike_mask = (singular > edge) & (aligned_frequencies >= threshold)
    status = (
        ComponentSplitStatus.OK
        if np.any(spike_mask)
        else ComponentSplitStatus.NO_STABLE_SPIKES
    )
    reason = (
        "stable components exceed the empirical null edge"
        if status is ComponentSplitStatus.OK
        else "no component jointly passed the edge and stability criteria"
    )
    return SpectralComponentSplitContract(
        status=status,
        singular_values=singular,
        empirical_bulk_edge=edge,
        selection_frequencies=aligned_frequencies,
        spike_mask=spike_mask,
        bulk_mask=~spike_mask,
        min_selection_frequency=threshold,
        frequency_source=frequency_source,
        reason=reason,
    )


def compute_row_spectral_participation(
    left_basis: np.ndarray,
    split: SpectralComponentSplitContract,
    *,
    epsilon: float = 1e-12,
) -> RowSpectralParticipationContract:
    """Compute spike/bulk energy and normalized spike signatures per row."""

    basis = np.asarray(left_basis, dtype=float)
    if basis.ndim != 2 or basis.shape[1] != split.singular_values.size:
        raise ValueError("left_basis columns must align with the component split")
    if not np.all(np.isfinite(basis)):
        raise ValueError("left_basis must contain only finite values")
    squared_basis = np.square(basis)
    squared_singular = np.square(split.singular_values)
    edge_squared = float(split.empirical_bulk_edge or 0.0) ** 2
    spike_weights = np.zeros_like(squared_singular)
    spike_weights[split.spike_mask] = np.maximum(
        squared_singular[split.spike_mask] - edge_squared,
        0.0,
    )
    bulk_weights = np.zeros_like(squared_singular)
    bulk_weights[split.bulk_mask] = squared_singular[split.bulk_mask]
    spike_contributions = squared_basis[:, split.spike_mask] * spike_weights[
        split.spike_mask
    ]
    spike_energy = np.sum(spike_contributions, axis=1)
    bulk_energy = squared_basis @ bulk_weights
    total = spike_energy + bulk_energy
    signalness = np.divide(
        spike_energy,
        total + float(epsilon),
        out=np.zeros_like(spike_energy),
        where=total > 0.0,
    )
    signatures = np.divide(
        spike_contributions,
        spike_energy.reshape(-1, 1) + float(epsilon),
        out=np.zeros_like(spike_contributions),
        where=spike_energy.reshape(-1, 1) > 0.0,
    )
    return RowSpectralParticipationContract(
        leverage=np.sum(squared_basis, axis=1),
        spike_energy=spike_energy,
        bulk_energy=bulk_energy,
        signalness=signalness,
        spike_signatures=signatures,
    )


def build_bulk_spike_training_plan(
    participation: RowSpectralParticipationContract,
    *,
    total_budget: int,
    policy: str | BulkSpikeBudgetPolicy = BulkSpikeBudgetPolicy.EXCESS_ENERGY,
    fixed_spike_share: float = 0.25,
    min_spike_share: float = 0.10,
    max_spike_share: float = 0.50,
    signal_threshold: float = 0.50,
    random_state: int | np.random.Generator | None = None,
) -> BulkSpikeTrainingPlan:
    """Allocate and sample an exact unique-row budget across two regimes."""

    resolved_policy = (
        policy if isinstance(policy, BulkSpikeBudgetPolicy) else BulkSpikeBudgetPolicy(str(policy))
    )
    n_rows = participation.n_rows
    budget = max(0, min(int(total_budget), n_rows))
    if budget < 1:
        raise ValueError("total_budget must select at least one row")
    if not 0.0 <= float(min_spike_share) <= float(max_spike_share) <= 1.0:
        raise ValueError("spike-share limits must satisfy 0 <= min <= max <= 1")
    if not 0.0 <= float(signal_threshold) <= 1.0:
        raise ValueError("signal_threshold must be in [0, 1]")

    total_energy = float(
        np.sum(participation.spike_energy) + np.sum(participation.bulk_energy)
    )
    energy_share = (
        float(np.sum(participation.spike_energy)) / total_energy
        if total_energy > 0.0
        else 0.0
    )
    requested_share = (
        energy_share
        if resolved_policy is BulkSpikeBudgetPolicy.EXCESS_ENERGY
        else float(fixed_spike_share)
    )
    requested_share = float(
        np.clip(requested_share, min_spike_share, max_spike_share)
    )
    all_indices = np.arange(n_rows, dtype=int)
    spike_candidates = all_indices[participation.signalness >= signal_threshold]
    bulk_candidates = all_indices[participation.signalness < signal_threshold]
    if not np.any(participation.spike_energy > 0.0) or spike_candidates.size == 0:
        spike_candidates = np.asarray([], dtype=int)
        bulk_candidates = all_indices
        requested_share = 0.0
    if bulk_candidates.size == 0:
        bulk_candidates = np.asarray([], dtype=int)
        spike_candidates = all_indices
        requested_share = 1.0

    spike_budget = int(round(budget * requested_share))
    spike_budget = min(spike_budget, spike_candidates.size)
    bulk_budget = min(budget - spike_budget, bulk_candidates.size)
    remaining = budget - spike_budget - bulk_budget
    if remaining:
        spike_room = spike_candidates.size - spike_budget
        add_spike = min(remaining, spike_room)
        spike_budget += add_spike
        remaining -= add_spike
    if remaining:
        bulk_room = bulk_candidates.size - bulk_budget
        add_bulk = min(remaining, bulk_room)
        bulk_budget += add_bulk
        remaining -= add_bulk
    if remaining:
        raise RuntimeError("unable to allocate the exact bulk/spike budget")

    rng = (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )
    spike_plan = build_exact_budget_sketch_plan(
        spike_candidates,
        target_size=spike_budget,
        policy="saturated_leverage",
        scores=np.maximum(participation.signalness, 1e-12),
        random_state=rng,
    )
    bulk_plan = build_exact_budget_sketch_plan(
        bulk_candidates,
        target_size=bulk_budget,
        policy="saturated_leverage",
        scores=np.maximum(1.0 - participation.signalness, 1e-12),
        random_state=rng,
    )
    return BulkSpikeTrainingPlan(
        policy=resolved_policy,
        total_budget=budget,
        requested_spike_share=requested_share,
        resolved_spike_share=float(spike_budget / budget),
        bulk_candidate_indices=bulk_candidates,
        spike_candidate_indices=spike_candidates,
        bulk_selected_indices=bulk_plan.selected_indices,
        spike_selected_indices=spike_plan.selected_indices,
    )
