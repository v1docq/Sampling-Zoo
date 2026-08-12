from __future__ import annotations

import numpy as np
import pytest

from sampling_zoo.core.sampling_strategies.spectral.bulk_spike import (
    RowSpectralParticipationContract,
)
from sampling_zoo.core.sampling_strategies.spectral.bulk_spike_topology import (
    BulkSpikeExpertTopologySpec,
    BulkSpikeHierarchicalRouter,
    BulkSpikePartitionContract,
    BulkSpikeTopologyBuilder,
    BulkSpikeTopologyFoldScore,
    BulkSpikeTopologyMode,
    CrossFittedBulkSpikeTopologySelector,
)


def _participation() -> RowSpectralParticipationContract:
    signalness = np.concatenate(
        [np.full(60, 0.05), np.full(30, 0.90), np.full(30, 0.85)]
    )
    signatures = np.zeros((120, 2), dtype=float)
    signatures[:60] = 0.5
    signatures[60:90, 0] = 1.0
    signatures[90:, 1] = 1.0
    return RowSpectralParticipationContract(
        leverage=np.full(120, 0.1),
        spike_energy=signalness,
        bulk_energy=1.0 - signalness,
        signalness=signalness,
        spike_signatures=signatures,
    )


def test_single_and_multi_topologies_preserve_exact_unique_budget() -> None:
    builder = BulkSpikeTopologyBuilder()
    single = builder.build(
        BulkSpikeExpertTopologySpec(
            name="B1_bulk_single_spike",
            mode="single_spike",
            max_experts=5,
            min_partition_size=5,
        ),
        _participation(),
        total_budget=60,
        random_state=17,
    )
    multi = builder.build(
        BulkSpikeExpertTopologySpec(
            name="B2_bulk_multi_spike",
            mode="multi_spike",
            max_experts=5,
            min_partition_size=5,
        ),
        _participation(),
        total_budget=60,
        random_state=17,
    )

    for topology in (single, multi):
        assert topology.selected_indices.size == 60
        assert np.unique(topology.selected_indices).size == 60
        assert topology.n_experts <= 5
    assert single.regimes == ("bulk", "spike")
    assert multi.n_experts == 3
    assert all(
        len(rows) >= 5
        for regime, rows in zip(multi.regimes, multi.row_indices)
        if regime.startswith("spike")
    )


def test_standard_topology_rejects_hidden_row_repeats() -> None:
    spec = BulkSpikeExpertTopologySpec(name="B0", mode="standard")

    with pytest.raises(ValueError, match="cannot repeat"):
        BulkSpikeTopologyBuilder().build(
            spec,
            _participation(),
            total_budget=4,
            baseline_partitions={"chunk_0": [0, 1], "chunk_1": [1, 2]},
        )


def test_hierarchical_router_combines_gate_and_conditional_weights() -> None:
    topology = BulkSpikePartitionContract(
        topology_name="B2",
        mode=BulkSpikeTopologyMode.MULTI_SPIKE,
        partition_names=("bulk", "spike_0", "spike_1"),
        regimes=("bulk", "spike:0", "spike:1"),
        row_indices=(np.arange(4), np.arange(4, 7), np.arange(7, 10)),
        total_budget=10,
        max_experts=3,
    )
    routing = BulkSpikeHierarchicalRouter().route(
        topology,
        spike_probability=np.asarray([0.2, 0.8]),
        conditional_spike_weights=np.asarray([[0.75, 0.25], [0.10, 0.90]]),
    )

    assert np.allclose(routing.weights.sum(axis=1), 1.0)
    assert routing.weights[0].tolist() == pytest.approx([0.8, 0.15, 0.05])
    assert routing.weights[1].tolist() == pytest.approx([0.2, 0.08, 0.72])


def test_hierarchical_routing_is_equivariant_to_spike_expert_order() -> None:
    original = BulkSpikePartitionContract(
        topology_name="B2",
        mode=BulkSpikeTopologyMode.MULTI_SPIKE,
        partition_names=("bulk", "spike_0", "spike_1"),
        regimes=("bulk", "spike:0", "spike:1"),
        row_indices=(np.arange(4), np.arange(4, 7), np.arange(7, 10)),
        total_budget=10,
        max_experts=3,
    )
    permuted = BulkSpikePartitionContract(
        topology_name="B2",
        mode=BulkSpikeTopologyMode.MULTI_SPIKE,
        partition_names=("bulk", "spike_1", "spike_0"),
        regimes=("bulk", "spike:1", "spike:0"),
        row_indices=(np.arange(4), np.arange(7, 10), np.arange(4, 7)),
        total_budget=10,
        max_experts=3,
    )
    conditional = np.asarray([[0.3, 0.7], [0.8, 0.2]])
    router = BulkSpikeHierarchicalRouter()

    first = router.route(
        original,
        spike_probability=[0.4, 0.6],
        conditional_spike_weights=conditional,
    )
    second = router.route(
        permuted,
        spike_probability=[0.4, 0.6],
        conditional_spike_weights=conditional[:, ::-1],
    )

    assert np.allclose(first.weights, second.weights[:, [0, 2, 1]])


def test_cross_fitted_topology_selector_keeps_b0_or_selects_stable_b2() -> None:
    scores = []
    for fold in range(5):
        scores.extend(
            (
                BulkSpikeTopologyFoldScore("B0_standard_A9", str(fold), 10.0, "lower"),
                BulkSpikeTopologyFoldScore(
                    "B1_bulk_single_spike",
                    str(fold),
                    10.2,
                    "lower",
                ),
                BulkSpikeTopologyFoldScore(
                    "B2_bulk_multi_spike",
                    str(fold),
                    9.5,
                    "lower",
                ),
            )
        )
    selector = CrossFittedBulkSpikeTopologySelector(
        bootstrap_iterations=500,
        random_state=17,
    )

    result = selector.select(scores)

    assert result.status == "selected"
    assert result.selected_arm == "B2_bulk_multi_spike"
    assert result.positive_fold_fraction == 1.0
