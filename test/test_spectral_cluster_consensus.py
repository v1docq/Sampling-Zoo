from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse
from sklearn.metrics import adjusted_rand_score

from sampling_zoo.core.sampling_strategies.spectral.cluster_consensus import (
    build_weighted_membership_embedding,
)
from sampling_zoo.core.sampling_strategies.spectral.cluster_selection import (
    ClusterSelectionResult,
    SpectralClusterSelector,
)
from sampling_zoo.core.sampling_strategies.spectral.cluster_selection_contracts import (
    build_cluster_candidate_plan,
    build_cluster_consensus_plan,
)


def test_consensus_plan_preserves_weighted_k_vote() -> None:
    plan = build_cluster_consensus_plan(
        algorithms=("kmeans", "gmm", "hdbscan"),
        n_clusters=(2, 3, 3),
        scores=(0.5, 0.5, 0.5),
        vote_temperature=1.0,
    )

    assert plan.selected_n_clusters == 3
    assert sum(source.weight for source in plan.sources) == pytest.approx(1.0)
    assert dict(plan.votes_by_n_clusters) == pytest.approx({2: 1 / 3, 3: 2 / 3})
    assert plan.representation_columns == 8
    assert len({source.key for source in plan.sources}) == 3


def test_selection_result_preserves_legacy_positional_constructor() -> None:
    candidate_plan = build_cluster_candidate_plan(
        n_samples=4,
        algorithms=("kmeans",),
        min_partitions=2,
        max_partitions=2,
        min_auto_partition_size=1,
    )

    result = ClusterSelectionResult(
        np.asarray([0, 0, 1, 1]),
        np.asarray([[0.0], [1.0]]),
        None,
        "kmeans",
        2,
        (),
        candidate_plan,
        (),
        {},
    )

    assert result.consensus_plan is None
    assert result.consensus_candidates == ()


def test_weighted_membership_gram_is_exact_coassociation() -> None:
    labels = (
        np.asarray([0, 0, 1, 1]),
        np.asarray([1, 1, 0, 0]),
        np.asarray([0, 1, 0, 1]),
    )
    plan = build_cluster_consensus_plan(
        algorithms=("kmeans", "gmm", "hdbscan"),
        n_clusters=(2, 2, 2),
        scores=(0.8, 0.4, 0.1),
        vote_temperature=0.5,
    )

    membership = build_weighted_membership_embedding(labels, plan)
    observed = (membership @ membership.T).toarray()
    expected = sum(
        source.weight * (source_labels[:, None] == source_labels[None, :])
        for source_labels, source in zip(labels, plan.sources)
    )

    assert sparse.isspmatrix_csr(membership)
    assert membership.shape == (4, 6)
    assert membership.nnz == 12
    assert observed == pytest.approx(expected)


def test_consensus_gram_is_invariant_to_cluster_label_permutations() -> None:
    labels = (
        np.asarray([0, 0, 1, 1, 2, 2]),
        np.asarray([0, 1, 0, 1, 2, 2]),
    )
    permuted = (
        np.asarray([7, 7, 3, 3, 9, 9]),
        np.asarray([5, 8, 5, 8, 2, 2]),
    )
    plan = build_cluster_consensus_plan(
        algorithms=("kmeans", "gmm"),
        n_clusters=(3, 3),
        scores=(0.7, 0.5),
        vote_temperature=1.0,
    )

    first = build_weighted_membership_embedding(labels, plan)
    second = build_weighted_membership_embedding(permuted, plan)
    reordered_plan = build_cluster_consensus_plan(
        algorithms=("gmm", "kmeans"),
        n_clusters=(3, 3),
        scores=(0.5, 0.7),
        vote_temperature=1.0,
    )
    reordered = build_weighted_membership_embedding(
        tuple(reversed(labels)),
        reordered_plan,
    )

    assert (first @ first.T).toarray() == pytest.approx((second @ second.T).toarray())
    assert (first @ first.T).toarray() == pytest.approx(
        (reordered @ reordered.T).toarray()
    )


def test_membership_storage_scales_linearly_with_rows_and_sources() -> None:
    n_samples = 10_000
    counts = (2, 3, 5)
    labels = tuple(np.arange(n_samples) % count for count in counts)
    plan = build_cluster_consensus_plan(
        algorithms=("kmeans", "gmm", "hdbscan"),
        n_clusters=counts,
        scores=(0.8, 0.6, 0.4),
        vote_temperature=1.0,
    )

    membership = build_weighted_membership_embedding(labels, plan)

    assert membership.shape == (n_samples, sum(counts))
    assert membership.nnz == n_samples * len(counts)


def test_selector_builds_consensus_labels_without_dense_coassociation() -> None:
    rng = np.random.default_rng(123)
    true_labels = np.repeat(np.arange(4), 30)
    centers = np.asarray(
        [
            [-4.0, -4.0],
            [-4.0, 4.0],
            [4.0, -4.0],
            [4.0, 4.0],
        ]
    )
    embedding = np.vstack(
        [center + rng.normal(scale=0.25, size=(30, 2)) for center in centers]
    )
    selector = SpectralClusterSelector(
        algorithms=("kmeans", "gmm"),
        ensemble_method="coassociation",
        min_partitions=2,
        max_partitions=5,
        min_auto_partition_size=1,
        vote_temperature=1.0,
        random_state=11,
        show_progress=False,
    )

    result = selector.select(embedding)
    consensus = result.diagnostics["consensus"]

    assert result.selected_algorithm == "coassociation_consensus"
    assert result.selected_n_clusters == 4
    assert adjusted_rand_score(true_labels, result.labels) == pytest.approx(1.0)
    assert result.consensus_plan is not None
    assert len(result.consensus_candidates) == 1
    assert consensus["fallback_to_source_candidate"] is False
    assert consensus["representation_shape"] == [120, 28]
    assert consensus["representation_nnz"] == 120 * 8
    assert consensus["score_delta_vs_best_source"] == pytest.approx(0.0)
    assert result.diagnostics["selected_candidate"]["algorithm"] == (
        "coassociation_consensus"
    )


def test_membership_embedding_rejects_misaligned_sources() -> None:
    plan = build_cluster_consensus_plan(
        algorithms=("kmeans", "gmm"),
        n_clusters=(2, 2),
        scores=(1.0, 0.5),
        vote_temperature=1.0,
    )

    with pytest.raises(ValueError, match="equal lengths"):
        build_weighted_membership_embedding(
            (np.asarray([0, 0, 1]), np.asarray([0, 1])),
            plan,
        )
