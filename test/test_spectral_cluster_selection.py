import numpy as np

from sampling_zoo.core.sampling_strategies.spectral.cluster_selection import SpectralClusterSelector


def test_spectral_cluster_selector_builds_weighted_multi_algorithm_candidates() -> None:
    rng = np.random.default_rng(123)
    centers = np.asarray([
        [-4.0, -4.0],
        [-4.0, 4.0],
        [4.0, -4.0],
        [4.0, 4.0],
    ])
    embedding = np.vstack([
        center + rng.normal(scale=0.25, size=(30, 2))
        for center in centers
    ])

    selector = SpectralClusterSelector(
        algorithms=("kmeans", "gmm"),
        selection_metric="balanced_silhouette",
        ensemble_method="weighted_vote",
        min_partitions=2,
        max_partitions=5,
        min_auto_partition_size=1,
        max_cluster_imbalance_ratio=3.0,
        min_cluster_fraction=0.10,
        selection_sample_size=200,
        random_state=11,
        show_progress=False,
    )

    result = selector.select(embedding)

    assert result.labels.shape == (embedding.shape[0],)
    assert result.centers.shape[0] == result.selected_n_clusters
    assert result.selected_algorithm in {"kmeans", "gmm"}
    assert result.diagnostics["cluster_selection_metric"] == "balanced_silhouette"
    assert result.diagnostics["cluster_ensemble_method"] == "weighted_vote"
    assert {candidate.algorithm for candidate in result.candidates} == {"kmeans", "gmm"}
    assert any(candidate.valid for candidate in result.candidates)


def test_spectral_cluster_selector_counts_candidate_fits_for_progress() -> None:
    selector = SpectralClusterSelector(
        algorithms=("kmeans", "gmm", "hdbscan"),
        min_partitions=2,
        max_partitions=4,
        min_auto_partition_size=1,
        show_progress=False,
    )

    assert selector._candidate_fit_count([2, 3, 4]) == 7
