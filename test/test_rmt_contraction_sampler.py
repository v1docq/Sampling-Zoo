from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

from sampling_zoo.core.sampling_strategies.spectral.backend.matrix_backend import MatrixRMTBackend
from sampling_zoo.core.sampling_strategies.spectral.backend.tensor_backend import TensorRMTBackend
from sampling_zoo.core.sampling_strategies.spectral.rmt_contraction_sampler import (
    RMTContractionConfig,
    RMTContractionTensorSampler,
)


def _frame(n_samples: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    X = pd.DataFrame(
        rng.normal(size=(n_samples, 6)),
        columns=[f"x_{idx}" for idx in range(6)],
    )
    X["cat"] = pd.Series([f"c_{idx % 12}" for idx in range(n_samples)], dtype="string")
    return X


def test_numpy_backend_predict_partition_proba_rows_sum_to_one() -> None:
    X = _frame()
    sampler = RMTContractionTensorSampler(
        n_partitions=4,
        n_views=5,
        projection_dim=3,
        max_encoded_features=10,
        backend="numpy",
        random_state=7,
        show_progress=False,
    )

    sampler.fit(X)
    proba = sampler.predict_partition_proba(X.iloc[:10])

    assert proba.shape == (10, len(sampler.partition_names_))
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert sampler.diagnostics_["backend"] == "numpy"
    assert sampler.diagnostics_["selected_rank"] <= sampler.diagnostics_["initial_rank"]
    assert "approx_rank" not in sampler.diagnostics_


def test_auto_n_views_subsample_uses_feature_coverage_policy() -> None:
    rng = np.random.default_rng(123)
    X = pd.DataFrame(rng.normal(size=(50, 20)), columns=[f"x_{idx}" for idx in range(20)])
    sampler = RMTContractionTensorSampler(
        n_partitions=3,
        n_views="auto",
        view_strategy="subsample",
        view_size=4,
        projection_dim=2,
        target_feature_coverage=0.9,
        min_views=2,
        max_views=50,
        backend="numpy",
        random_state=23,
        show_progress=False,
    )

    sampler.fit(X)

    expected_views = int(np.ceil(np.log(1.0 - 0.9) / np.log(1.0 - 4 / 20)))
    assert sampler.diagnostics_["n_views_policy"] == "coverage"
    assert sampler.diagnostics_["n_views"] == expected_views
    assert sampler.diagnostics_["estimated_feature_coverage"] >= 0.9


def test_auto_n_views_gaussian_uses_spectrum_stability_policy() -> None:
    rng = np.random.default_rng(321)
    X = pd.DataFrame(rng.normal(size=(36, 8)), columns=[f"x_{idx}" for idx in range(8)])
    sampler = RMTContractionTensorSampler(
        n_partitions=3,
        n_views="auto",
        view_strategy="gaussian",
        projection_dim=2,
        min_views=2,
        max_views=4,
        spectrum_stability_tolerance=10.0,
        backend="numpy",
        random_state=29,
        show_progress=False,
    )

    sampler.fit(X)

    assert sampler.diagnostics_["n_views_policy"] == "spectrum_stability"
    assert sampler.diagnostics_["n_views"] in {2, 4}
    assert sampler.diagnostics_["spectrum_stability_candidates"] == [2, 4]


def test_sv_scaled_embedding_mode_changes_embedding_geometry() -> None:
    rng = np.random.default_rng(222)
    X = pd.DataFrame(rng.normal(size=(48, 7)), columns=[f"x_{idx}" for idx in range(7)])
    shared_kwargs = dict(
        n_partitions=3,
        n_views=3,
        projection_dim=3,
        backend="numpy",
        random_state=31,
        show_progress=False,
    )
    scaled = RMTContractionTensorSampler(embedding_mode="sv_scaled", **shared_kwargs).fit(X)
    whitened = RMTContractionTensorSampler(embedding_mode="whitened", **shared_kwargs).fit(X)

    assert scaled.diagnostics_["embedding_mode"] == "sv_scaled"
    assert whitened.diagnostics_["embedding_mode"] == "whitened"
    assert scaled.sample_embedding_.shape == whitened.sample_embedding_.shape
    assert not np.allclose(scaled.sample_embedding_, whitened.sample_embedding_)


def test_auto_partition_selection_records_diagnostics() -> None:
    rng = np.random.default_rng(333)
    centers = np.asarray([
        [-3.0, 0.0, 0.0, 0.0],
        [0.0, 3.0, 0.0, 0.0],
        [3.0, 0.0, 0.0, 0.0],
    ])
    X = np.vstack([
        center + rng.normal(scale=0.2, size=(24, centers.shape[1]))
        for center in centers
    ])
    frame = pd.DataFrame(X, columns=[f"x_{idx}" for idx in range(X.shape[1])])
    sampler = RMTContractionTensorSampler(
        n_partitions=2,
        partition_selection_method="auto",
        min_partitions=2,
        max_partitions=4,
        min_auto_partition_size=5,
        partition_selection_sample_size=100,
        n_views=3,
        projection_dim=2,
        backend="numpy",
        random_state=37,
        show_progress=False,
    )

    sampler.fit(frame)

    assert sampler.diagnostics_["partition_selection_method"] == "auto"
    assert sampler.diagnostics_["selected_cluster_algorithm"] == "kmeans"
    assert sampler.diagnostics_["cluster_selection_metric"] == "balanced_silhouette"
    assert 2 <= sampler.diagnostics_["selected_n_partitions"] <= 4
    assert sampler.diagnostics_["partition_selection_candidates"] == [2, 3, 4]
    assert sampler.diagnostics_["partition_selection_candidate_details"]
    assert sampler.diagnostics_["partition_selection_candidate_plan"]["eligible_count_candidates"] == [2, 3, 4]
    assert sampler.diagnostics_["partition_selection_candidate_failures"] == []
    assert len(sampler.partitions) == sampler.diagnostics_["selected_n_partitions"]


def test_auto_partition_selection_skips_unavailable_optional_algorithms() -> None:
    X = _frame(n_samples=48).select_dtypes(include=[np.number])
    sampler = RMTContractionTensorSampler(
        n_partitions=3,
        partition_selection_method="auto",
        cluster_algorithms=("kmeans", "bisecting_kmeans", "gmm", "hdbscan"),
        cluster_ensemble_method="weighted_vote",
        min_partitions=2,
        max_partitions=4,
        min_auto_partition_size=1,
        n_views=2,
        projection_dim=2,
        backend="numpy",
        random_state=41,
        show_progress=False,
    )

    sampler.fit(X)

    assert sampler.diagnostics_["selected_cluster_algorithm"] in {
        "kmeans",
        "bisecting_kmeans",
        "gmm",
        "hdbscan",
    }
    assert "partition_selection_candidate_plan" in sampler.diagnostics_
    assert "partition_selection_candidate_failures" in sampler.diagnostics_
    assert sampler.predict_partition_proba(X.iloc[:5]).shape[1] == len(sampler.partition_names_)


def test_config_constructor_and_legacy_positional_arguments() -> None:
    config = RMTContractionConfig(
        n_partitions=2,
        n_views=3,
        projection_dim=2,
        backend="numpy",
        show_progress=False,
    )

    from_config = RMTContractionTensorSampler(config=config, n_partitions=4)
    from_positionals = RMTContractionTensorSampler(3, 4, projection_dim=2, backend="numpy", show_progress=False)

    assert from_config.n_partitions == 4
    assert from_config.n_views == 3
    assert from_config.backend == "numpy"
    assert from_positionals.n_partitions == 3
    assert from_positionals.n_views == 4


def test_removed_approx_rank_is_rejected() -> None:
    with pytest.raises(ValueError, match="approx_rank is no longer supported"):
        RMTContractionTensorSampler(approx_rank=3)


def test_explained_variance_rank_selection() -> None:
    sampler = RMTContractionTensorSampler(backend="numpy", show_progress=False)

    rank, explained_variance = sampler._select_rank_from_spectrum(np.asarray([4.0, 3.0, 1.0]))

    assert rank == 2
    assert explained_variance >= 0.95


def test_zero_spectrum_rank_selection_uses_min_rank() -> None:
    sampler = RMTContractionTensorSampler(min_rank=2, backend="numpy", show_progress=False)

    rank, explained_variance = sampler._select_rank_from_spectrum(np.zeros(5))

    assert rank == 2
    assert explained_variance == 0.0


def test_null_diagnostic_is_opt_in_and_does_not_change_selected_rank() -> None:
    X = _frame(n_samples=48).select_dtypes(include=[np.number])
    shared = dict(
        n_partitions=3,
        n_views=3,
        projection_dim=2,
        initial_rank_fraction=0.5,
        backend="numpy",
        random_state=43,
        show_progress=False,
    )

    baseline = RMTContractionTensorSampler(**shared).fit(X)
    diagnosed = RMTContractionTensorSampler(
        null_diagnostic_enabled=True,
        null_resamples=2,
        **shared,
    ).fit(X)

    assert baseline.diagnostics_["null_model_status"] == "disabled"
    assert diagnosed.diagnostics_["null_model_status"] == "ok"
    assert diagnosed.diagnostics_["null_primary_policy"] == "feature_permutation"
    assert diagnosed.diagnostics_["null_successful_resamples"] == 6
    assert diagnosed.diagnostics_["null_empirical_bulk_edge"] is not None
    assert diagnosed.diagnostics_["rank_by_null_edge"] is not None
    assert diagnosed.diagnostics_["rank_by_stability"] is not None
    assert diagnosed.diagnostics_["selected_rank_reason"] == "explained_variance"
    assert diagnosed.diagnostics_["selected_rank"] == baseline.diagnostics_["selected_rank"]
    assert np.allclose(
        diagnosed.diagnostics_["singular_values"],
        baseline.diagnostics_["singular_values"],
    )
    assert len(diagnosed.diagnostics_["initial_singular_values"]) >= len(
        diagnosed.diagnostics_["singular_values"]
    )


def test_invalid_null_diagnostic_config_is_rejected() -> None:
    with pytest.raises(ValueError, match="null_primary_policy must be included"):
        RMTContractionTensorSampler(
            null_model_policies=("moment_matched_gaussian",),
            null_primary_policy="feature_permutation",
        )

    with pytest.raises(ValueError, match="null_resamples must be at least 2"):
        RMTContractionTensorSampler(null_resamples=1)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_backend_runs_spectral_null_diagnostic() -> None:
    X = _frame(n_samples=32).select_dtypes(include=[np.number])
    sampler = RMTContractionTensorSampler(
        n_partitions=2,
        n_views=2,
        projection_dim=2,
        backend="torch",
        device="cpu",
        null_diagnostic_enabled=True,
        null_model_policies=("feature_permutation", "view_resampling"),
        null_resamples=2,
        random_state=47,
        show_progress=False,
    )

    sampler.fit(X)

    assert sampler.diagnostics_["backend"] == "torch"
    assert sampler.diagnostics_["null_model_status"] == "ok"
    assert sampler.diagnostics_["null_successful_resamples"] == 4


def test_subspace_diagnostic_is_opt_in_and_does_not_change_sampling() -> None:
    X = _frame(n_samples=48).select_dtypes(include=[np.number])
    shared = dict(
        n_partitions=3,
        n_views=3,
        projection_dim=2,
        backend="numpy",
        random_state=53,
        show_progress=False,
    )

    baseline = RMTContractionTensorSampler(**shared).fit(X)
    diagnosed = RMTContractionTensorSampler(
        subspace_diagnostic_enabled=True,
        subspace_resamples=2,
        **shared,
    ).fit(X)

    assert baseline.diagnostics_["subspace_stability_status"] == "disabled"
    assert diagnosed.diagnostics_["subspace_stability_status"] == "ok"
    assert diagnosed.diagnostics_["subspace_successful_resamples"] == 2
    assert diagnosed.diagnostics_["subspace_comparison_rank"] == (
        diagnosed.diagnostics_["selected_rank"]
    )
    assert diagnosed.diagnostics_["rank_by_subspace_stability"] is not None
    assert diagnosed.diagnostics_["selected_rank_reason"] == "explained_variance"
    assert diagnosed.diagnostics_["selected_rank"] == baseline.diagnostics_["selected_rank"]
    assert np.allclose(
        diagnosed.diagnostics_["singular_values"],
        baseline.diagnostics_["singular_values"],
    )
    assert all(
        np.array_equal(
            diagnosed.partitions[name],
            baseline.partitions[name],
        )
        for name in baseline.partition_names_
    )


def test_invalid_subspace_diagnostic_config_is_rejected() -> None:
    with pytest.raises(ValueError, match="subspace_resamples must be at least 2"):
        RMTContractionTensorSampler(subspace_resamples=1)

    with pytest.raises(
        ValueError,
        match="subspace_max_principal_angle_degrees",
    ):
        RMTContractionTensorSampler(
            subspace_max_principal_angle_degrees=91.0
        )


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_backend_runs_subspace_stability_diagnostic() -> None:
    X = _frame(n_samples=32).select_dtypes(include=[np.number])
    sampler = RMTContractionTensorSampler(
        n_partitions=2,
        n_views=2,
        projection_dim=2,
        backend="torch",
        device="cpu",
        subspace_diagnostic_enabled=True,
        subspace_resamples=2,
        random_state=59,
        show_progress=False,
    ).fit(X)

    assert sampler.diagnostics_["backend"] == "torch"
    assert sampler.diagnostics_["subspace_stability_status"] == "ok"
    assert sampler.diagnostics_["subspace_successful_resamples"] == 2


def test_matrix_rmt_backend_returns_expected_shapes() -> None:
    X = np.random.default_rng(123).normal(size=(40, 6))
    sampler = RMTContractionTensorSampler(
        n_views=2,
        projection_dim=3,
        backend="numpy",
        show_progress=False,
        random_state=42,
    )
    specs = sampler._make_view_specs(n_features=X.shape[1], rng=np.random.default_rng(42))
    backend = MatrixRMTBackend(oversample_factor=2, power_iterations=1, random_state=42)

    M = backend.build_mode0_unfolding(X, specs)
    basis = backend.compute_spectral_basis(M, rank=2)

    assert M.shape == (40, 6)
    assert basis.U.shape == (40, 2)
    assert basis.singular_values.shape == (2,)
    assert basis.Vt.shape == (2, 6)
    assert np.isclose(basis.leverage_scores.sum(), 1.0)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_tensor_rmt_backend_routing_probability_rows_sum_to_one() -> None:
    backend = TensorRMTBackend(device="cpu", dtype="float32", random_state=42)
    embedding = np.asarray([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]], dtype=np.float32)
    centroids = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)

    proba = backend.routing_probability(embedding, centroids, temperature=1.0)

    assert proba.shape == (3, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_sampler_no_longer_owns_torch_randomized_svd() -> None:
    assert not hasattr(RMTContractionTensorSampler, "_torch_randomized_svd")


def test_auto_backend_resolves_to_available_backend() -> None:
    X = _frame()
    sampler = RMTContractionTensorSampler(
        n_partitions=3,
        n_views=4,
        projection_dim=2,
        backend="auto",
        random_state=11,
        show_progress=False,
    )

    sampler.fit(X)

    assert sampler.backend_ in {"numpy", "torch"}
    if importlib.util.find_spec("torch") is None:
        assert sampler.backend_ == "numpy"


def test_sparse_encoded_features_are_capped_before_dense_math() -> None:
    X = _frame(n_samples=120)
    sampler = RMTContractionTensorSampler(
        n_partitions=3,
        n_views=3,
        projection_dim=2,
        max_one_hot_cardinality=20,
        max_encoded_features=5,
        backend="numpy",
        random_state=13,
        show_progress=False,
    )

    sampler.fit(X)

    assert sampler.encoded_feature_subset_ is not None
    assert len(sampler.encoded_feature_subset_) == 5
    assert sampler.diagnostics_["raw_encoded_feature_count"] > sampler.diagnostics_["encoded_feature_count"]
    assert sampler.diagnostics_["encoded_feature_count"] == 5


def test_partition_probability_columns_align_with_partition_names() -> None:
    X = _frame()
    sampler = RMTContractionTensorSampler(
        n_partitions=5,
        n_views=4,
        projection_dim=3,
        chunks_percent=60,
        backend="numpy",
        random_state=17,
        show_progress=False,
    )

    sampler.fit(X)
    proba = sampler.predict_partition_proba(X.iloc[:7])

    assert proba.shape[1] == len(sampler.partition_names_)
    assert set(sampler.partition_names_) == set(sampler.partitions)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_backend_if_available() -> None:
    X = _frame()
    sampler = RMTContractionTensorSampler(
        n_partitions=3,
        n_views=4,
        projection_dim=2,
        backend="torch",
        device="cpu",
        random_state=19,
        show_progress=False,
    )

    sampler.fit(X)
    proba = sampler.predict_partition_proba(X.iloc[:5])

    assert sampler.backend_ == "torch"
    assert np.allclose(proba.sum(axis=1), 1.0)
