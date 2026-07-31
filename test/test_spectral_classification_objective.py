from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sampling_zoo.core.sampling_strategies.spectral.cluster_selection import (
    SpectralClusterSelector,
)
from sampling_zoo.core.sampling_strategies.spectral.cluster_selection_contracts import (
    ClusterConstraintViolation,
    evaluate_classification_partition_components,
    evaluate_cluster_score_components,
    score_cluster_components,
)
from sampling_zoo.core.sampling_strategies.spectral.rmt_contraction_sampler import (
    RMTContractionTensorSampler,
)
from sampling_zoo.core.utils.sampling_ensemble import SamplingEnsemble


def test_classification_profile_detects_pure_class_chunks() -> None:
    profile = evaluate_classification_partition_components(
        global_class_counts=(50, 50),
        cluster_class_counts=((50, 0), (0, 50)),
        class_labels=("negative", "positive"),
    )

    assert profile.missing_class_fraction == pytest.approx(0.5)
    assert profile.single_class_cluster_fraction == pytest.approx(1.0)
    assert profile.single_class_sample_fraction == pytest.approx(1.0)
    assert profile.class_distribution_drift == pytest.approx(0.5)
    assert profile.violations == (
        ClusterConstraintViolation.SINGLE_CLASS_CLUSTER,
    )
    assert profile.to_dict()["class_labels"] == ["negative", "positive"]


def test_classification_profile_is_zero_for_globally_balanced_chunks() -> None:
    profile = evaluate_classification_partition_components(
        global_class_counts=(50, 50),
        cluster_class_counts=((25, 25), (25, 25)),
    )

    assert profile.missing_class_fraction == 0.0
    assert profile.single_class_cluster_fraction == 0.0
    assert profile.single_class_sample_fraction == 0.0
    assert profile.class_distribution_drift == 0.0
    assert profile.violations == ()


def test_classification_profile_is_invariant_to_chunk_and_class_order() -> None:
    original = evaluate_classification_partition_components(
        global_class_counts=(50, 30, 20),
        cluster_class_counts=((30, 5, 5), (20, 25, 15)),
        class_labels=("a", "b", "c"),
    )
    permuted = evaluate_classification_partition_components(
        global_class_counts=(20, 50, 30),
        cluster_class_counts=((15, 20, 25), (5, 30, 5)),
        class_labels=("c", "a", "b"),
    )

    assert permuted.missing_class_fraction == original.missing_class_fraction
    assert (
        permuted.single_class_cluster_fraction
        == original.single_class_cluster_fraction
    )
    assert (
        permuted.single_class_sample_fraction
        == original.single_class_sample_fraction
    )
    assert permuted.class_distribution_drift == pytest.approx(
        original.class_distribution_drift
    )
    assert permuted.violations == original.violations


def test_balanced_score_applies_classification_penalties_and_hard_guard() -> None:
    classification = evaluate_classification_partition_components(
        global_class_counts=(50, 50),
        cluster_class_counts=((50, 0), (0, 50)),
    )
    components = evaluate_cluster_score_components(
        counts=(50, 50),
        silhouette=0.6,
        target_contrast=0.0,
        max_cluster_imbalance_ratio=5.0,
        min_cluster_fraction=0.05,
        classification=classification,
    )

    score = score_cluster_components(
        components,
        selection_metric="balanced_silhouette",
        imbalance_penalty_weight=0.15,
        tiny_cluster_penalty_weight=0.30,
        target_contrast_weight=0.0,
        missing_class_penalty_weight=0.2,
        single_class_penalty_weight=0.3,
        class_distribution_drift_weight=0.4,
    )

    assert components.valid is False
    assert score == pytest.approx(-1.0)


def test_selector_infers_string_classification_and_records_class_counts() -> None:
    rng = np.random.default_rng(17)
    embedding = np.vstack(
        (
            rng.normal(loc=-3.0, scale=0.1, size=(20, 2)),
            rng.normal(loc=3.0, scale=0.1, size=(20, 2)),
        )
    )
    target = np.tile(np.asarray(["negative", "positive"]), 20)
    selector = SpectralClusterSelector(
        algorithms=("kmeans",),
        target_type="auto",
        min_partitions=2,
        max_partitions=2,
        min_auto_partition_size=1,
        random_state=19,
        show_progress=False,
    )

    result = selector.select(embedding, target=target)
    classification = result.candidates[0].components["classification"]

    assert result.candidates[0].valid is True
    assert classification["class_labels"] == ["negative", "positive"]
    assert classification["global_class_counts"] == [20, 20]
    assert classification["cluster_class_counts"] == [[10, 10], [10, 10]]
    assert classification["missing_class_fraction"] == 0.0
    assert result.diagnostics["cluster_target_type"] == "auto"
    assert result.diagnostics["resolved_cluster_target_type"] == "classification"


def test_explicit_regression_type_preserves_integer_target_contrast() -> None:
    embedding = np.asarray(
        [[-2.0], [-1.8], [1.8], [2.0]],
        dtype=float,
    )
    target = np.asarray([0, 1, 10, 11], dtype=int)
    selector = SpectralClusterSelector(
        algorithms=("kmeans",),
        target_type="regression",
        target_contrast_weight=0.2,
        min_partitions=2,
        max_partitions=2,
        min_auto_partition_size=1,
        random_state=31,
        show_progress=False,
    )

    result = selector.select(embedding, target=target)
    components = result.candidates[0].components

    assert result.diagnostics["resolved_cluster_target_type"] == "regression"
    assert components["classification"] is None
    assert components["target_contrast"] > 0.0


def test_classification_weights_do_not_change_regression_score() -> None:
    components = evaluate_cluster_score_components(
        counts=(40, 40),
        silhouette=0.4,
        target_contrast=0.3,
        max_cluster_imbalance_ratio=5.0,
        min_cluster_fraction=0.05,
    )
    baseline = score_cluster_components(
        components,
        selection_metric="balanced_silhouette",
        imbalance_penalty_weight=0.15,
        tiny_cluster_penalty_weight=0.30,
        target_contrast_weight=0.2,
    )
    with_classification_weights = score_cluster_components(
        components,
        selection_metric="balanced_silhouette",
        imbalance_penalty_weight=0.15,
        tiny_cluster_penalty_weight=0.30,
        target_contrast_weight=0.2,
        missing_class_penalty_weight=10.0,
        single_class_penalty_weight=10.0,
        class_distribution_drift_weight=10.0,
    )

    assert with_classification_weights == baseline


def test_rmt_sampler_exposes_classification_candidate_diagnostics() -> None:
    rng = np.random.default_rng(23)
    frame = pd.DataFrame(
        rng.normal(size=(60, 6)),
        columns=[f"x_{index}" for index in range(6)],
    )
    target = pd.Series(np.tile(["class_a", "class_b"], 30))
    sampler = RMTContractionTensorSampler(
        partition_selection_method="auto",
        cluster_algorithms=("kmeans",),
        cluster_target_type="classification",
        min_partitions=2,
        max_partitions=2,
        min_auto_partition_size=1,
        n_views=2,
        projection_dim=2,
        backend="numpy",
        random_state=29,
        show_progress=False,
    )

    sampler.fit(frame, target=target)

    candidate = sampler.diagnostics_["partition_selection_candidate_details"][0]
    classification = candidate["components"]["classification"]
    assert sampler.diagnostics_["cluster_target_type"] == "classification"
    assert sampler.diagnostics_["resolved_cluster_target_type"] == "classification"
    assert classification["global_class_counts"] == [30, 30]
    assert sum(map(sum, classification["cluster_class_counts"])) == len(frame)


def test_sampling_ensemble_passes_target_only_to_rmt_standard_path() -> None:
    features = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
    target = pd.Series(["a", "b", "a", "b"])
    partitioner = _TargetAwarePartitioner()
    ensemble = SamplingEnsemble(
        problem="classification",
        partitioner_config={"strategy": "rmt_contraction"},
        model_factory=lambda: object(),
        show_progress=False,
    )

    partitions = ensemble._fit_and_collect_partitions(
        partitioner,
        "rmt_contraction",
        features,
        target,
    )

    assert partitioner.fit_target is target
    assert partitions["chunk_0"]["target"] is target


class _TargetAwarePartitioner:
    def __init__(self) -> None:
        self.fit_target = None

    def fit(self, features, target=None):
        self.fit_target = target
        return self

    @staticmethod
    def get_partitions(features, target):
        return {"chunk_0": {"feature": features, "target": target}}
