from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor

from sampling_zoo.core.sampling_strategies.spectral.cluster_selection import (
    SpectralClusterSelector,
)
from sampling_zoo.core.sampling_strategies.spectral.cluster_selection_contracts import (
    evaluate_cluster_score_components,
    score_cluster_components,
)
from sampling_zoo.core.sampling_strategies.spectral.partition_validation import (
    PartitionDownstreamProxyEvaluator,
    PartitionValidationProxyEvaluator,
    build_partition_validation_plan,
)
from sampling_zoo.core.sampling_strategies.spectral.rmt_contraction_sampler import (
    RMTContractionTensorSampler,
)


def test_classification_validation_plan_is_deterministic_and_stratified() -> None:
    target = np.repeat(np.asarray(["a", "b", "c"]), 12)

    first = build_partition_validation_plan(
        target,
        target_type="classification",
        validation_fraction=0.25,
        min_partition_train_rows=2,
        random_state=17,
    )
    second = build_partition_validation_plan(
        target,
        target_type="classification",
        validation_fraction=0.25,
        min_partition_train_rows=2,
        random_state=17,
    )

    assert first == second
    assert set(first.train_indices).isdisjoint(first.validation_indices)
    assert set(first.train_indices) | set(first.validation_indices) == set(
        range(target.size)
    )
    assert set(target[list(first.train_indices)]) == {"a", "b", "c"}
    assert set(target[list(first.validation_indices)]) == {"a", "b", "c"}
    assert first.to_dict() == second.to_dict()


def test_classification_validation_plan_rejects_single_class_target() -> None:
    with pytest.raises(ValueError, match="at least two classes"):
        build_partition_validation_plan(
            np.zeros(12, dtype=int),
            target_type="classification",
            validation_fraction=0.2,
            min_partition_train_rows=2,
            random_state=17,
        )


def test_regression_single_partition_matches_global_baseline() -> None:
    rng = np.random.default_rng(19)
    embedding = rng.normal(size=(80, 3))
    target = rng.normal(size=80)
    evaluator = PartitionValidationProxyEvaluator(
        embedding,
        target,
        target_type="regression",
        validation_fraction=0.2,
        min_partition_train_rows=4,
        random_state=23,
    )

    result = evaluator.evaluate(np.zeros(80, dtype=int))

    assert result.loss_name == "rmse"
    assert result.candidate_loss == pytest.approx(result.baseline_loss)
    assert result.relative_gain == pytest.approx(0.0)
    assert result.routed_validation_counts == (16,)
    assert result.fallback_validation_fraction == 0.0


def test_regression_target_aligned_partitions_improve_validation_proxy() -> None:
    embedding, target, labels = _regression_regions()
    evaluator = PartitionValidationProxyEvaluator(
        embedding,
        target,
        target_type="regression",
        validation_fraction=0.2,
        min_partition_train_rows=4,
        random_state=29,
    )

    result = evaluator.evaluate(labels)

    assert result.candidate_loss < result.baseline_loss
    assert result.relative_gain > 0.8
    assert sum(result.routed_validation_counts) == len(
        evaluator.plan.validation_indices
    )


def test_classification_target_aligned_partitions_improve_log_loss_proxy() -> None:
    rng = np.random.default_rng(31)
    embedding = np.vstack(
        (
            rng.normal(-2.0, 0.15, size=(100, 2)),
            rng.normal(2.0, 0.15, size=(100, 2)),
        )
    )
    target = np.concatenate(
        (
            np.tile(np.asarray([0, 0, 0, 0, 1]), 20),
            np.tile(np.asarray([0, 1, 1, 1, 1]), 20),
        )
    )
    labels = np.repeat(np.asarray([0, 1]), 100)
    evaluator = PartitionValidationProxyEvaluator(
        embedding,
        target,
        target_type="classification",
        validation_fraction=0.2,
        min_partition_train_rows=4,
        classification_smoothing=1.0,
        random_state=37,
    )

    result = evaluator.evaluate(labels)

    assert result.loss_name == "log_loss"
    assert result.candidate_loss < result.baseline_loss
    assert result.relative_gain > 0.1
    assert sum(result.routed_validation_counts) == len(
        evaluator.plan.validation_indices
    )


def test_validation_proxy_is_invariant_to_partition_label_encoding() -> None:
    embedding, target, labels = _regression_regions()
    evaluator = PartitionValidationProxyEvaluator(
        embedding,
        target,
        target_type="regression",
        validation_fraction=0.2,
        min_partition_train_rows=4,
        random_state=41,
    )

    original = evaluator.evaluate(labels)
    remapped = evaluator.evaluate(np.where(labels == 0, 17, -4))

    assert remapped.baseline_loss == original.baseline_loss
    assert remapped.candidate_loss == original.candidate_loss
    assert remapped.relative_gain == original.relative_gain
    assert remapped.fallback_validation_fraction == (
        original.fallback_validation_fraction
    )
    assert sorted(remapped.routed_validation_counts) == sorted(
        original.routed_validation_counts
    )


def test_validation_proxy_reports_fallback_for_undersized_partition() -> None:
    rng = np.random.default_rng(42)
    embedding = np.vstack(
        (
            rng.normal(-3.0, 0.1, size=(100, 2)),
            rng.normal(3.0, 0.1, size=(10, 2)),
        )
    )
    target = np.concatenate((np.zeros(100), np.ones(10)))
    labels = np.concatenate(
        (np.zeros(100, dtype=int), np.ones(10, dtype=int))
    )
    evaluator = PartitionValidationProxyEvaluator(
        embedding,
        target,
        target_type="regression",
        validation_fraction=0.2,
        min_partition_train_rows=20,
        random_state=43,
    )

    result = evaluator.evaluate(labels)

    assert result.fallback_partition_count == 1
    assert result.fallback_validation_fraction > 0.0
    assert sum(result.routed_validation_counts) == len(
        evaluator.plan.validation_indices
    )


def test_downstream_proxy_preserves_full_train_absolute_budget() -> None:
    rng = np.random.default_rng(101)
    embedding = np.vstack(
        (
            rng.normal(-1.0, 0.2, size=(50, 3)),
            rng.normal(1.0, 0.2, size=(50, 3)),
        )
    )
    target = embedding[:, 0] + rng.normal(0.0, 0.05, size=100)
    labels = np.repeat(np.asarray([0, 1]), 50)
    evaluator = PartitionDownstreamProxyEvaluator(
        embedding=embedding,
        features=embedding,
        target=target,
        sample_scores=np.ones(100),
        target_type="regression",
        model_factory=lambda: DummyRegressor(strategy="mean"),
        budget_ratio=0.2,
        min_partition_rows=9,
        max_imbalance_ratio=5.0,
        min_partition_fraction=0.05,
        selection_method="all",
        routing_temperature=1.0,
        routing_shrinkage=0.0,
        validation_fraction=0.2,
        random_state=103,
    )

    result = evaluator.evaluate(labels)

    assert result.status == "ok"
    assert result.budget_reference_rows == 100
    assert result.proxy_train_rows == 80
    assert result.requested_budget_size == 20
    assert result.selected_budget_size == 20
    assert sum(result.sampled_partition_sizes) == 20
    assert min(result.sampled_partition_sizes) >= 9
    assert result.budget_violations == ()


def test_validation_only_partition_does_not_define_proxy_centroid() -> None:
    rng = np.random.default_rng(44)
    embedding = rng.normal(size=(60, 2))
    target = rng.normal(size=60)
    evaluator = PartitionValidationProxyEvaluator(
        embedding,
        target,
        target_type="regression",
        validation_fraction=0.2,
        min_partition_train_rows=2,
        random_state=45,
    )
    labels = np.zeros(60, dtype=int)
    labels[list(evaluator.plan.validation_indices)] = 1

    result = evaluator.evaluate(labels)

    assert result.fallback_partition_count == 1
    assert result.routed_validation_counts[1] == 0
    assert result.fallback_validation_fraction == 0.0


def test_validation_proxy_score_uses_relative_gain_and_hard_guard() -> None:
    embedding, target, labels = _regression_regions()
    validation = PartitionValidationProxyEvaluator(
        embedding,
        target,
        target_type="regression",
        validation_fraction=0.2,
        min_partition_train_rows=4,
        random_state=43,
    ).evaluate(labels)
    components = evaluate_cluster_score_components(
        counts=(100, 100),
        silhouette=0.7,
        target_contrast=0.8,
        max_cluster_imbalance_ratio=5.0,
        min_cluster_fraction=0.05,
        validation_proxy=validation,
    )

    score = score_cluster_components(
        components,
        selection_metric="validation_proxy",
        imbalance_penalty_weight=100.0,
        tiny_cluster_penalty_weight=100.0,
        target_contrast_weight=100.0,
    )

    assert score == validation.relative_gain
    guarded_components = evaluate_cluster_score_components(
        counts=(190, 10),
        silhouette=0.9,
        target_contrast=1.0,
        max_cluster_imbalance_ratio=5.0,
        min_cluster_fraction=0.10,
        validation_proxy=validation,
    )
    guarded_score = score_cluster_components(
        guarded_components,
        selection_metric="validation_proxy",
        imbalance_penalty_weight=0.0,
        tiny_cluster_penalty_weight=0.0,
        target_contrast_weight=0.0,
    )
    assert guarded_score == pytest.approx(validation.relative_gain - 1.0)


def test_selector_reuses_one_validation_plan_for_all_candidates() -> None:
    embedding, target, _labels = _regression_regions()
    selector = SpectralClusterSelector(
        algorithms=("kmeans", "gmm"),
        selection_metric="validation_proxy",
        ensemble_method="best_score",
        target_type="regression",
        min_partitions=2,
        max_partitions=3,
        min_auto_partition_size=1,
        validation_proxy_fraction=0.2,
        validation_proxy_min_partition_rows=4,
        random_state=47,
        show_progress=False,
    )

    result = selector.select(embedding, target=target)

    plan = result.diagnostics["validation_proxy_plan"]
    assert plan["target_type"] == "regression"
    assert plan["n_train"] + plan["n_validation"] == len(target)
    assert result.diagnostics["cluster_selection_metric"] == "validation_proxy"
    assert all(
        candidate.components["validation_proxy"] is not None
        for candidate in result.candidates
    )
    assert all(
        candidate.score
        == pytest.approx(candidate.components["validation_proxy"]["relative_gain"])
        for candidate in result.candidates
        if candidate.valid
    )


def test_validation_proxy_scores_coassociation_candidate() -> None:
    embedding, target, _labels = _regression_regions()
    selector = SpectralClusterSelector(
        algorithms=("kmeans", "gmm"),
        selection_metric="validation_proxy",
        ensemble_method="coassociation",
        target_type="regression",
        min_partitions=2,
        max_partitions=3,
        min_auto_partition_size=1,
        validation_proxy_min_partition_rows=4,
        random_state=49,
        show_progress=False,
    )

    result = selector.select(embedding, target=target)

    consensus = result.diagnostics["consensus"]
    assert consensus is not None
    assert consensus["candidates"]
    assert (
        consensus["candidates"][0]["components"]["validation_proxy"]
        is not None
    )


def test_validation_proxy_selector_requires_target() -> None:
    selector = SpectralClusterSelector(
        algorithms=("kmeans",),
        selection_metric="validation_proxy",
        min_partitions=2,
        max_partitions=2,
        min_auto_partition_size=1,
        show_progress=False,
    )

    with pytest.raises(ValueError, match="requires a supported target"):
        selector.select(np.arange(40, dtype=float).reshape(20, 2))


def test_rmt_sampler_exposes_validation_proxy_diagnostics() -> None:
    rng = np.random.default_rng(53)
    features = pd.DataFrame(
        np.vstack(
            (
                rng.normal(-1.5, 0.3, size=(50, 5)),
                rng.normal(1.5, 0.3, size=(50, 5)),
            )
        ),
        columns=[f"x_{index}" for index in range(5)],
    )
    target = pd.Series(
        np.concatenate(
            (
                rng.normal(-4.0, 0.2, size=50),
                rng.normal(4.0, 0.2, size=50),
            )
        )
    )
    sampler = RMTContractionTensorSampler(
        partition_selection_method="auto",
        cluster_algorithms=("kmeans",),
        cluster_selection_metric="validation_proxy",
        cluster_ensemble_method="best_score",
        cluster_target_type="regression",
        min_partitions=2,
        max_partitions=3,
        min_auto_partition_size=1,
        validation_proxy_min_partition_rows=4,
        n_views=2,
        projection_dim=2,
        backend="numpy",
        random_state=59,
        show_progress=False,
    )

    sampler.fit(features, target=target)

    plan = sampler.diagnostics_["partition_selection_validation_proxy_plan"]
    candidates = sampler.diagnostics_["partition_selection_candidate_details"]
    selected = sampler.diagnostics_["partition_selection_selected_candidate"]
    assert plan["target_type"] == "regression"
    assert plan["n_samples"] == len(features)
    assert candidates
    assert selected["components"]["validation_proxy"] is not None
    assert all(
        candidate["components"]["validation_proxy"] is not None
        for candidate in candidates
    )


def _regression_regions() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(13)
    embedding = np.vstack(
        (
            rng.normal(-3.0, 0.2, size=(100, 2)),
            rng.normal(3.0, 0.2, size=(100, 2)),
        )
    )
    target = np.concatenate(
        (
            rng.normal(-5.0, 0.2, size=100),
            rng.normal(5.0, 0.2, size=100),
        )
    )
    labels = np.repeat(np.asarray([0, 1]), 100)
    return embedding, target, labels
