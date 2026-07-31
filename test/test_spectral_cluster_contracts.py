from __future__ import annotations

import math

import numpy as np
import pytest

import sampling_zoo.core.sampling_strategies.spectral.cluster_selection as cluster_module
from sampling_zoo.core.sampling_strategies.spectral.cluster_selection import (
    SpectralClusterSelector,
)
from sampling_zoo.core.sampling_strategies.spectral.cluster_selection_contracts import (
    ClusterCandidateFitFailureCode,
    ClusterCandidateKind,
    ClusterConstraintViolation,
    ClusterSelectionUnavailableError,
    build_cluster_candidate_plan,
    evaluate_cluster_score_components,
    score_cluster_components,
)


def test_candidate_plan_retains_size_guard_rejections() -> None:
    plan = build_cluster_candidate_plan(
        n_samples=512,
        algorithms=("kmeans", "gmm", "hdbscan"),
        min_partitions=2,
        max_partitions=6,
        min_auto_partition_size=256,
    )

    assert plan.requested_count_candidates == (2, 3, 4, 5, 6)
    assert plan.eligible_count_candidates == (2,)
    assert tuple(rejection.n_clusters for rejection in plan.size_guard_rejections) == (
        3,
        4,
        5,
        6,
    )
    assert plan.size_guard_fallback_applied is False
    assert plan.total_fit_count == 3
    assert [request.key for request in plan.requests] == [
        "kmeans|k=2",
        "gmm|k=2",
        "hdbscan|k=auto",
    ]
    assert plan.requests[-1].kind is ClusterCandidateKind.DENSITY_BASED


def test_candidate_plan_records_all_rejected_fallback_without_changing_grid() -> None:
    plan = build_cluster_candidate_plan(
        n_samples=96,
        algorithms=("kmeans", "hdbscan"),
        min_partitions=2,
        max_partitions=6,
        min_auto_partition_size=256,
    )

    assert plan.requested_count_candidates == (2, 3, 4, 5, 6)
    assert plan.eligible_count_candidates == plan.requested_count_candidates
    assert len(plan.size_guard_rejections) == 5
    assert plan.size_guard_fallback_applied is True
    assert plan.total_fit_count == 6
    assert plan.to_dict() == plan.to_dict()


@pytest.mark.parametrize("n_samples", [0, 1])
def test_candidate_plan_preserves_single_cluster_edge_request(
    n_samples: int,
) -> None:
    plan = build_cluster_candidate_plan(
        n_samples=n_samples,
        algorithms=("kmeans",),
        min_partitions=2,
        max_partitions=4,
        min_auto_partition_size=1,
    )

    assert plan.eligible_count_candidates == (1,)
    assert tuple(request.n_clusters for request in plan.requests) == (1,)


def test_balanced_score_exposes_hard_constraint_violations() -> None:
    components = evaluate_cluster_score_components(
        counts=(90, 10),
        silhouette=0.5,
        target_contrast=0.2,
        max_cluster_imbalance_ratio=5.0,
        min_cluster_fraction=0.15,
    )

    assert components.valid is False
    assert components.violations == (
        ClusterConstraintViolation.MAX_IMBALANCE_RATIO,
        ClusterConstraintViolation.MIN_CLUSTER_FRACTION,
    )
    assert components.tiny_cluster_mass == pytest.approx(0.1)
    expected = 0.5 - 0.15 * math.log(9.0) - 0.30 * 0.1 + 0.1 * 0.2 - 1.0
    assert score_cluster_components(
        components,
        selection_metric="balanced_silhouette",
        imbalance_penalty_weight=0.15,
        tiny_cluster_penalty_weight=0.30,
        target_contrast_weight=0.1,
    ) == pytest.approx(expected)
    assert components.to_dict()["constraint_violations"] == [
        "max_imbalance_ratio",
        "min_cluster_fraction",
    ]


def test_silhouette_score_ignores_balancing_penalties_by_contract() -> None:
    components = evaluate_cluster_score_components(
        counts=(99, 1),
        silhouette=0.4,
        target_contrast=3.0,
        max_cluster_imbalance_ratio=2.0,
        min_cluster_fraction=0.2,
    )

    score = score_cluster_components(
        components,
        selection_metric="silhouette",
        imbalance_penalty_weight=10.0,
        tiny_cluster_penalty_weight=10.0,
        target_contrast_weight=10.0,
    )

    assert score == pytest.approx(0.4)


def test_selector_keeps_successes_and_reports_failed_adapter(monkeypatch) -> None:
    embedding = _two_cluster_embedding()
    selector = SpectralClusterSelector(
        algorithms=("kmeans", "gmm"),
        min_partitions=2,
        max_partitions=2,
        min_auto_partition_size=1,
        show_progress=False,
    )
    original_factory = selector._make_count_based_estimator

    def fail_gmm(algorithm: str, n_clusters: int):
        if algorithm == "gmm":
            raise RuntimeError("synthetic GMM fit failure")
        return original_factory(algorithm, n_clusters)

    monkeypatch.setattr(selector, "_make_count_based_estimator", fail_gmm)

    result = selector.select(embedding)

    assert result.selected_algorithm == "kmeans"
    assert len(result.candidates) == 1
    assert len(result.candidate_failures) == 1
    failure = result.candidate_failures[0]
    assert failure.request.key == "gmm|k=2"
    assert failure.code is ClusterCandidateFitFailureCode.FIT_FAILED
    assert result.diagnostics["candidate_failures"] == [failure.to_dict()]


def test_selector_raises_structured_error_when_adapter_is_unavailable(
    monkeypatch,
) -> None:
    monkeypatch.setattr(cluster_module, "SklearnHDBSCAN", None)
    monkeypatch.setattr(cluster_module, "hdbscan_package", None)
    selector = SpectralClusterSelector(
        algorithms=("hdbscan",),
        min_partitions=2,
        max_partitions=3,
        min_auto_partition_size=1,
        show_progress=False,
    )

    with pytest.raises(ClusterSelectionUnavailableError) as error:
        selector.select(_two_cluster_embedding())

    assert error.value.plan.total_fit_count == 1
    assert len(error.value.failures) == 1
    assert (
        error.value.failures[0].code
        is ClusterCandidateFitFailureCode.ADAPTER_UNAVAILABLE
    )


def _two_cluster_embedding() -> np.ndarray:
    rng = np.random.default_rng(17)
    return np.vstack(
        (
            rng.normal(loc=-3.0, scale=0.1, size=(20, 2)),
            rng.normal(loc=3.0, scale=0.1, size=(20, 2)),
        )
    )
