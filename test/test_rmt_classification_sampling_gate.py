from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from examples.benchmark.benchmark_dataset_interfaces import (
    stratified_cap_indices,
)
from examples.benchmark.rmt_classification_sampling_gate import (
    make_rmt_classification_gate_strategy_configs,
)
from sampling_zoo.core.experiment.errors import (
    ClassificationProbabilitiesRequiredError,
)
from sampling_zoo.core.metrics.eval_metrics import (
    calculate_metrics,
    metric_drop,
    primary_classification_metric,
)
from sampling_zoo.core.sampling_strategies.spectral.classification_sampling import (
    build_classification_partition_budget_plan,
    select_class_aware_partition_indices,
)
from sampling_zoo.core.sampling_strategies.spectral.rmt_contraction_sampler import (
    RMTContractionTensorSampler,
)
from sampling_zoo.core.utils.sampling_ensemble import SamplingEnsemble


def _selection_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(30)
    target = np.asarray([0] * 18 + [1] * 8 + [2] * 4)
    scores = np.linspace(0.01, 1.0, indices.size)
    embedding = np.column_stack([scores, np.sin(scores * 4.0)])
    return indices, target, scores, embedding


def test_class_aware_selection_preserves_classes_and_exact_budget() -> None:
    indices, target, scores, embedding = _selection_inputs()
    first = select_class_aware_partition_indices(
        indices,
        target=target,
        target_size=9,
        min_samples_per_class=1,
        selection_method="capped_leverage",
        scores=scores,
        embedding=embedding,
        random_state=17,
    )
    second = select_class_aware_partition_indices(
        indices,
        target=target,
        target_size=9,
        min_samples_per_class=1,
        selection_method="capped_leverage",
        scores=scores,
        embedding=embedding,
        random_state=17,
    )

    assert first.feasible
    assert first.selected_indices.size == 9
    assert np.unique(first.selected_indices).size == 9
    assert set(target[first.selected_indices]) == {0, 1, 2}
    assert np.array_equal(first.selected_indices, second.selected_indices)


def test_proportional_class_allocation_preserves_distribution_and_budget() -> None:
    indices, target, scores, embedding = _selection_inputs()
    plan = select_class_aware_partition_indices(
        indices,
        target=target,
        target_size=15,
        min_samples_per_class=1,
        class_allocation_policy="proportional",
        selection_method="capped_leverage",
        scores=scores,
        embedding=embedding,
        random_state=17,
    )

    assert plan.feasible
    assert plan.selected_indices.size == 15
    assert dict(plan.allocated_class_counts) == {"0": 9, "1": 4, "2": 2}
    assert plan.allocated_class_counts == plan.selected_class_counts
    assert plan.distribution_total_variation == pytest.approx(0.0)
    assert plan.to_dict()["allocation_policy"] == "proportional"


def test_class_allocation_policy_rejects_unknown_value() -> None:
    indices, target, scores, embedding = _selection_inputs()
    with pytest.raises(ValueError, match="class_allocation_policy"):
        select_class_aware_partition_indices(
            indices,
            target=target,
            target_size=9,
            min_samples_per_class=1,
            class_allocation_policy="unknown",
            selection_method="capped_leverage",
            scores=scores,
            embedding=embedding,
            random_state=17,
        )


def test_class_aware_selection_reports_infeasible_budget() -> None:
    indices, target, scores, embedding = _selection_inputs()
    plan = select_class_aware_partition_indices(
        indices,
        target=target,
        target_size=2,
        min_samples_per_class=1,
        selection_method="hybrid",
        scores=scores,
        embedding=embedding,
        random_state=42,
    )

    assert not plan.feasible
    assert plan.selected_indices.size == 0
    assert "budget_below_class_coverage_minimum" in plan.violations


def test_classification_budget_plan_reduces_partitions_to_preserve_classes() -> None:
    target = np.resize(np.asarray([0, 1, 2, 3]), 400)

    plan = build_classification_partition_budget_plan(
        target,
        n_rows=400,
        sampling_budget_ratio=0.01,
        min_samples_per_class=1,
        requested_n_partitions=5,
        requested_min_partitions=2,
        requested_max_partitions=8,
    )

    assert plan.feasible
    assert plan.total_budget == 4
    assert plan.min_rows_per_partition == 4
    assert plan.effective_n_partitions == 1
    assert plan.effective_min_partitions == 1
    assert plan.effective_max_partitions == 1
    assert plan.include_single_partition_candidate


def test_rmt_sampler_uses_single_partition_when_class_budget_requires_it() -> None:
    rng = np.random.default_rng(29)
    features = pd.DataFrame(rng.normal(size=(400, 6)))
    target = pd.Series(np.resize(np.asarray([0, 1, 2, 3]), len(features)))
    sampler = RMTContractionTensorSampler(
        n_partitions=5,
        partition_selection_method="fixed",
        min_partitions=2,
        max_partitions=8,
        cluster_target_type="classification",
        class_coverage_policy="preserve_local_classes",
        sampling_budget_ratio=0.01,
        n_views=2,
        projection_dim=2,
        backend="numpy",
        show_progress=False,
        random_state=13,
    ).fit(features, target=target)

    assert len(sampler.partitions) == 1
    selected = next(iter(sampler.partitions.values()))
    assert selected.size == 4
    assert set(target.iloc[selected]) == {0, 1, 2, 3}
    diagnostics = sampler.diagnostics_["classification_partition_budget_plan"]
    assert diagnostics["effective_n_partitions"] == 1
    assert sampler.diagnostics_["effective_budget_feasibility_mode"] == "hard"


def test_stratified_cap_is_exact_deterministic_and_preserves_classes() -> None:
    target = np.asarray([0] * 80 + [1] * 15 + [2] * 5)
    first = stratified_cap_indices(target, cap=20, seed=9)
    second = stratified_cap_indices(target, cap=20, seed=9)

    assert first.size == 20
    assert np.array_equal(first, second)
    assert set(target[first]) == {0, 1, 2}


def test_binary_and_multiclass_metrics_follow_amlb_policy() -> None:
    binary_true = np.asarray([0, 1, 1, 0])
    binary_proba = np.asarray(
        [[0.9, 0.1], [0.2, 0.8], [0.3, 0.7], [0.8, 0.2]]
    )
    binary = calculate_metrics(
        binary_true,
        np.argmax(binary_proba, axis=1),
        binary_proba,
        "classification",
        classes=[0, 1],
    )
    assert binary["roc_auc"] == pytest.approx(1.0)
    assert binary["brier_score"] == pytest.approx(0.045)
    assert binary["expected_calibration_error"] == pytest.approx(0.2)
    assert primary_classification_metric([0, 1]) == "roc_auc"

    multiclass_true = np.asarray([0, 1, 2, 1])
    multiclass_proba = np.asarray(
        [
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.2, 0.7],
            [0.2, 0.7, 0.1],
        ]
    )
    multiclass = calculate_metrics(
        multiclass_true,
        np.argmax(multiclass_proba, axis=1),
        multiclass_proba,
        "classification",
        classes=[0, 1, 2],
    )
    assert np.isnan(multiclass["roc_auc"])
    assert multiclass["log_loss"] < 0.5
    assert 0.0 <= multiclass["brier_score"] < 1.0
    assert 0.0 <= multiclass["expected_calibration_error"] <= 1.0
    assert primary_classification_metric([0, 1, 2]) == "log_loss"
    assert metric_drop("roc_auc", 0.9, 0.95) == pytest.approx(0.05)
    assert metric_drop("log_loss", 0.45, 0.4) == pytest.approx(0.05)


class _StaticProbabilityModel:
    def __init__(self, classes, probabilities) -> None:
        self.classes_ = np.asarray(classes)
        self.probabilities = np.asarray(probabilities, dtype=float)

    def predict_proba(self, features):
        return self.probabilities[: len(features)]


class _LabelOnlyModel:
    def predict(self, features):
        return np.zeros(len(features), dtype=int)


def test_ensemble_aligns_chunk_probabilities_to_global_classes() -> None:
    features = pd.DataFrame({"x": [0, 1, 2]})
    ensemble = SamplingEnsemble(
        problem="classification",
        ensemble_method="voting",
        show_progress=False,
    )
    ensemble.classes_ = np.asarray([0, 1, 2])
    ensemble.models = [
        {
            "name": "chunk_0",
            "model": _StaticProbabilityModel(
                [0, 1],
                [[0.8, 0.2], [0.2, 0.8], [0.6, 0.4]],
            ),
            "metrics": {"f1_weighted": 0.8},
        },
        {
            "name": "chunk_1",
            "model": _StaticProbabilityModel(
                [2, 1],
                [[0.1, 0.9], [0.7, 0.3], [0.8, 0.2]],
            ),
            "metrics": {"f1_weighted": 0.7},
        },
    ]

    probabilities = ensemble.ensemble_predict_proba(features)

    assert probabilities.shape == (3, 3)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert np.array_equal(
        ensemble.ensemble_predict(features),
        ensemble.classes_[np.argmax(probabilities, axis=1)],
    )


def test_probability_required_error_is_structured() -> None:
    ensemble = SamplingEnsemble(
        problem="classification",
        ensemble_method="voting",
        show_progress=False,
    )
    ensemble.classes_ = np.asarray([0, 1])
    ensemble.models = [
        {
            "name": "chunk_0",
            "model": _LabelOnlyModel(),
            "metrics": {"f1_weighted": 0.5},
        }
    ]

    with pytest.raises(ClassificationProbabilitiesRequiredError) as error:
        ensemble.ensemble_predict_proba(pd.DataFrame({"x": [1, 2]}))
    assert error.value.code == "classification_probabilities_required"


def test_class_coverage_repair_uses_global_classes_as_source_of_truth() -> None:
    ensemble = SamplingEnsemble(
        problem="classification",
        ensemble_method="voting",
        show_progress=False,
    )
    ensemble.classes_ = np.asarray([0, 1, 2])
    ensemble._record_class_coverage_repair(
        "chunk_0",
        {"0": 5, "1": 2},
        {"0": 5, "1": 2},
        status="sampler_guaranteed",
    )

    diagnostics = ensemble.class_coverage_repairs_["chunk_0"]
    assert diagnostics["missing_classes_before"] == ["2"]
    assert diagnostics["missing_classes_after"] == ["2"]


def test_rmt_sampler_preserves_classes_after_budget_selection() -> None:
    rng = np.random.default_rng(23)
    features = pd.DataFrame(rng.normal(size=(180, 8)))
    target = pd.Series(np.tile(np.asarray([0, 1, 2]), 60))
    sampler = RMTContractionTensorSampler(
        n_partitions=2,
        partition_selection_method="fixed",
        cluster_target_type="classification",
        class_coverage_policy="preserve_local_classes",
        class_allocation_policy="proportional",
        sampling_budget_ratio=0.25,
        selection_method="capped_leverage",
        n_views=4,
        projection_dim=2,
        backend="numpy",
        show_progress=False,
        random_state=11,
    )

    sampler.fit(features, target=target)

    assert sampler.class_coverage_guaranteed_
    assert sampler.diagnostics_["class_coverage_guaranteed"]
    assert sampler.diagnostics_["class_allocation_policy"] == "proportional"
    for indices in sampler.partitions.values():
        assert np.unique(target.iloc[indices]).size >= 2


def test_classification_gate_grid_has_probability_baseline_and_class_aware_rmt() -> None:
    grid = make_rmt_classification_gate_strategy_configs(
        budget_ratios=(0.1,),
        ensemble_methods=("voting", "routed_weighted"),
        n_partitions=4,
        seed=42,
    )

    assert "full_dataset" in grid
    rmt_configs = [
        config
        for config in grid.values()
        if config.get("strategy") == "rmt_contraction"
    ]
    assert len(rmt_configs) == 2
    assert all(
        config["class_coverage_policy"] == "preserve_local_classes"
        for config in rmt_configs
    )
    assert all(
        config["selection_method"] == "capped_leverage"
        for config in rmt_configs
    )
