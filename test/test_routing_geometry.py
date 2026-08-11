from __future__ import annotations

import importlib.util
import json

import numpy as np
import pandas as pd
import pytest

from sampling_zoo.core.experiment.routing_replay import (
    RoutingGeometrySelectionPolicy,
    RoutingReplayEvaluator,
    RoutingReplayRequest,
    RoutingReplayResult,
    ValidationRoutingGeometrySelector,
    upper_tail_mean_absolute_error,
)
from sampling_zoo.core.experiment.routing_geometry_selection import (
    CrossFittedRoutingGeometrySelectionSpec,
    CrossFittedRoutingGeometrySelector,
)
from sampling_zoo.core.experiment.errors import ResumeCompatibilityError
from sampling_zoo.core.sampling_strategies.spectral.backend.matrix_backend import MatrixRMTBackend
from sampling_zoo.core.sampling_strategies.spectral.backend.tensor_backend import TensorRMTBackend
from sampling_zoo.core.sampling_strategies.spectral.rmt_contraction_sampler import (
    RMTContractionTensorSampler,
)
from sampling_zoo.core.sampling_strategies.spectral.routing_contracts import (
    PartitionGeometryContract,
    PartitionGeometrySpec,
    RoutingDistanceContract,
    RoutingWeightContract,
)
from sampling_zoo.core.metrics.eval_metrics import classification_brier_score
from sampling_zoo.core.sampling_strategies.spectral.routing_geometry import (
    PartitionGeometryBuilder,
    route_partition_geometry,
)
from examples.benchmark.routing_geometry_replay import (
    FittedEnsembleRoutingReplay,
    default_routing_geometry_arms,
    default_validation_selector_arms,
    default_validation_selector_policy,
)
from examples.benchmark.rmt_routing_geometry_experiment import (
    RoutingGeometryExperimentConfig,
    RoutingGeometryExperimentOrchestrator,
)
from examples.benchmark.rmt_validation_geometry_selector_experiment import (
    RoutingGeometrySelectorExperimentConfig,
    ValidationRoutingGeometryExperimentOrchestrator,
)
from examples.benchmark.rmt_cross_fitted_geometry_selector_experiment import (
    CrossFittedRoutingGeometryExperimentConfig,
    CrossFittedRoutingGeometryExperimentOrchestrator,
)
from examples.benchmark.benchmark_dataset_interfaces import make_synthetic_regression_smoke_dataset


def _geometry_inputs() -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, int]]:
    embedding = np.asarray(
        [
            [-2.0, -0.2],
            [-1.8, 0.1],
            [-2.2, 0.0],
            [2.0, -0.1],
            [1.8, 0.2],
            [2.2, 0.0],
        ],
        dtype=float,
    )
    labels = np.asarray([0, 0, 0, 1, 1, 1])
    partitions = {
        "chunk_0": np.asarray([0, 1]),
        "chunk_1": np.asarray([3, 4]),
    }
    mapping = {"chunk_0": 0, "chunk_1": 1}
    return embedding, labels, partitions, mapping


def _selector_result(
    arm_name: str,
    metric: str,
    value: float,
) -> RoutingReplayResult:
    routing = RoutingWeightContract(
        partition_names=("chunk_0", "chunk_1"),
        weights=np.full((2, 2), 0.5),
        temperature=1.0,
    )
    return RoutingReplayResult(
        arm_name=arm_name,
        temperature=1.0,
        primary_metric=metric,
        primary_value=value,
        metrics={metric: value},
        routing=routing,
        blended_output=np.zeros(2),
    )


@pytest.mark.parametrize(
    ("metric", "kernel"),
    [
        ("squared_euclidean", "softmax"),
        ("median_scaled_euclidean", "softmax"),
        ("diag_shrinkage_mahalanobis", "softmax"),
        ("full_shrinkage_mahalanobis", "softmax"),
        ("full_shrinkage_mahalanobis", "gmm_posterior"),
        ("cosine", "softmax"),
    ],
)
def test_routing_geometry_produces_aligned_normalized_weights(metric: str, kernel: str) -> None:
    embedding, labels, partitions, mapping = _geometry_inputs()
    spec = PartitionGeometrySpec(
        metric=metric,
        kernel=kernel,
        uniform_shrinkage=0.0,
        covariance_shrinkage=0.2,
    )
    geometry = PartitionGeometryBuilder().build(
        embedding=embedding,
        cluster_labels=labels,
        partitions=partitions,
        partition_to_cluster=mapping,
        spec=spec,
    )

    distances, routing = route_partition_geometry(
        backend=MatrixRMTBackend(),
        embedding=np.asarray([[-2.0, 0.0], [2.0, 0.0], [0.0, 0.0]]),
        geometry=geometry,
    )

    assert geometry.partition_names == ("chunk_0", "chunk_1")
    assert distances.values.shape == (3, 2)
    assert routing.weights.shape == (3, 2)
    assert np.allclose(routing.weights.sum(axis=1), 1.0)
    assert np.argmax(routing.weights[0]) == 0
    assert np.argmax(routing.weights[1]) == 1


def test_sampled_centroid_uses_only_selected_partition_rows() -> None:
    embedding, labels, partitions, mapping = _geometry_inputs()
    source = PartitionGeometryBuilder().build(
        embedding=embedding,
        cluster_labels=labels,
        partitions=partitions,
        partition_to_cluster=mapping,
        spec=PartitionGeometrySpec(representation="source_centroid"),
    )
    sampled = PartitionGeometryBuilder().build(
        embedding=embedding,
        cluster_labels=labels,
        partitions=partitions,
        partition_to_cluster=mapping,
        spec=PartitionGeometrySpec(representation="sampled_centroid"),
    )

    assert source.representation_counts == (3, 3)
    assert sampled.representation_counts == (2, 2)
    assert not np.allclose(source.centers, sampled.centers)


def test_geometry_contract_arrays_are_read_only() -> None:
    embedding, labels, partitions, mapping = _geometry_inputs()
    geometry = PartitionGeometryBuilder().build(
        embedding=embedding,
        cluster_labels=labels,
        partitions=partitions,
        partition_to_cluster=mapping,
        spec=PartitionGeometrySpec(),
    )

    with pytest.raises(ValueError):
        geometry.centers[0, 0] = 999.0


def test_invalid_gmm_metric_combination_fails_at_config_boundary() -> None:
    with pytest.raises(ValueError, match="gmm_posterior requires"):
        PartitionGeometrySpec(metric="squared_euclidean", kernel="gmm_posterior")


def test_identity_mahalanobis_matches_squared_euclidean() -> None:
    embedding = np.asarray([[-1.0, 0.5], [2.0, -0.25]])
    centers = np.asarray([[0.0, 0.0], [1.0, 1.0]])
    backend = MatrixRMTBackend()

    euclidean = backend.compute_routing_values(
        embedding,
        centers,
        metric="squared_euclidean",
        kernel="softmax",
    )
    mahalanobis = backend.compute_routing_values(
        embedding,
        centers,
        metric="full_shrinkage_mahalanobis",
        kernel="softmax",
        precisions=np.repeat(np.eye(2)[None, :, :], 2, axis=0),
    )

    assert np.allclose(euclidean, mahalanobis)


def test_median_scaled_distance_is_invariant_to_global_embedding_scale() -> None:
    embedding, labels, partitions, mapping = _geometry_inputs()
    spec = PartitionGeometrySpec(
        metric="median_scaled_euclidean",
        uniform_shrinkage=0.0,
    )
    builder = PartitionGeometryBuilder()
    geometry = builder.build(
        embedding=embedding,
        cluster_labels=labels,
        partitions=partitions,
        partition_to_cluster=mapping,
        spec=spec,
    )
    scaled_geometry = builder.build(
        embedding=embedding * 7.0,
        cluster_labels=labels,
        partitions=partitions,
        partition_to_cluster=mapping,
        spec=spec,
    )
    evaluation = np.asarray([[-1.5, 0.0], [1.5, 0.0], [0.0, 0.2]])
    backend = MatrixRMTBackend()

    original = backend.compute_routing_values(
        evaluation,
        geometry.centers,
        metric=str(spec.metric),
        kernel=str(spec.kernel),
        scales=geometry.scales,
    )
    scaled = backend.compute_routing_values(
        evaluation * 7.0,
        scaled_geometry.centers,
        metric=str(spec.metric),
        kernel=str(spec.kernel),
        scales=scaled_geometry.scales,
    )

    assert np.allclose(original, scaled)


def test_gmm_posterior_respects_partition_priors() -> None:
    spec = PartitionGeometrySpec(
        metric="full_shrinkage_mahalanobis",
        kernel="gmm_posterior",
        uniform_shrinkage=0.0,
    )
    geometry = PartitionGeometryContract(
        spec=spec,
        partition_names=("chunk_0", "chunk_1"),
        cluster_ids=(0, 1),
        centers=np.zeros((2, 2)),
        scales=np.ones(2),
        priors=np.asarray([0.9, 0.1]),
        precisions=np.repeat(np.eye(2)[None, :, :], 2, axis=0),
        log_determinants=np.zeros(2),
        source_counts=(9, 1),
        representation_counts=(9, 1),
    )

    _distances, routing = route_partition_geometry(
        backend=MatrixRMTBackend(),
        embedding=np.zeros((1, 2)),
        geometry=geometry,
    )

    assert routing.weights[0, 0] == pytest.approx(0.9)
    assert routing.weights[0, 1] == pytest.approx(0.1)


def test_sampler_can_replay_alternative_geometry_without_refit() -> None:
    rng = np.random.default_rng(51)
    frame = pd.DataFrame(rng.normal(size=(60, 5)), columns=[f"x_{idx}" for idx in range(5)])
    sampler = RMTContractionTensorSampler(
        n_partitions=3,
        n_views=3,
        projection_dim=2,
        backend="numpy",
        random_state=51,
        show_progress=False,
    ).fit(frame)
    original_basis = np.array(sampler.right_basis_, copy=True)

    distances, routing = sampler.predict_partition_routing_contracts(
        frame.iloc[:8],
        geometry_spec=PartitionGeometrySpec(
            metric="diag_shrinkage_mahalanobis",
            covariance_shrinkage=0.2,
        ),
    )

    assert distances.partition_names == tuple(sampler.partition_names_)
    assert routing.partition_names == tuple(sampler.partition_names_)
    assert np.allclose(routing.weights.sum(axis=1), 1.0)
    assert np.array_equal(original_basis, sampler.right_basis_)


def test_temperature_replay_selects_better_regression_routing() -> None:
    spec = PartitionGeometrySpec(temperature=1.0, uniform_shrinkage=0.0)
    distances = RoutingDistanceContract(
        partition_names=("chunk_0", "chunk_1"),
        values=np.asarray([[0.0, 1.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0]]),
        kind="distance",
    )
    request = RoutingReplayRequest(
        arm_name="euclidean",
        geometry_spec=spec,
        distances=distances,
        target=np.asarray([0.0, 0.0, 10.0, 10.0]),
        expert_outputs=np.asarray(
            [
                [0.0, 10.0],
                [0.0, 10.0],
                [0.0, 10.0],
                [0.0, 10.0],
            ]
        ),
        problem_type="regression",
    )

    selection = RoutingReplayEvaluator().calibrate_temperature(request, (0.1, 1.0, 10.0))

    assert selection.best.temperature == 0.1
    assert selection.best.primary_metric == "rmse"
    assert selection.best.diagnostics["oracle_expert_regret"] >= 0.0
    assert "tail_mean_absolute_error" in selection.best.metrics
    assert selection.best.diagnostics["tail_absolute_error_quantile"] == pytest.approx(0.9)


def test_upper_tail_mean_absolute_error_uses_worst_requested_fraction() -> None:
    target = np.zeros(10)
    prediction = np.arange(10, dtype=float)

    value = upper_tail_mean_absolute_error(target, prediction, quantile=0.8)

    assert value == pytest.approx(8.5)


def test_regression_stratification_balances_target_rank_bins() -> None:
    target = np.linspace(-5.0, 5.0, 100)
    labels = FittedEnsembleRoutingReplay._regression_stratification_labels(
        target,
        n_folds=5,
        requested_bins=10,
    )
    splits = FittedEnsembleRoutingReplay._selection_splits(
        target,
        problem_type="regression",
        n_folds=5,
        random_state=91,
        regression_stratification_bins=10,
    )

    assert set(labels) == set(range(10))
    assert all(set(labels[evaluation]) == set(range(10)) for _, evaluation in splits)


def test_tail_guarded_config_keeps_classification_and_validates_tail_policy() -> None:
    config = CrossFittedRoutingGeometryExperimentConfig(
        regression_tasks=("regression",),
        classification_tasks=("classification",),
        regression_tail_guard=True,
        tail_risk_quantile=0.9,
        regression_stratification_bins=10,
        show_progress=False,
    )
    orchestrator = CrossFittedRoutingGeometryExperimentOrchestrator(config)

    assert orchestrator.selector.spec.tail_guard_primary_metrics == ("rmse",)

    with pytest.raises(ValueError, match="tail_risk_quantile"):
        CrossFittedRoutingGeometryExperimentConfig(tail_risk_quantile=0.4)
    with pytest.raises(ValueError, match="regression_stratification_bins is required"):
        CrossFittedRoutingGeometryExperimentConfig(regression_tail_guard=True)


def test_classification_replay_uses_probability_metrics() -> None:
    distances = RoutingDistanceContract(
        partition_names=("chunk_0", "chunk_1"),
        values=np.asarray([[0.0, 2.0], [2.0, 0.0], [0.0, 2.0], [2.0, 0.0]]),
        kind="distance",
    )
    expert_proba = np.asarray(
        [
            [[0.95, 0.05], [0.10, 0.90]],
            [[0.90, 0.10], [0.05, 0.95]],
            [[0.90, 0.10], [0.15, 0.85]],
            [[0.85, 0.15], [0.10, 0.90]],
        ]
    )
    request = RoutingReplayRequest(
        arm_name="binary",
        geometry_spec=PartitionGeometrySpec(uniform_shrinkage=0.0),
        distances=distances,
        target=np.asarray([0, 1, 0, 1]),
        expert_outputs=expert_proba,
        problem_type="classification",
        classes=(0, 1),
    )

    result = RoutingReplayEvaluator().evaluate(request, temperature=0.2)

    assert result.primary_metric == "roc_auc"
    assert result.metrics["roc_auc"] == pytest.approx(1.0)
    assert np.isfinite(result.metrics["log_loss"])
    assert np.isfinite(result.metrics["brier_score"])
    assert result.metrics["brier_score"] == pytest.approx(
        classification_brier_score(
            request.target,
            result.blended_output,
            request.classes,
        )
    )
    assert "mean_top1_margin" in result.routing.diagnostics
    assert np.allclose(result.blended_output.sum(axis=1), 1.0)


def test_validation_geometry_selector_is_metric_aware_and_order_independent() -> None:
    policy = default_validation_selector_policy()
    results = (
        _selector_result("A2_median_scaled_euclidean", "rmse", 2.0),
        _selector_result("A5_gmm_posterior", "rmse", 1.0),
        _selector_result("A6_cosine_negative_control", "rmse", 3.0),
    )
    selector = ValidationRoutingGeometrySelector()

    forward = selector.select(results, policy)
    reverse = selector.select(tuple(reversed(results)), policy)

    assert forward.selected.arm_name == "A5_gmm_posterior"
    assert reverse.selected.arm_name == forward.selected.arm_name
    assert forward.improvement_vs_fallback == pytest.approx(1.0)
    assert forward.fallback_used is False


def test_validation_geometry_selector_maximizes_roc_auc() -> None:
    policy = default_validation_selector_policy()
    selection = ValidationRoutingGeometrySelector().select(
        (
            _selector_result("A2_median_scaled_euclidean", "roc_auc", 0.80),
            _selector_result("A5_gmm_posterior", "roc_auc", 0.82),
            _selector_result("A6_cosine_negative_control", "roc_auc", 0.81),
        ),
        policy,
    )

    assert selection.selected.arm_name == "A5_gmm_posterior"
    assert selection.improvement_vs_fallback == pytest.approx(0.02)


def test_validation_geometry_selector_keeps_fallback_below_minimum_improvement() -> None:
    policy = RoutingGeometrySelectionPolicy(
        candidate_arm_names=("fallback", "candidate"),
        fallback_arm_name="fallback",
        minimum_improvement=0.01,
    )
    selection = ValidationRoutingGeometrySelector().select(
        (
            _selector_result("fallback", "roc_auc", 0.80),
            _selector_result("candidate", "roc_auc", 0.805),
        ),
        policy,
    )

    assert selection.selected.arm_name == "fallback"
    assert selection.fallback_used is True
    assert selection.reason == "fallback_minimum_improvement"


def test_validation_geometry_selector_rejects_incomplete_or_mixed_results() -> None:
    selector = ValidationRoutingGeometrySelector()
    policy = RoutingGeometrySelectionPolicy(
        candidate_arm_names=("fallback", "candidate"),
        fallback_arm_name="fallback",
    )

    with pytest.raises(ValueError, match="Missing validation results"):
        selector.select((_selector_result("fallback", "rmse", 1.0),), policy)
    with pytest.raises(ValueError, match="same primary metric"):
        selector.select(
            (
                _selector_result("fallback", "rmse", 1.0),
                _selector_result("candidate", "roc_auc", 0.8),
            ),
            policy,
        )


def test_default_phase_a_arms_have_stable_order() -> None:
    arms = default_routing_geometry_arms()

    assert tuple(arm.name.split("_", 1)[0] for arm in arms) == (
        "A0",
        "A1",
        "A2",
        "A3",
        "A4",
        "A5",
        "A6",
    )
    assert arms[0].use_validation_priors is False
    assert arms[1].temperature_candidates == (1.0,)


def test_default_phase_a_selector_arms_have_stable_order_and_a2_fallback() -> None:
    arms = default_validation_selector_arms()
    policy = default_validation_selector_policy()

    assert tuple(arm.name for arm in arms) == policy.candidate_arm_names
    assert policy.fallback_arm_name == "A2_median_scaled_euclidean"


def test_fitted_ensemble_adapter_replays_without_model_fit() -> None:
    rng = np.random.default_rng(81)
    frame = pd.DataFrame(rng.normal(size=(50, 4)), columns=[f"x_{idx}" for idx in range(4)])
    sampler = RMTContractionTensorSampler(
        n_partitions=2,
        n_views=2,
        projection_dim=2,
        backend="numpy",
        random_state=81,
        show_progress=False,
    ).fit(frame)
    target = frame.iloc[:12, 0].to_numpy()
    names = tuple(sampler.partition_names_)
    outputs = np.column_stack(
        [target + 0.1 * (idx + 1) for idx in range(len(names))]
    )

    class _FittedEnsemble:
        problem = "regression"
        classes_ = None
        partitioner = sampler

        @staticmethod
        def export_expert_outputs(features, *, stage="validation"):
            assert stage == "validation"
            return names, outputs

        @staticmethod
        def validation_prior_weights():
            return np.full(len(names), 1.0 / len(names))

    selected_arms = default_routing_geometry_arms()[:3]
    results = FittedEnsembleRoutingReplay(show_progress=False).run_validation(
        ensemble=_FittedEnsemble(),
        X_val=frame.iloc[:12],
        y_val=target,
        arms=selected_arms,
    )

    assert [result.arm_name for result in results] == [arm.name for arm in selected_arms]
    assert all(result.primary_metric == "rmse" for result in results)
    assert all(np.isfinite(result.primary_value) for result in results)


def test_cross_fitted_adapter_uses_stratified_classification_folds() -> None:
    rng = np.random.default_rng(82)
    frame = pd.DataFrame(rng.normal(size=(60, 4)), columns=[f"x_{idx}" for idx in range(4)])
    sampler = RMTContractionTensorSampler(
        n_partitions=2,
        n_views=2,
        projection_dim=2,
        backend="numpy",
        random_state=82,
        show_progress=False,
    ).fit(frame)
    target = np.tile(np.asarray([0, 1]), 30)
    names = tuple(sampler.partition_names_)
    expert_outputs = np.empty((len(frame), len(names), 2), dtype=float)
    for expert_index in range(len(names)):
        positive = np.where(
            target == 1,
            0.75 - 0.05 * expert_index,
            0.25 + 0.05 * expert_index,
        )
        expert_outputs[:, expert_index, 1] = positive
        expert_outputs[:, expert_index, 0] = 1.0 - positive

    class _FittedEnsemble:
        problem = "classification"
        classes_ = np.asarray([0, 1])
        partitioner = sampler

        @staticmethod
        def export_expert_outputs(features, *, stage="validation"):
            assert stage == "validation"
            return names, expert_outputs

        @staticmethod
        def validation_prior_weights():
            return np.full(len(names), 1.0 / len(names))

    arms = default_validation_selector_arms()
    selector = CrossFittedRoutingGeometrySelector(
        CrossFittedRoutingGeometrySelectionSpec(
            min_folds=3,
            bootstrap_iterations=100,
        )
    )
    selection = FittedEnsembleRoutingReplay(
        show_progress=False
    ).select_geometry_cross_fitted(
        ensemble=_FittedEnsemble(),
        X_val=frame,
        y_val=target,
        arms=arms,
        selector=selector,
        n_folds=3,
        random_state=82,
    )

    assert len(selection.fold_scores) == 9
    assert {score.primary_metric for score in selection.fold_scores} == {"roc_auc"}
    assert sum(selection.expert_priors) == pytest.approx(1.0)
    assert selection.selected_arm.name in {arm.name for arm in arms}


def test_targeted_runner_persists_incremental_synthetic_results(tmp_path) -> None:
    config = RoutingGeometryExperimentConfig(
        regression_suite=None,
        classification_suite=None,
        regression_tasks=(),
        classification_tasks=(),
        models=("ridge",),
        budget_ratios=(0.20,),
        seeds=(42,),
        n_partitions=3,
        output_dir=tmp_path,
        show_progress=False,
    )

    class _SyntheticOrchestrator(RoutingGeometryExperimentOrchestrator):
        def _load_datasets(self):
            return [make_synthetic_regression_smoke_dataset(42)]

    result = _SyntheticOrchestrator(
        config,
        arms=default_routing_geometry_arms()[:2],
    ).run()

    assert result.shape[0] == 2
    assert set(result["status"]) == {"completed"}
    assert (tmp_path / "routing_geometry_runs.jsonl").exists()
    assert (tmp_path / "routing_geometry_replay.csv").exists()
    assert '"status": "completed"' in (tmp_path / "run_meta.json").read_text(encoding="utf-8")

    resumed = _SyntheticOrchestrator(
        config,
        arms=default_routing_geometry_arms()[:2],
    ).run()

    assert resumed.shape[0] == 2
    lines = (tmp_path / "routing_geometry_runs.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(lines) == 2


def test_validation_selector_runner_uses_independent_holdout_and_resumes(tmp_path) -> None:
    config = RoutingGeometrySelectorExperimentConfig(
        regression_suite=None,
        classification_suite=None,
        regression_tasks=(),
        classification_tasks=(),
        models=("ridge",),
        budget_ratios=(0.20,),
        seeds=(42,),
        n_partitions=3,
        output_dir=tmp_path,
        show_progress=False,
    )

    class _SyntheticOrchestrator(ValidationRoutingGeometryExperimentOrchestrator):
        def _load_datasets(self):
            return [make_synthetic_regression_smoke_dataset(42)]

    orchestrator = _SyntheticOrchestrator(config)
    result = orchestrator.run()

    assert result.shape[0] == 4
    assert set(result["status"]) == {"completed"}
    selector_row = result.loc[
        result["arm_name"] == "A7_validation_selected_top3"
    ].iloc[0]
    assert selector_row["selected_arm_name"] in {
        arm.name for arm in default_validation_selector_arms()
    }
    assert selector_row["n_calibration"] > 0
    assert selector_row["n_selection"] > 0
    assert selector_row["calibration_primary_metric"] == "rmse"
    assert selector_row["validation_primary_metric"] == "rmse"
    assert selector_row["test_primary_metric"] == "rmse"

    resumed = _SyntheticOrchestrator(config).run()
    assert resumed.shape[0] == 4
    lines = (tmp_path / "routing_geometry_runs.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(lines) == 4


def test_cross_fitted_selector_runner_persists_fold_evidence_and_resumes(tmp_path) -> None:
    config = CrossFittedRoutingGeometryExperimentConfig(
        regression_suite=None,
        classification_suite=None,
        regression_tasks=(),
        classification_tasks=(),
        models=("ridge",),
        budget_ratios=(0.20,),
        seeds=(42,),
        n_partitions=3,
        selection_folds=3,
        bootstrap_iterations=200,
        output_dir=tmp_path,
        show_progress=False,
    )

    class _SyntheticOrchestrator(CrossFittedRoutingGeometryExperimentOrchestrator):
        def _load_datasets(self):
            return [make_synthetic_regression_smoke_dataset(42)]

    result = _SyntheticOrchestrator(config).run()

    assert result.shape[0] == 4
    assert set(result["status"]) == {"completed"}
    selector_row = result.loc[
        result["arm_name"] == "A8_cross_fitted_selected"
    ].iloc[0]
    assert selector_row["selected_arm_name"] in {
        arm.name for arm in default_validation_selector_arms()
    }
    assert selector_row["selection_status"] in {"selected", "fallback_to_a2"}

    fold_table = pd.read_csv(tmp_path / "geometry_selection_fold_losses.csv")
    summary_table = pd.read_csv(tmp_path / "geometry_selection_summary.csv")
    selection_lines = (tmp_path / "cross_fitted_geometry_runs.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    assert fold_table.shape[0] == 9
    assert summary_table.shape[0] == 2
    assert fold_table["tail_metric"].eq("tail_mean_absolute_error").all()
    assert fold_table["tail_quantile"].eq(0.9).all()
    assert "tail_mean_relative_gain" in summary_table
    assert len(selection_lines) == 1

    (tmp_path / "geometry_selection_fold_losses.csv").unlink()
    (tmp_path / "geometry_selection_summary.csv").unlink()
    resumed = _SyntheticOrchestrator(config).run()
    assert resumed.shape[0] == 4
    assert len(
        (tmp_path / "routing_geometry_runs.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ) == 4
    assert pd.read_csv(tmp_path / "geometry_selection_fold_losses.csv").shape[0] == 9
    assert pd.read_csv(tmp_path / "geometry_selection_summary.csv").shape[0] == 2
    assert len(
        (tmp_path / "cross_fitted_geometry_runs.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ) == 1


def test_tail_guarded_selector_runner_persists_a9_tail_evidence(tmp_path) -> None:
    config = CrossFittedRoutingGeometryExperimentConfig(
        regression_suite=None,
        classification_suite=None,
        regression_tasks=(),
        classification_tasks=(),
        models=("ridge",),
        budget_ratios=(0.20,),
        seeds=(42,),
        n_partitions=3,
        selection_folds=3,
        selector_arm_name="A9_tail_guarded_cross_fitted_selected",
        bootstrap_iterations=200,
        regression_tail_guard=True,
        tail_risk_quantile=0.90,
        regression_stratification_bins=5,
        output_dir=tmp_path,
        show_progress=False,
    )

    class _SyntheticOrchestrator(CrossFittedRoutingGeometryExperimentOrchestrator):
        def _load_datasets(self):
            return [make_synthetic_regression_smoke_dataset(42)]

    result = _SyntheticOrchestrator(config).run()

    assert result.shape[0] == 4
    selector_row = result.loc[
        result["arm_name"] == "A9_tail_guarded_cross_fitted_selected"
    ].iloc[0]
    assert selector_row["selection_status"] in {"selected", "fallback_to_a2"}
    summary = pd.read_csv(tmp_path / "geometry_selection_summary.csv")
    assert summary["tail_metric"].eq("tail_mean_absolute_error").all()
    assert summary["tail_quantile"].eq(0.9).all()
    assert summary["tail_confidence_lower"].notna().all()


def test_tail_guarded_runner_rejects_a8_resume_directory(tmp_path) -> None:
    (tmp_path / "run_meta.json").write_text(
        json.dumps(
            {
                "config": {
                    "selector_arm_name": "A8_cross_fitted_selected",
                    "selection_folds": 5,
                }
            }
        ),
        encoding="utf-8",
    )
    config = CrossFittedRoutingGeometryExperimentConfig(
        selector_arm_name="A9_tail_guarded_cross_fitted_selected",
        regression_tail_guard=True,
        regression_stratification_bins=10,
        output_dir=tmp_path,
        show_progress=False,
    )

    with pytest.raises(ResumeCompatibilityError) as exc_info:
        CrossFittedRoutingGeometryExperimentOrchestrator(
            config
        )._initialize_artifacts()

    assert exc_info.value.code == "routing_geometry_resume_selector_mismatch"


def test_validation_selector_stratifies_only_when_both_splits_can_preserve_classes() -> None:
    safe_target = pd.Series([0, 0, 0, 1, 1, 1])
    rare_target = pd.Series([0, 0, 0, 1])

    safe = ValidationRoutingGeometryExperimentOrchestrator._selection_stratify_target(
        safe_target,
        problem_type="classification",
        selector_fraction=0.5,
    )
    rare = ValidationRoutingGeometryExperimentOrchestrator._selection_stratify_target(
        rare_target,
        problem_type="classification",
        selector_fraction=0.5,
    )

    assert safe is safe_target
    assert rare is None


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
@pytest.mark.parametrize(
    ("metric", "kernel"),
    [
        ("median_scaled_euclidean", "softmax"),
        ("full_shrinkage_mahalanobis", "softmax"),
        ("full_shrinkage_mahalanobis", "gmm_posterior"),
    ],
)
def test_matrix_and_tensor_routing_kernels_are_consistent(metric: str, kernel: str) -> None:
    embedding, labels, partitions, mapping = _geometry_inputs()
    geometry = PartitionGeometryBuilder().build(
        embedding=embedding,
        cluster_labels=labels,
        partitions=partitions,
        partition_to_cluster=mapping,
        spec=PartitionGeometrySpec(
            metric=metric,
            kernel=kernel,
            covariance_shrinkage=0.2,
            uniform_shrinkage=0.0,
        ),
    )
    evaluation = np.asarray([[-1.5, 0.0], [1.5, 0.0], [0.0, 0.2]])

    _, matrix_weights = route_partition_geometry(
        backend=MatrixRMTBackend(),
        embedding=evaluation,
        geometry=geometry,
    )
    _, tensor_weights = route_partition_geometry(
        backend=TensorRMTBackend(device="cpu", dtype="float64"),
        embedding=evaluation,
        geometry=geometry,
    )

    assert np.allclose(matrix_weights.weights, tensor_weights.weights, atol=1e-6)
