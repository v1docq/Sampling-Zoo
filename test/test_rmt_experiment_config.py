from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from examples.benchmark.benchmark_sampling_strategies import make_chunking_strategy_configs
from examples.benchmark.rmt_regression_medium_datasets import (
    make_rmt_experiment_strategy_configs as make_medium_rmt_experiment_strategy_configs,
    make_rmt_strategy_grid as make_medium_rmt_strategy_grid,
)
from examples.benchmark.run_rmt_contraction_regression_experiment import (
    RMTRegressionExperimentConfig,
    RMTRegressionExperimentOrchestrator,
    build_rmt_report_tables,
    make_rmt_experiment_strategy_configs,
    make_rmt_strategy_grid,
)
from sampling_zoo.core.utils.sampling_ensemble import SamplingEnsemble


def test_chunking_configs_support_rmt_and_experiment_metadata() -> None:
    configs = make_chunking_strategy_configs(
        problem_type="regression",
        strategy_names=("rmt_contraction", "feature_clustering", "random", "difficulty"),
        n_partitions=4,
        seed=42,
        ensemble_method="routed_weighted",
        chunk_fraction=0.5,
        budget_ratio=0.1,
        force_chunking=True,
    )

    assert configs["rmt_contraction"]["chunk_fraction"] == 0.5
    assert configs["rmt_contraction"]["backend"] == "auto"
    assert configs["rmt_contraction"]["n_views"] == "auto"
    assert configs["rmt_contraction"]["n_views_policy"] == "auto"
    assert configs["rmt_contraction"]["view_strategy"] == "gaussian"
    assert configs["rmt_contraction"]["embedding_mode"] == "sv_scaled"
    assert configs["rmt_contraction"]["partition_selection_method"] == "auto"
    assert configs["rmt_contraction"]["cluster_selection_metric"] == "balanced_silhouette"
    assert configs["rmt_contraction"]["cluster_ensemble_method"] == "weighted_vote"
    assert "gmm" in configs["rmt_contraction"]["cluster_algorithms"]
    assert configs["rmt_contraction"]["max_partitions"] >= 4
    assert configs["rmt_contraction"]["max_cluster_imbalance_ratio"] == 5.0
    assert configs["rmt_contraction"]["min_cluster_fraction"] == 0.05
    assert "approx_rank" not in configs["rmt_contraction"]
    assert configs["rmt_contraction"]["initial_rank_fraction"] == 0.25
    assert configs["rmt_contraction"]["rank_selection_method"] == "explained_variance"
    assert configs["rmt_contraction"]["explained_variance_threshold"] == 0.95
    assert configs["rmt_contraction"]["null_diagnostic_enabled"] is False
    assert configs["rmt_contraction"]["null_primary_policy"] == "feature_permutation"
    assert "view_resampling" in configs["rmt_contraction"]["null_model_policies"]
    assert configs["rmt_contraction"]["subspace_diagnostic_enabled"] is False
    assert configs["rmt_contraction"]["subspace_resamples"] == 16
    assert configs["rmt_contraction"]["subspace_quantile"] == 0.90
    assert configs["rmt_contraction"]["subspace_max_rank"] == 64
    assert configs["feature_clustering"]["experiment_chunk_fraction"] == 0.5
    assert "chunk_fraction" not in configs["feature_clustering"]
    assert configs["random"]["budget_ratio"] == 0.1
    assert configs["difficulty"]["problem"] == "regression"


def test_routed_weighted_falls_back_for_non_routing_sampler() -> None:
    rng = np.random.default_rng(123)
    X = pd.DataFrame(rng.normal(size=(90, 5)), columns=[f"x_{idx}" for idx in range(5)])
    y = pd.Series(X["x_0"] * 3.0 - X["x_1"] + rng.normal(scale=0.1, size=len(X)), name="target")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=7)

    ensemble = SamplingEnsemble(
        problem="regression",
        partitioner_config={
            "strategy": "random",
            "n_partitions": 3,
            "random_state": 42,
            "budget_ratio": 0.8,
        },
        model_factory=lambda: RandomForestRegressor(n_estimators=8, random_state=42),
        ensemble_method="routed_weighted",
        show_progress=False,
    )

    ensemble.train_partition_models(
        X_train=X_train.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        X_val=X_val.reset_index(drop=True),
        y_val=y_val.reset_index(drop=True),
        class_samples=None,
        cv_fold=1,
        validation_metric="rmse",
        train_all_chunks=True,
        save_models_to_disk=False,
    )
    predictions = ensemble.ensemble_predict(X_val.reset_index(drop=True))

    assert predictions.shape[0] == len(X_val)
    assert np.all(np.isfinite(predictions))
    assert ensemble.partition_diagnostics_["chunk_size_imbalance"]["max"] > 0
    assert "target_drift_summary" in ensemble.partition_diagnostics_
    assert "routing" in ensemble.validation_diagnostics_
    assert "local_partition_metrics" in ensemble.validation_diagnostics_
    routing_diagnostics = ensemble.build_routing_diagnostics(X_val.reset_index(drop=True))
    assert routing_diagnostics["n_rows"] == len(X_val)
    assert routing_diagnostics["n_models"] == len(ensemble.models)
    assert routing_diagnostics["mean_normalized_entropy"] >= 0.0


class _ToySoftRouter:
    partition_names_ = ["chunk_0", "chunk_1", "chunk_2"]

    def predict_partition_proba(self, features):
        route = np.asarray(features["route"], dtype=int)
        proba = np.zeros((len(route), 3), dtype=float)
        proba[np.arange(len(route)), route] = 1.0
        return proba


class _NoisyEmbeddingRouter:
    partition_names_ = ["chunk_0", "chunk_1", "chunk_2"]

    def predict_partition_proba(self, features):
        route = np.asarray(features["route"], dtype=int)
        proba = np.full((len(route), 3), 0.15, dtype=float)
        proba[np.arange(len(route)), route] = 0.70
        return proba

    def transform_embedding(self, features):
        route = np.asarray(features["route"], dtype=int)
        embedding = np.zeros((len(route), 3), dtype=float)
        embedding[np.arange(len(route)), route] = 1.0
        return embedding


class _MeanRegressor:
    def fit(self, X, y):
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_, dtype=float)


def test_routed_weighted_keeps_soft_routed_experts_and_fits_router_head() -> None:
    X_val = pd.DataFrame({"route": [0, 0, 1, 1, 2, 2]})
    y_val = pd.Series([0.0, 0.0, 10.0, 10.0, 20.0, 20.0])
    predictions = [
        [0.0, 0.0, 100.0, 100.0, 100.0, 100.0],
        [100.0, 100.0, 10.0, 10.0, 100.0, 100.0],
        [100.0, 100.0, 100.0, 100.0, 20.0, 20.0],
    ]
    ensemble = SamplingEnsemble(
        problem="regression",
        partitioner_config={
            "strategy": "rmt_contraction",
            "router": "learned_head",
            "random_state": 42,
            "router_n_estimators": 4,
        },
        model_factory=lambda: RandomForestRegressor(n_estimators=2, random_state=42),
        ensemble_method="routed_weighted",
        show_progress=False,
    )
    ensemble.partitioner = _ToySoftRouter()
    ensemble.models = [
        {
            "name": f"chunk_{idx}",
            "metrics": {"rmse": 75.0},
            "val_predictions": np.asarray(pred),
        }
        for idx, pred in enumerate(predictions)
    ]

    ensemble._finalize_partition_training(
        partitions={
            f"chunk_{idx}": {"feature": X_val.iloc[idx * 2:(idx + 1) * 2], "target": y_val.iloc[idx * 2:(idx + 1) * 2]}
            for idx in range(3)
        },
        X_val=X_val,
        y_val=y_val,
        metric_is_better=lambda current, best: current < best,
        validation_metric="rmse",
    )

    assert len(ensemble.models) == 3
    assert ensemble.validation_diagnostics_["selection_policy"] == "moe_keep_routed_experts"
    assert ensemble.validation_diagnostics_["router_head"]["status"] == "fitted"
    assert ensemble.validation_diagnostics_["router_head"]["training_accuracy"] == 1.0
    assert all("local_metrics" in model_info for model_info in ensemble.models)


def test_routed_weighted_constrained_gating_uses_validation_predictions() -> None:
    routes = np.tile(np.arange(3), 20)
    X_val = pd.DataFrame({"route": routes})
    y_val = pd.Series(np.where(routes == 0, 0.0, np.where(routes == 1, 10.0, 20.0)))
    predictions = [
        np.where(routes == 0, 0.0, 100.0),
        np.where(routes == 1, 10.0, 100.0),
        np.where(routes == 2, 20.0, 100.0),
    ]
    ensemble = SamplingEnsemble(
        problem="regression",
        partitioner_config={
            "strategy": "rmt_contraction",
            "router": "constrained_gating",
            "random_state": 42,
            "gating_epochs": 80,
            "gating_hidden_dim": 12,
            "gating_lr": 0.05,
            "gating_kl_weight": 0.001,
            "gating_balance_weight": 0.0,
            "gating_batch_size": 32,
            "gating_device": "cpu",
        },
        model_factory=lambda: RandomForestRegressor(n_estimators=2, random_state=42),
        ensemble_method="routed_weighted",
        show_progress=False,
    )
    ensemble.partitioner = _NoisyEmbeddingRouter()
    ensemble.models = [
        {
            "name": f"chunk_{idx}",
            "metrics": {"rmse": 75.0},
            "val_predictions": np.asarray(pred, dtype=float),
        }
        for idx, pred in enumerate(predictions)
    ]

    ensemble._finalize_partition_training(
        partitions={
            f"chunk_{idx}": {"feature": X_val.iloc[routes == idx], "target": y_val.iloc[routes == idx]}
            for idx in range(3)
        },
        X_val=X_val,
        y_val=y_val,
        metric_is_better=lambda current, best: current < best,
        validation_metric="rmse",
    )

    router_diagnostics = ensemble.validation_diagnostics_["router"]
    if router_diagnostics["status"] == "skipped_missing_torch":
        return

    assert len(ensemble.models) == 3
    assert ensemble.validation_diagnostics_["selection_policy"] == "moe_keep_routed_experts"
    assert router_diagnostics["status"] == "fitted"
    assert router_diagnostics["training_rmse"] < router_diagnostics["prior_rmse"]
    assert ensemble.router.weights_are_final()
    predictions = ensemble.ensemble_predict(X_val, stage="validation")
    assert predictions.shape == y_val.shape
    assert np.all(np.isfinite(predictions))


def test_routed_weighted_em_refinement_improves_or_restores_best_metric() -> None:
    routes = np.tile(np.arange(3), 20)
    X_val = pd.DataFrame({"route": routes})
    y_val = pd.Series(np.where(routes == 0, 0.0, np.where(routes == 1, 10.0, 20.0)))
    base_model = _MeanRegressor().fit(X_val, y_val)
    base_predictions = base_model.predict(X_val)
    ensemble = SamplingEnsemble(
        problem="regression",
        partitioner_config={
            "strategy": "rmt_contraction",
            "router": "spectral",
            "random_state": 42,
            "routing_refinement": "em_retraining",
            "em_max_iterations": 2,
            "em_min_improvement": 1e-6,
            "em_min_partition_size": 2,
            "em_assignment_policy": "hard_top1",
            "em_refit_router": False,
            "em_keep_best": True,
        },
        model_factory=_MeanRegressor,
        ensemble_method="routed_weighted",
        show_progress=False,
    )
    ensemble.partitioner = _NoisyEmbeddingRouter()
    ensemble.models = [
        {
            "name": f"chunk_{idx}",
            "model": base_model,
            "data_size": len(X_val),
            "metrics": {"rmse": float(np.sqrt(np.mean((base_predictions - y_val.to_numpy()) ** 2)))},
            "val_predictions": base_predictions.copy(),
        }
        for idx in range(3)
    ]
    ensemble.partition_metrics = {
        model_info["name"]: dict(model_info["metrics"])
        for model_info in ensemble.models
    }

    ensemble._finalize_partition_training(
        partitions={"chunk_all": {"feature": X_val, "target": y_val}},
        X_val=X_val,
        y_val=y_val,
        metric_is_better=lambda current, best: current < best,
        validation_metric="rmse",
    )

    refinement = ensemble.validation_diagnostics_["routing_refinement"]
    assert refinement["status"] == "completed"
    assert refinement["best_iteration"] >= 1
    assert refinement["metric_improvement"] > 0.0
    assert refinement["final_imbalance_ratio"] is not None
    assert refinement["final_imbalance_ratio"] <= 2.0


def test_rmt_strategy_grid_points_are_stable() -> None:
    grid = make_rmt_strategy_grid(
        strategies=("rmt_contraction",),
        ensemble_methods=("voting", "routed_weighted"),
        budget_ratios=(0.01,),
    )

    assert [point.config_name for point in grid] == [
        "rmt_contraction__voting__budget_01",
        "rmt_contraction__routed_weighted__budget_01",
    ]


def test_rmt_runner_strategy_configs_vary_only_budget_ratio() -> None:
    configs = make_rmt_experiment_strategy_configs(
        problem_type="regression",
        strategies=("rmt_contraction",),
        ensemble_methods=("voting",),
        budget_ratios=(0.1, 0.2),
        n_partitions=3,
        seed=42,
        show_progress=False,
    )

    assert [name for name in configs if name != "full_dataset"] == [
        "rmt_contraction__voting__budget_10",
        "rmt_contraction__voting__budget_20",
    ]
    assert configs["full_dataset"]["force_direct_model"] is True
    assert configs["rmt_contraction__voting__budget_10"]["budget_ratio"] == 0.1
    assert "chunk_fraction" not in configs["rmt_contraction__voting__budget_10"]
    assert "experiment_chunk_fraction" not in configs["rmt_contraction__voting__budget_10"]


def test_medium_rmt_runner_strategy_configs_vary_view_strategy_for_rmt_only() -> None:
    grid = make_medium_rmt_strategy_grid(
        strategies=("rmt_contraction", "random"),
        ensemble_methods=("voting",),
        budget_ratios=(0.1,),
        view_strategies=("subsample", "gaussian"),
    )

    assert [point.config_name for point in grid] == [
        "rmt_contraction__view_subsample__voting__budget_10",
        "rmt_contraction__view_gaussian__voting__budget_10",
        "random__voting__budget_10",
    ]

    configs = make_medium_rmt_experiment_strategy_configs(
        problem_type="regression",
        strategies=("rmt_contraction", "random"),
        ensemble_methods=("voting",),
        budget_ratios=(0.1,),
        view_strategies=("subsample", "gaussian"),
        n_partitions=3,
        seed=42,
        show_progress=False,
    )

    assert configs["rmt_contraction__view_subsample__voting__budget_10"]["view_strategy"] == "subsample"
    assert configs["rmt_contraction__view_gaussian__voting__budget_10"]["view_strategy"] == "gaussian"
    assert "view_strategy" not in configs["random__voting__budget_10"]


def test_medium_rmt_runner_adds_router_ablation_only_for_routed_rmt() -> None:
    grid = make_medium_rmt_strategy_grid(
        strategies=("rmt_contraction", "random"),
        ensemble_methods=("voting", "routed_weighted"),
        budget_ratios=(0.1,),
        view_strategies=("gaussian",),
        router_modes=("spectral", "constrained_gating"),
    )

    assert [point.config_name for point in grid] == [
        "rmt_contraction__view_gaussian__voting__budget_10",
        "rmt_contraction__view_gaussian__routed_weighted__router_spectral__budget_10",
        "rmt_contraction__view_gaussian__routed_weighted__router_constrained_gating__budget_10",
        "random__voting__budget_10",
        "random__routed_weighted__budget_10",
    ]

    configs = make_medium_rmt_experiment_strategy_configs(
        problem_type="regression",
        strategies=("rmt_contraction",),
        ensemble_methods=("routed_weighted",),
        budget_ratios=(0.1,),
        view_strategies=("gaussian",),
        n_partitions=3,
        seed=42,
        show_progress=False,
        router_modes=("spectral", "constrained_gating"),
    )

    assert configs[
        "rmt_contraction__view_gaussian__routed_weighted__router_spectral__budget_10"
    ]["router"] == "spectral"
    assert configs[
        "rmt_contraction__view_gaussian__routed_weighted__router_constrained_gating__budget_10"
    ]["router"] == "constrained_gating"
    assert configs[
        "rmt_contraction__view_gaussian__routed_weighted__router_constrained_gating__budget_10"
    ]["gating_hidden_dim"] == 64


def test_rmt_orchestrator_run_is_thin_sequence(tmp_path) -> None:
    class RecordingOrchestrator(RMTRegressionExperimentOrchestrator):
        def __init__(self) -> None:
            super().__init__(RMTRegressionExperimentConfig(show_progress=False, synthetic_smoke=True))
            self.calls: list[str] = []

        def _prepare_runtime(self) -> None:
            self.calls.append("prepare_runtime")

        def _create_logger(self):
            self.calls.append("create_logger")

            class _Paths:
                root = tmp_path
                metrics = tmp_path

            class _Logger:
                run_id = "test_run"
                paths = _Paths()

            return _Logger()

        def _create_runner(self, logger):
            self.calls.append("create_runner")
            assert logger.run_id == "test_run"
            return object()

        def _load_available_datasets(self):
            self.calls.append("load_datasets")
            return ["dataset"]

        def _build_strategy_grid(self):
            self.calls.append("build_strategy_grid")
            return {"strategy": {}}

        def _run_experiment(self, datasets, strategy_configs, runner):
            self.calls.append("run_experiment")
            assert datasets == ["dataset"]
            assert strategy_configs == {"strategy": {}}
            assert runner is not None
            return [{"record": 1}]

        def _build_report_artifacts(self, run_records, logger) -> None:
            self.calls.append("build_report_artifacts")
            assert run_records == [{"record": 1}]
            assert logger.run_id == "test_run"

        def _write_run_metadata(self, logger, run_records) -> None:
            self.calls.append("write_run_metadata")
            assert run_records == [{"record": 1}]
            assert logger.run_id == "test_run"

        def _announce_completion(self, logger):
            self.calls.append("announce_completion")
            return logger.paths.root

    orchestrator = RecordingOrchestrator()

    assert orchestrator.run() == tmp_path
    assert orchestrator.calls == [
        "prepare_runtime",
        "create_logger",
        "create_runner",
        "load_datasets",
        "build_strategy_grid",
        "run_experiment",
        "build_report_artifacts",
        "write_run_metadata",
        "announce_completion",
    ]


def test_sample_efficiency_summary_selects_minimal_budget(tmp_path) -> None:
    records = [
        {
            "dataset": "demo",
            "strategy_params": {
                "strategy": "full_dataset",
                "model": "random_forest",
                "ensemble_method": "full_dataset",
                "budget_ratio": 1.0,
                "experiment_chunk_fraction": 1.0,
            },
            "model_metrics": {"rmse": 10.0},
            "timings_sec": {"fit": 1.0, "inference": 0.1},
            "sample_stats": {"sample_size": 100},
            "extra": {},
        },
        {
            "dataset": "demo",
            "strategy_params": {
                "strategy": "rmt_contraction",
                "model": "random_forest",
                "ensemble_method": "routed_weighted",
                "budget_ratio": 0.05,
                "chunk_fraction": 0.5,
            },
            "model_metrics": {"rmse": 10.8},
            "timings_sec": {"fit": 0.5, "inference": 0.1},
            "sample_stats": {"sample_size": 5},
            "extra": {"sampler_diagnostics": {"leverage_entropy": 1.2, "effective_sample_count": 3.3}},
        },
        {
            "dataset": "demo",
            "strategy_params": {
                "strategy": "rmt_contraction",
                "model": "random_forest",
                "ensemble_method": "routed_weighted",
                "budget_ratio": 0.1,
                "chunk_fraction": 0.5,
            },
            "model_metrics": {"rmse": 10.4},
            "timings_sec": {"fit": 0.7, "inference": 0.1},
            "sample_stats": {"sample_size": 10},
            "extra": {"sampler_diagnostics": {"leverage_entropy": 1.4, "effective_sample_count": 4.0}},
        },
    ]

    tables = build_rmt_report_tables(records, tmp_path)
    minimal = tables["minimal_budget"]

    row = minimal[(minimal["delta"] == 0.05) & (minimal["sampler"] == "rmt_contraction")].iloc[0]
    assert row["budget_ratio"] == 0.1
    assert "chunk_fraction" not in tables["efficiency"].columns
    assert "embedding_mode" in tables["raw"].columns
    assert "target_mean_abs_drift_avg" in tables["raw"].columns
    assert "validation_mean_max_routing_proba" in tables["raw"].columns
    assert "test_mean_max_routing_proba" in tables["raw"].columns
    assert "null_model_status" in tables["raw"].columns
    assert "rank_by_null_edge" in tables["raw"].columns
    assert "rank_by_stability" in tables["raw"].columns
    assert "subspace_stability_status" in tables["raw"].columns
    assert "subspace_rank_source" in tables["raw"].columns
    assert "rank_by_subspace_stability" in tables["raw"].columns
    assert "subspace_max_angle_quantile_degrees" in tables["raw"].columns
    assert (tmp_path / "rmt_raw_runs.csv").exists()
    assert (tmp_path / "sample_efficiency_curve.csv").exists()
    assert (tmp_path / "minimal_effective_budget.csv").exists()
