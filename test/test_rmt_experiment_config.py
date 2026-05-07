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
    assert "approx_rank" not in configs["rmt_contraction"]
    assert configs["rmt_contraction"]["initial_rank_fraction"] == 0.25
    assert configs["rmt_contraction"]["rank_selection_method"] == "explained_variance"
    assert configs["rmt_contraction"]["explained_variance_threshold"] == 0.95
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

        def _build_strategy_configs(self):
            self.calls.append("build_strategy_configs")
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
        "build_strategy_configs",
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
    assert (tmp_path / "rmt_raw_runs.csv").exists()
    assert (tmp_path / "sample_efficiency_curve.csv").exists()
    assert (tmp_path / "minimal_effective_budget.csv").exists()
