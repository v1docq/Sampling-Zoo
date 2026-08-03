from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import numpy as np
import pytest
from sklearn.linear_model import Ridge

BENCHMARK_DIR = Path(__file__).resolve().parents[1] / "examples" / "benchmark"
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

from benchmark_datasets import RawDatasetBundle, RawDatasetMetadata  # noqa: E402
from benchmark_logging import BenchmarkLogger  # noqa: E402
from benchmark_runner import EnsembleChunkBenchmarkRunner  # noqa: E402
from sampling_zoo.core.experiment.contracts import (  # noqa: E402
    ModelStrategyScenarioGridContract,
    ModelStrategyScenarioSpec,
    PartitionSizeDiagnosticsContract,
    StrategySpec,
)
from sampling_zoo.core.experiment.errors import (  # noqa: E402
    InvalidExperimentConfigError,
)
from sampling_zoo.core.experiment.resume import (  # noqa: E402
    build_resume_plan,
    leaf_run_key_from_components,
)


def _tiny_regression_dataset() -> RawDatasetBundle:
    X = pd.DataFrame({"x0": [0.0, 1.0, 2.0, 3.0], "x1": [1.0, 1.0, 2.0, 2.0]})
    y = pd.Series([0.0, 1.0, 2.0, 3.0], name="target")
    return RawDatasetBundle(
        name="tiny_runner_dataset",
        problem_type="regression",
        target_name="target",
        source_path="memory://tiny_runner_dataset",
        X=X,
        y=y,
        metadata=RawDatasetMetadata(
            n_objects=len(X),
            n_features=X.shape[1],
            n_train_candidates=len(X),
            n_categorical=0,
            n_numeric=X.shape[1],
        ),
        feature_columns=list(X.columns),
        categorical_columns=[],
        numeric_columns=list(X.columns),
    )


def test_ensemble_chunk_runner_delegates_fold_work_to_executor(tmp_path) -> None:
    recorded: list[dict] = []
    logger = BenchmarkLogger(run_id="runner_refactor_test", artifacts_root=tmp_path)
    runner = EnsembleChunkBenchmarkRunner(logger=logger, cv_folds=2, show_progress=False, on_record=recorded.append)

    class FakeFoldExecutor:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []

        def run_strategy_folds(
            self,
            dataset,
            strategy_name,
            partitioner_config,
            model_name,
            model_factory,
            openml_split_data=None,
        ):
            self.calls.append((model_name, strategy_name))
            assert dataset.name == "tiny_runner_dataset"
            assert partitioner_config["strategy"] == strategy_name
            assert openml_split_data is None
            return [{"model": model_name, "strategy": strategy_name}]

    fake_executor = FakeFoldExecutor()
    runner.fold_executor = fake_executor

    records = runner.run_dataset(
        dataset=_tiny_regression_dataset(),
        strategy_configs={
            "random": {"strategy": "random"},
            "feature_clustering": {"strategy": "feature_clustering"},
        },
        model_pool={"ridge": object},
    )

    assert fake_executor.calls == [("ridge", "random"), ("ridge", "feature_clustering")]
    assert records == [
        {"model": "ridge", "strategy": "random"},
        {"model": "ridge", "strategy": "feature_clustering"},
    ]
    assert recorded == records


def test_ensemble_chunk_runner_executes_only_bound_scenarios(tmp_path) -> None:
    recorded: list[dict] = []
    runner = EnsembleChunkBenchmarkRunner(
        logger=BenchmarkLogger(
            run_id="scenario_grid_test",
            artifacts_root=tmp_path,
        ),
        cv_folds=2,
        show_progress=False,
        on_record=recorded.append,
    )

    class FakeFoldExecutor:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str, str]] = []

        def run_strategy_folds(
            self,
            dataset,
            strategy_name,
            partitioner_config,
            model_name,
            model_factory,
            openml_split_data=None,
        ):
            self.calls.append(
                (
                    model_name,
                    strategy_name,
                    partitioner_config["strategy"],
                )
            )
            return [{"model": model_name, "scenario": strategy_name}]

    fake_executor = FakeFoldExecutor()
    runner.fold_executor = fake_executor
    grid = ModelStrategyScenarioGridContract(
        scenarios=(
            ModelStrategyScenarioSpec(
                name="ridge__random",
                model_name="ridge",
                strategy=StrategySpec(
                    name="random",
                    config={"strategy": "random"},
                ),
            ),
            ModelStrategyScenarioSpec(
                name="forest__difficulty",
                model_name="forest",
                strategy=StrategySpec(
                    name="difficulty",
                    config={"strategy": "difficulty"},
                ),
            ),
        )
    )

    records = runner.run_scenario_grid(
        _tiny_regression_dataset(),
        grid,
        model_pool={"ridge": object, "forest": object},
    )

    assert fake_executor.calls == [
        ("ridge", "ridge__random", "random"),
        ("forest", "forest__difficulty", "difficulty"),
    ]
    assert records == recorded
    assert len(records) == 2

    with pytest.raises(InvalidExperimentConfigError):
        runner.run_scenario_grid(
            _tiny_regression_dataset(),
            grid,
            model_pool={"ridge": object},
        )


def test_ensemble_chunk_runner_skips_fully_completed_dataset(
    tmp_path,
) -> None:
    dataset = _tiny_regression_dataset()
    strategy_config = {"strategy": "random", "random_state": 42}
    records = []
    for split_label in ("fold_1", "fold_2"):
        leaf_run = leaf_run_key_from_components(
            dataset=dataset.name,
            split=split_label,
            model="ridge",
            strategy="random",
            strategy_config=strategy_config,
            seed=42,
        )
        records.append(
            {
                "dataset": dataset.name,
                "strategy_params": {
                    **strategy_config,
                    "model": "ridge",
                    "split_label": split_label,
                },
                "extra": {
                    "strategy": "random",
                    "model": "ridge",
                    "split_label": split_label,
                    "leaf_run_key": leaf_run.key,
                    "leaf_run": leaf_run.to_dict(),
                },
            }
        )
    resume_plan = build_resume_plan(
        records,
        policy="retry_failed",
        default_seed=42,
    )
    runner = EnsembleChunkBenchmarkRunner(
        logger=BenchmarkLogger(
            run_id="resume_skip_test",
            artifacts_root=tmp_path,
        ),
        cv_folds=2,
        seed=42,
        show_progress=False,
        resume_plan=resume_plan,
    )
    loader_called = False

    def _unexpected_load(_dataset):
        nonlocal loader_called
        loader_called = True
        raise AssertionError("completed dataset must not be loaded")

    runner._load_openml_split = _unexpected_load

    result = runner.run_dataset(
        dataset=dataset,
        strategy_configs={"random": strategy_config},
        model_pool={"ridge": object},
    )

    assert result == []
    assert loader_called is False
    assert runner.resume_diagnostics()["skipped_leaf_runs"] == 2


def test_ensemble_sample_stats_separate_budget_from_pruned_models(
    tmp_path,
) -> None:
    runner = EnsembleChunkBenchmarkRunner(
        logger=BenchmarkLogger(
            run_id="sample_accounting_test",
            artifacts_root=tmp_path,
        ),
        cv_folds=2,
        show_progress=False,
    )
    ensemble = SimpleNamespace(
        models=[{"data_size": 3}],
        partition_size_diagnostics_contract_=(
            PartitionSizeDiagnosticsContract(
                pre_budget_sizes={"chunk_0": 5, "chunk_1": 5},
                post_budget_sizes={"chunk_0": 3, "chunk_1": 3},
                requested_budget_size=6,
                selected_rows=6,
                unique_selected_rows=6,
                duplicate_rows=0,
            )
        ),
        partition_diagnostics_={
            "chunks": {
                "chunk_0": {"class_counts": {"0": 2, "1": 1}},
                "chunk_1": {"class_counts": {"0": 2, "1": 1}},
            }
        },
        class_coverage_repairs_={
            "chunk_0": {"rows_added": 1},
            "chunk_1": {"rows_added": 0},
        },
    )

    sample_stats, active_chunk_sizes = (
        runner.fold_executor._build_ensemble_sample_stats(
            ensemble=ensemble,
            y_train=pd.Series([0, 0, 0, 0, 0, 0, 1, 1, 1, 1]),
            problem_type="classification",
        )
    )

    assert sample_stats["sample_size"] == 6
    assert sample_stats["selected_rows"] == 6
    assert sample_stats["model_fit_rows_total"] == 7
    assert sample_stats["active_model_rows"] == 3
    assert sample_stats["selected_partition_count"] == 2
    assert sample_stats["active_model_count"] == 1
    assert sample_stats["class_distribution"] == {"0": 4, "1": 2}
    assert sample_stats["source_class_distribution"] == {"0": 6, "1": 4}
    assert active_chunk_sizes == [3]


def test_rmt_downstream_proxy_fold_emits_budget_and_runtime_contracts(
    tmp_path,
) -> None:
    rng = np.random.default_rng(17)
    X = pd.DataFrame(
        rng.normal(size=(120, 5)),
        columns=[f"x{index}" for index in range(5)],
    )
    y = pd.Series(2 * X["x0"] - X["x1"] + rng.normal(scale=0.2, size=120))
    dataset = RawDatasetBundle(
        name="rmt_contract_smoke",
        problem_type="regression",
        target_name="target",
        source_path="memory://rmt_contract_smoke",
        X=X,
        y=y,
        metadata=RawDatasetMetadata(
            n_objects=len(X),
            n_features=X.shape[1],
            n_train_candidates=len(X),
            n_categorical=0,
            n_numeric=X.shape[1],
        ),
        feature_columns=list(X.columns),
        categorical_columns=[],
        numeric_columns=list(X.columns),
    )
    runner = EnsembleChunkBenchmarkRunner(
        logger=BenchmarkLogger(
            run_id="rmt_contract_smoke",
            artifacts_root=tmp_path,
        ),
        cv_folds=2,
        seed=42,
        show_progress=False,
    )

    records = runner.run_dataset(
        dataset=dataset,
        strategy_configs={
            "rmt_downstream": {
                "strategy": "rmt_contraction",
                "force_chunking": True,
                "ensemble_method": "routed_weighted",
                "router": "spectral",
                "budget_ratio": 0.4,
                "n_partitions": 2,
                "partition_selection_method": "auto",
                "cluster_algorithms": ("kmeans",),
                "cluster_selection_metric": "downstream_proxy",
                "cluster_ensemble_method": "best_score",
                "min_partitions": 2,
                "max_partitions": 3,
                "min_auto_partition_size": 1,
                "budget_feasibility_mode": "hard",
                "min_sampled_rows_per_partition": 5,
                "budget_max_imbalance_ratio": 5.0,
                "budget_min_partition_fraction": 0.05,
                "include_single_partition_candidate": True,
                "downstream_proxy_shortlist_size": 2,
                "downstream_proxy_n_estimators": 8,
                "n_views": 2,
                "projection_dim": 2,
                "backend": "numpy",
            }
        },
        model_pool={"ridge": lambda: Ridge()},
    )

    assert len(records) == 2
    for record in records:
        assert record["model_metrics"]["rmse"] >= 0
        assert record["extra"]["budget_policy"]["applied_by"] == "partitioner"
        assert record["extra"]["budget_policy"]["feasible"] is True
        assert record["extra"]["runtime_contract"]["total_seconds"] > 0
        assert record["extra"]["partition_size_contract"]["selected_rows"] > 0
        assert record["extra"]["sampler_diagnostics"][
            "partition_selection_selected_candidate"
        ]["components"]["downstream_proxy"]["status"] == "ok"
