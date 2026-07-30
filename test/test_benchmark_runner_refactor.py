from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

BENCHMARK_DIR = Path(__file__).resolve().parents[1] / "examples" / "benchmark"
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

from benchmark_datasets import RawDatasetBundle, RawDatasetMetadata  # noqa: E402
from benchmark_logging import BenchmarkLogger  # noqa: E402
from benchmark_runner import EnsembleChunkBenchmarkRunner  # noqa: E402
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
