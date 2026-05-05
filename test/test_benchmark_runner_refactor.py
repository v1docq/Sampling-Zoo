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
