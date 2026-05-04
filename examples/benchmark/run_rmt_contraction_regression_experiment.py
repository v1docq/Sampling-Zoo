from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.datasets import make_regression

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
BENCHMARK_DIR = Path(__file__).resolve().parent
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

from benchmark_datasets import (  # noqa: E402
    OpenMLRawDatasetBundle,
    RawDatasetBundle,
    RawDatasetMetadata,
    load_suite_raw_datasets,
)
from benchmark_logging import BenchmarkLogger  # noqa: E402
from benchmark_models import make_model_pool  # noqa: E402
from benchmark_repo import OPENML_REGRESSION_SUITE  # noqa: E402
from benchmark_runner import EnsembleChunkBenchmarkRunner  # noqa: E402
from benchmark_sampling_strategies import make_chunking_strategy_configs  # noqa: E402
from run_big_datasets_ensemble import EnsembleReportBuilder  # noqa: E402


DEFAULT_RMT_REGRESSION_TASKS: tuple[str, ...] = (
    "elevators",
    "diamonds",
    "Allstate_Claims_Severity",
    "Yolanda",
)
DEFAULT_CHUNK_FRACTIONS: tuple[float, ...] = (1.0, 0.75, 0.5, 0.3, 0.2, 0.1)
DEFAULT_BUDGET_RATIOS: tuple[float, ...] = (0.01, 0.03, 0.05, 0.10, 0.20)
DEFAULT_ENSEMBLE_METHODS: tuple[str, ...] = ("voting", "routed_weighted")
DEFAULT_STRATEGIES: tuple[str, ...] = (
    "rmt_contraction",
    "random",
    "difficulty",
    "feature_clustering",
)
EFFICIENCY_DELTAS: tuple[float, ...] = (0.01, 0.03, 0.05)


@dataclass(frozen=True)
class RMTRegressionExperimentConfig:
    regression_suite: int | None = OPENML_REGRESSION_SUITE
    regression_tasks: Sequence[str] | None = DEFAULT_RMT_REGRESSION_TASKS
    strategies: Sequence[str] = DEFAULT_STRATEGIES
    models: Sequence[str] = ("lightgbm",)
    ensemble_methods: Sequence[str] = DEFAULT_ENSEMBLE_METHODS
    chunk_fractions: Sequence[float] = DEFAULT_CHUNK_FRACTIONS
    budget_ratios: Sequence[float] = DEFAULT_BUDGET_RATIOS
    n_partitions: int = 5
    max_train_rows: int | None = 300_000
    seed: int = 42
    show_progress: bool = True
    synthetic_smoke: bool = False


@dataclass(frozen=True)
class CappedOpenMLRawDatasetBundle(OpenMLRawDatasetBundle):
    max_train_rows: int | None = None
    seed: int = 42

    def load_split_data(self):
        X_train, y_train, X_test, y_test, feature_columns, categorical_columns, numeric_columns = super().load_split_data()
        if self.max_train_rows is not None and len(y_train) > self.max_train_rows:
            rng = np.random.default_rng(self.seed)
            keep = np.sort(rng.choice(np.arange(len(y_train)), size=int(self.max_train_rows), replace=False))
            X_train = X_train.iloc[keep].reset_index(drop=True)
            y_train = y_train.iloc[keep].reset_index(drop=True)
        return X_train, y_train, X_test, y_test, feature_columns, categorical_columns, numeric_columns


def _cap_openml_dataset(
    dataset: OpenMLRawDatasetBundle,
    max_train_rows: int | None,
    seed: int,
) -> CappedOpenMLRawDatasetBundle:
    return CappedOpenMLRawDatasetBundle(
        name=dataset.name,
        problem_type=dataset.problem_type,
        target_name=dataset.target_name,
        source_path=dataset.source_path,
        X=dataset.X,
        y=dataset.y,
        metadata=dataset.metadata,
        feature_columns=dataset.feature_columns,
        categorical_columns=dataset.categorical_columns,
        numeric_columns=dataset.numeric_columns,
        task_id=dataset.task_id,
        task_name=dataset.task_name,
        suite_id=dataset.suite_id,
        dataset_id=dataset.dataset_id,
        max_train_rows=max_train_rows,
        seed=seed,
    )


def _make_synthetic_smoke_dataset(seed: int) -> RawDatasetBundle:
    X_values, y_values = make_regression(
        n_samples=900,
        n_features=14,
        n_informative=8,
        noise=15.0,
        random_state=seed,
    )
    X = pd.DataFrame(X_values, columns=[f"x_{idx}" for idx in range(X_values.shape[1])])
    y = pd.Series(y_values, name="target")
    numeric_columns = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_columns = [col for col in X.columns if col not in numeric_columns]
    return RawDatasetBundle(
        name="synthetic_rmt_regression_smoke",
        problem_type="regression",
        target_name="target",
        source_path="synthetic://rmt_regression_smoke",
        X=X,
        y=y,
        metadata=RawDatasetMetadata(
            n_objects=int(X.shape[0]),
            n_features=int(X.shape[1]),
            n_train_candidates=int(X.shape[0]),
            n_categorical=len(categorical_columns),
            n_numeric=len(numeric_columns),
        ),
        feature_columns=X.columns.tolist(),
        categorical_columns=categorical_columns,
        numeric_columns=numeric_columns,
    )


def _load_reference_metrics() -> pd.DataFrame:
    path = Path(__file__).resolve().parent / "benchmark_metrics" / "AMLB_regression_suite_040526.csv"
    if not path.exists():
        return pd.DataFrame(columns=["Task", "foundational"])
    return pd.read_csv(path)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _value_series(df: pd.DataFrame, column: str, default: Any = np.nan) -> pd.Series:
    if column in df.columns:
        return df[column]
    return pd.Series([default] * len(df), index=df.index)


def _task_key(dataset_name: str) -> str:
    return str(dataset_name).split("__task_")[0]


def make_rmt_experiment_strategy_configs(
    problem_type: str,
    strategies: Sequence[str],
    ensemble_methods: Sequence[str],
    chunk_fractions: Sequence[float],
    budget_ratios: Sequence[float],
    n_partitions: int,
    seed: int,
) -> dict[str, dict[str, Any]]:
    configs: dict[str, dict[str, Any]] = {
        "full_dataset": {
            "strategy": "full_dataset",
            "force_direct_model": True,
            "ensemble_method": "full_dataset",
            "budget_ratio": 1.0,
            "experiment_chunk_fraction": 1.0,
        }
    }

    for strategy in strategies:
        for ensemble_method in ensemble_methods:
            for chunk_fraction in chunk_fractions:
                for budget_ratio in budget_ratios:
                    base_config = make_chunking_strategy_configs(
                        problem_type=problem_type,
                        strategy_names=(strategy,),
                        n_partitions=n_partitions,
                        seed=seed,
                        ensemble_method=ensemble_method,
                        chunk_fraction=chunk_fraction,
                        budget_ratio=budget_ratio,
                        force_chunking=True,
                    )[strategy]
                    ratio_tag = f"{int(round(budget_ratio * 100)):02d}"
                    chunk_tag = str(chunk_fraction).replace(".", "p")
                    config_name = f"{strategy}__{ensemble_method}__cf_{chunk_tag}__budget_{ratio_tag}"
                    configs[config_name] = base_config
    return configs


def build_rmt_report_tables(
    run_records: Sequence[Mapping[str, Any]],
    output_dir: Path,
    reference_metrics: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    df = pd.json_normalize(list(run_records), sep=".")
    if df.empty:
        empty = pd.DataFrame()
        empty.to_csv(output_dir / "rmt_raw_runs.csv", index=False)
        return {"raw": empty, "efficiency": empty, "minimal_budget": empty}

    raw = pd.DataFrame({
        "dataset": _value_series(df, "dataset"),
        "task_key": _value_series(df, "dataset").map(_task_key),
        "model": _value_series(df, "strategy_params.model"),
        "sampler": _value_series(df, "strategy_params.strategy"),
        "ensemble_method": _value_series(df, "strategy_params.ensemble_method"),
        "chunk_fraction": pd.to_numeric(
            _value_series(df, "strategy_params.chunk_fraction").fillna(
                _value_series(df, "strategy_params.experiment_chunk_fraction")
            ),
            errors="coerce",
        ),
        "budget_ratio": pd.to_numeric(_value_series(df, "strategy_params.budget_ratio"), errors="coerce"),
        "total_train_rows": pd.to_numeric(_value_series(df, "sample_stats.sample_size"), errors="coerce"),
        "rmse": pd.to_numeric(_value_series(df, "model_metrics.rmse"), errors="coerce"),
        "fit_time": pd.to_numeric(_value_series(df, "timings_sec.fit"), errors="coerce"),
        "inference_time": pd.to_numeric(_value_series(df, "timings_sec.inference"), errors="coerce"),
        "leverage_entropy": pd.to_numeric(_value_series(df, "extra.sampler_diagnostics.leverage_entropy"), errors="coerce"),
        "effective_sample_count": pd.to_numeric(_value_series(df, "extra.sampler_diagnostics.effective_sample_count"), errors="coerce"),
        "singular_values": _value_series(df, "extra.sampler_diagnostics.singular_values", default=None),
        "chunk_sizes": _value_series(df, "extra.sampler_diagnostics.chunk_sizes", default=None),
    })

    baseline = (
        raw[raw["sampler"] == "full_dataset"]
        .dropna(subset=["rmse"])
        .groupby(["dataset", "model"], as_index=False)["rmse"]
        .min()
        .rename(columns={"rmse": "rmse_ref"})
    )
    if baseline.empty:
        baseline = (
            raw.dropna(subset=["rmse"])
            .groupby(["dataset", "model"], as_index=False)["rmse"]
            .min()
            .rename(columns={"rmse": "rmse_ref"})
        )
    raw = raw.merge(baseline, on=["dataset", "model"], how="left")

    if reference_metrics is not None and not reference_metrics.empty and "Task" in reference_metrics.columns:
        refs = reference_metrics.copy()
        refs["task_key"] = refs["Task"].astype(str)
        if "foundational" in refs.columns:
            refs["foundational_rmse_ref"] = pd.to_numeric(refs["foundational"], errors="coerce")
            raw = raw.merge(refs[["task_key", "foundational_rmse_ref"]], on="task_key", how="left")

    raw["rmse_drop"] = (raw["rmse"] - raw["rmse_ref"]) / raw["rmse_ref"].replace(0, np.nan)
    raw.to_csv(output_dir / "rmt_raw_runs.csv", index=False)

    efficiency = (
        raw[raw["sampler"] != "full_dataset"]
        .groupby(["dataset", "model", "sampler", "ensemble_method", "chunk_fraction", "budget_ratio"], as_index=False)
        .agg({
            "total_train_rows": "mean",
            "rmse": "mean",
            "rmse_ref": "mean",
            "rmse_drop": "mean",
            "fit_time": "mean",
            "inference_time": "mean",
            "leverage_entropy": "mean",
            "effective_sample_count": "mean",
        })
        .sort_values(["dataset", "sampler", "ensemble_method", "chunk_fraction", "budget_ratio"])
    )
    efficiency.to_csv(output_dir / "sample_efficiency_curve.csv", index=False)

    minimal_rows: list[dict[str, Any]] = []
    for delta in EFFICIENCY_DELTAS:
        eligible = efficiency[efficiency["rmse"] <= efficiency["rmse_ref"] * (1.0 + delta)].copy()
        if eligible.empty:
            continue
        eligible = eligible.sort_values(["budget_ratio", "chunk_fraction"])
        grouped = eligible.groupby(["dataset", "model", "sampler", "ensemble_method"], as_index=False).first()
        grouped["delta"] = delta
        minimal_rows.extend(grouped.to_dict(orient="records"))

    minimal_budget = pd.DataFrame(minimal_rows)
    minimal_budget.to_csv(output_dir / "minimal_effective_budget.csv", index=False)
    return {"raw": raw, "efficiency": efficiency, "minimal_budget": minimal_budget}


class RMTRegressionExperimentOrchestrator:
    def __init__(self, config: RMTRegressionExperimentConfig) -> None:
        self.config = config
        self.report_builder = EnsembleReportBuilder()

    def _load_datasets(self) -> list[RawDatasetBundle]:
        if self.config.synthetic_smoke:
            return [_make_synthetic_smoke_dataset(self.config.seed)]

        datasets = load_suite_raw_datasets(
            classification_suite=None,
            regression_suite=self.config.regression_suite,
            classification_tasks=None,
            regression_tasks=self.config.regression_tasks,
        )
        return [
            _cap_openml_dataset(dataset, self.config.max_train_rows, self.config.seed)
            if isinstance(dataset, OpenMLRawDatasetBundle)
            else dataset
            for dataset in datasets
        ]

    def run(self) -> Path:
        if self.config.synthetic_smoke:
            os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
            os.environ.setdefault("OMP_NUM_THREADS", "1")

        base_dir = Path(__file__).resolve().parent
        run_id = f"run_rmt_contraction_regression_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger = BenchmarkLogger(run_id=run_id, artifacts_root=base_dir / "results")
        incremental_path = logger.paths.metrics / "rmt_regression_runs.jsonl"

        datasets = self._load_datasets()
        if not datasets:
            raise RuntimeError("No regression datasets available for RMT contraction experiment.")

        def _append_incremental(record: Mapping[str, Any]) -> None:
            with incremental_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(_json_ready(dict(record)), ensure_ascii=False) + "\n")

        runner = EnsembleChunkBenchmarkRunner(
            logger=logger,
            cv_folds=2 if self.config.synthetic_smoke else 1,
            seed=self.config.seed,
            show_progress=self.config.show_progress,
            on_record=_append_incremental,
        )

        run_records: list[dict[str, Any]] = []
        strategy_configs = make_rmt_experiment_strategy_configs(
            problem_type="regression",
            strategies=self.config.strategies,
            ensemble_methods=self.config.ensemble_methods,
            chunk_fractions=self.config.chunk_fractions,
            budget_ratios=self.config.budget_ratios,
            n_partitions=self.config.n_partitions,
            seed=self.config.seed,
        )

        for dataset in datasets:
            model_pool = make_model_pool(
                seed=self.config.seed,
                model_names=self.config.models,
                problem_type="regression",
            )
            run_records.extend(runner.run_dataset(dataset, strategy_configs, model_pool))

        self.report_builder.build_tables(run_records, logger.paths.metrics)
        build_rmt_report_tables(run_records, logger.paths.metrics, _load_reference_metrics())
        logger.create_markdown_report(run_records)

        run_meta = {
            "run_id": logger.run_id,
            "output_dir": str(logger.paths.root),
            "regression_suite": self.config.regression_suite,
            "regression_tasks": list(self.config.regression_tasks or []),
            "strategies": list(self.config.strategies),
            "models": list(self.config.models),
            "ensemble_methods": list(self.config.ensemble_methods),
            "chunk_fractions": list(self.config.chunk_fractions),
            "budget_ratios": list(self.config.budget_ratios),
            "max_train_rows": self.config.max_train_rows,
            "synthetic_smoke": self.config.synthetic_smoke,
            "records": len(run_records),
        }
        (logger.paths.root / "run_meta.json").write_text(
            json.dumps(_json_ready(run_meta), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print(f"RMT contraction regression experiment completed. Artifacts: {logger.paths.root}")
        return logger.paths.root


def run_rmt_contraction_regression_experiment(
    regression_tasks: Sequence[str] | None = None,
    models: Sequence[str] = ("lightgbm",),
    max_train_rows: int | None = 300_000,
    show_progress: bool = True,
    synthetic_smoke: bool = False,
) -> Path:
    config = RMTRegressionExperimentConfig(
        regression_tasks=regression_tasks or DEFAULT_RMT_REGRESSION_TASKS,
        models=models,
        max_train_rows=max_train_rows,
        show_progress=show_progress,
        synthetic_smoke=synthetic_smoke,
    )
    return RMTRegressionExperimentOrchestrator(config).run()


if __name__ == "__main__":
    run_rmt_contraction_regression_experiment(
        regression_tasks=("synthetic_rmt_regression_smoke",),
        models=("ridge",),
        max_train_rows=2_000,
        show_progress=False,
        synthetic_smoke=True,
    )
