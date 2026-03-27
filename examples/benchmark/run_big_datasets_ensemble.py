from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from benchmark_models import make_model_pool
from benchmark_datasets import load_custom_raw_datasets
from benchmark_logging import BenchmarkLogger
from benchmark_runner import EnsembleChunkBenchmarkRunner
from benchmark_sampling_strategies import make_chunking_strategy_configs
from benchmark_repo import (
    AMLB_CUSTOM_CLASSIFICATION_DATASETS,
    AMLB_CUSTOM_REGRESSION_DATASETS,
    ENSEMBLE_CV_FOLDS,
    ENSEMBLE_MODELS,
    ENSEMBLE_N_PARTITIONS,
    ENSEMBLE_STRATEGIES,
)


@dataclass(frozen=True)
class BigEnsembleRunConfig:
    classification_datasets: Sequence[str] = AMLB_CUSTOM_CLASSIFICATION_DATASETS
    regression_datasets: Sequence[str] = AMLB_CUSTOM_REGRESSION_DATASETS
    strategies: Sequence[str] = ENSEMBLE_STRATEGIES
    models: Sequence[str] = ENSEMBLE_MODELS
    cv_folds: int = ENSEMBLE_CV_FOLDS
    n_partitions: int = ENSEMBLE_N_PARTITIONS
    seed: int = 42
    show_progress: bool = True


def resolve_dataset_names(
    classification_datasets: Sequence[str],
    regression_datasets: Sequence[str],
) -> list[str]:
    combined = [*classification_datasets, *regression_datasets]
    return list(dict.fromkeys(combined))


def load_datasets(dataset_names: Sequence[str]) -> list:
    return load_custom_raw_datasets(dataset_names)


class EnsembleReportBuilder:
    @staticmethod
    def build_tables(run_records: Sequence[Mapping[str, Any]], output_dir: Path) -> pd.DataFrame:
        df = pd.json_normalize(list(run_records), sep=".")
        if df.empty:
            return df

        df["problem_type"] = df.get("extra.problem_type")
        df["strategy"] = df.get("extra.strategy")
        df["model"] = df.get("extra.model")
        df["cv_fold"] = pd.to_numeric(df.get("extra.cv_fold"), errors="coerce")

        df.to_csv(output_dir / "ensemble_runs.csv", index=False)
        (output_dir / "ensemble_runs.json").write_text(
            df.to_json(orient="records", force_ascii=False, indent=2),
            encoding="utf-8",
        )

        metric_columns = [col for col in df.columns if col.startswith("model_metrics.")]
        agg_candidates: dict[str, str] = {
            "timings_sec.fit": "mean",
            "timings_sec.inference": "mean",
            "extra.n_chunks": "mean",
            "sample_stats.sample_size": "mean",
        }
        agg_map: dict[str, str] = {column: reducer for column, reducer in agg_candidates.items() if column in df.columns}
        for metric_column in metric_columns:
            agg_map[metric_column] = "mean"

        summary = (
            df.groupby(["dataset", "problem_type", "strategy", "model"], as_index=False)
            .agg(agg_map)
            .sort_values(["problem_type", "dataset", "strategy", "model"])
        )
        summary.to_csv(output_dir / "ensemble_summary.csv", index=False)
        return df


class BigDatasetsEnsembleOrchestrator:
    def __init__(self, config: BigEnsembleRunConfig) -> None:
        self.config = config
        self.report_builder = EnsembleReportBuilder()

    def run(self) -> Path:
        base_dir = Path(__file__).resolve().parent
        run_id = f"run_big_datasets_ensemble_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger = BenchmarkLogger(run_id=run_id, artifacts_root=base_dir / "results")
        incremental_path = logger.paths.metrics / "ensemble_runs.jsonl"

        dataset_names = resolve_dataset_names(
            classification_datasets=self.config.classification_datasets,
            regression_datasets=self.config.regression_datasets,
        )
        datasets = load_datasets(dataset_names)
        if not datasets:
            raise RuntimeError("No datasets available for ensemble benchmark run.")

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

        def _append_incremental(record: Mapping[str, Any]) -> None:
            with incremental_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(_json_ready(dict(record)), ensure_ascii=False) + "\n")

        runner = EnsembleChunkBenchmarkRunner(
            logger=logger,
            cv_folds=self.config.cv_folds,
            seed=self.config.seed,
            show_progress=self.config.show_progress,
            on_record=_append_incremental,
        )

        run_records: list[dict[str, Any]] = []
        for dataset in datasets:
            strategy_configs = make_chunking_strategy_configs(
                problem_type=dataset.problem_type,
                strategy_names=self.config.strategies,
                n_partitions=self.config.n_partitions,
                seed=self.config.seed,
            )
            model_pool = make_model_pool(
                seed=self.config.seed,
                model_names=self.config.models,
                problem_type=dataset.problem_type,
            )
            run_records.extend(runner.run_dataset(dataset, strategy_configs, model_pool))

        self.report_builder.build_tables(run_records, logger.paths.metrics)
        logger.create_markdown_report(run_records)

        run_meta = {
            "run_id": logger.run_id,
            "output_dir": str(logger.paths.root),
            "datasets": dataset_names,
            "strategies": list(self.config.strategies),
            "models": list(self.config.models),
            "cv_folds": self.config.cv_folds,
            "n_partitions": self.config.n_partitions,
            "records": len(run_records),
        }
        (logger.paths.root / "run_meta.json").write_text(
            json.dumps(run_meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print(f"Ensemble benchmark completed. Artifacts: {logger.paths.root}")
        return logger.paths.root


def run_big_datasets_ensemble(
    classification_datasets: Sequence[str] = AMLB_CUSTOM_CLASSIFICATION_DATASETS,
    regression_datasets: Sequence[str] = AMLB_CUSTOM_REGRESSION_DATASETS,
    strategies: Sequence[str] = ENSEMBLE_STRATEGIES,
    models: Sequence[str] = ENSEMBLE_MODELS,
    cv_folds: int = ENSEMBLE_CV_FOLDS,
    n_partitions: int = ENSEMBLE_N_PARTITIONS,
    show_progress: bool = True,
) -> None:
    config = BigEnsembleRunConfig(
        classification_datasets=classification_datasets,
        regression_datasets=regression_datasets,
        strategies=strategies,
        models=models,
        cv_folds=cv_folds,
        n_partitions=n_partitions,
        show_progress=show_progress,
    )
    BigDatasetsEnsembleOrchestrator(config).run()


if __name__ == "__main__":
    run_big_datasets_ensemble()
