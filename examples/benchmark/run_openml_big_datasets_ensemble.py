from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from benchmark_datasets import load_suite_raw_datasets
from benchmark_logging import BenchmarkLogger
from benchmark_models import make_model_pool
from benchmark_repo import (
    ENSEMBLE_MODELS,
    ENSEMBLE_N_PARTITIONS,
    ENSEMBLE_STRATEGIES,
    OPENML_CLASSIFICATION_SUITE,
    OPENML_REGRESSION_SUITE,
)
from benchmark_runner import EnsembleChunkBenchmarkRunner
from benchmark_sampling_strategies import make_chunking_strategy_configs
from run_big_datasets_ensemble import EnsembleReportBuilder


@dataclass(frozen=True)
class OpenMLBigEnsembleRunConfig:
    classification_suite: int | None = OPENML_CLASSIFICATION_SUITE
    regression_suite: int | None = OPENML_REGRESSION_SUITE
    classification_tasks: Sequence[str] | None = None
    regression_tasks: Sequence[str] | None = None
    strategies: Sequence[str] = ENSEMBLE_STRATEGIES
    models: Sequence[str] = ENSEMBLE_MODELS
    n_partitions: int = ENSEMBLE_N_PARTITIONS
    seed: int = 42
    show_progress: bool = True


class OpenMLBigDatasetsEnsembleOrchestrator:
    def __init__(self, config: OpenMLBigEnsembleRunConfig) -> None:
        self.config = config
        self.report_builder = EnsembleReportBuilder()

    def run(self) -> Path:
        base_dir = Path(__file__).resolve().parent
        run_id = f"run_openml_big_datasets_ensemble_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger = BenchmarkLogger(run_id=run_id, artifacts_root=base_dir / "results")
        incremental_path = logger.paths.metrics / "ensemble_runs.jsonl"
        if not self.config.classification_tasks and not self.config.regression_tasks:
            raise ValueError(
                "No OpenML task names provided. "
                "Pass classification_tasks and/or regression_tasks explicitly."
            )

        datasets = load_suite_raw_datasets(
            classification_suite=self.config.classification_suite,
            regression_suite=self.config.regression_suite,
            classification_tasks=self.config.classification_tasks,
            regression_tasks=self.config.regression_tasks,
        )
        if not datasets:
            raise RuntimeError("No OpenML datasets available for ensemble benchmark run.")

        def _append_incremental(record: dict) -> None:
            with incremental_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        runner = EnsembleChunkBenchmarkRunner(
            logger=logger,
            cv_folds=1,
            seed=self.config.seed,
            show_progress=self.config.show_progress,
            on_record=_append_incremental,
        )

        run_records: list[dict] = []
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
            "classification_suite": self.config.classification_suite,
            "regression_suite": self.config.regression_suite,
            "classification_tasks": list(self.config.classification_tasks or []),
            "regression_tasks": list(self.config.regression_tasks or []),
            "strategies": list(self.config.strategies),
            "models": list(self.config.models),
            "n_partitions": self.config.n_partitions,
            "records": len(run_records),
        }
        (logger.paths.root / "run_meta.json").write_text(
            json.dumps(run_meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print(f"OpenML ensemble benchmark completed. Artifacts: {logger.paths.root}")
        return logger.paths.root


def run_openml_big_datasets_ensemble(
    classification_suite: int | None = OPENML_CLASSIFICATION_SUITE,
    regression_suite: int | None = OPENML_REGRESSION_SUITE,
    classification_tasks: Sequence[str] | None = None,
    regression_tasks: Sequence[str] | None = None,
    strategies: Sequence[str] = ENSEMBLE_STRATEGIES,
    models: Sequence[str] = ENSEMBLE_MODELS,
    n_partitions: int = ENSEMBLE_N_PARTITIONS,
    show_progress: bool = True,
) -> None:
    config = OpenMLBigEnsembleRunConfig(
        classification_suite=classification_suite,
        regression_suite=regression_suite,
        classification_tasks=classification_tasks,
        regression_tasks=regression_tasks,
        strategies=strategies,
        models=models,
        n_partitions=n_partitions,
        show_progress=show_progress,
    )
    OpenMLBigDatasetsEnsembleOrchestrator(config).run()


if __name__ == "__main__":
    run_openml_big_datasets_ensemble(
        classification_suite=OPENML_CLASSIFICATION_SUITE,
        regression_suite=OPENML_REGRESSION_SUITE,
        classification_tasks=["KDDCup99"],
        regression_tasks=None, #["MIP-2016-regression"],
        models=("tabicl",),
        strategies=("random",),
    )
