from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from tqdm.auto import tqdm

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
BENCHMARK_DIR = Path(__file__).resolve().parent
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

from benchmark_dataset_interfaces import cap_openml_dataset, make_synthetic_regression_smoke_dataset  # noqa: E402
from benchmark_datasets import OpenMLRawDatasetBundle, RawDatasetBundle, load_suite_raw_datasets  # noqa: E402
from benchmark_logging import BenchmarkLogger  # noqa: E402
from benchmark_models import make_model_pool  # noqa: E402
from benchmark_repo import OPENML_REGRESSION_SUITE  # noqa: E402
from benchmark_runner import EnsembleChunkBenchmarkRunner  # noqa: E402
from benchmark_sampling_strategies import make_chunking_strategy_configs  # noqa: E402
from rmt_experiment_utils import json_ready, load_reference_metrics  # noqa: E402
from rmt_report_tables import EFFICIENCY_DELTAS, RMTReportTableBuilder, build_rmt_report_tables  # noqa: E402
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
class RMTStrategyGridPoint:
    strategy: str
    ensemble_method: str
    chunk_fraction: float
    budget_ratio: float

    @property
    def config_name(self) -> str:
        ratio_tag = f"{int(round(self.budget_ratio * 100)):02d}"
        chunk_tag = str(self.chunk_fraction).replace(".", "p")
        return f"{self.strategy}__{self.ensemble_method}__cf_{chunk_tag}__budget_{ratio_tag}"


def make_rmt_experiment_strategy_configs(
    problem_type: str,
    strategies: Sequence[str],
    ensemble_methods: Sequence[str],
    chunk_fractions: Sequence[float],
    budget_ratios: Sequence[float],
    n_partitions: int,
    seed: int,
    show_progress: bool = True,
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

    grid = make_rmt_strategy_grid(
        strategies=strategies,
        ensemble_methods=ensemble_methods,
        chunk_fractions=chunk_fractions,
        budget_ratios=budget_ratios,
    )
    for grid_point in tqdm(
        grid,
        desc="Build RMT strategy grid",
        disable=not show_progress,
        leave=False,
    ):
        base_config = make_chunking_strategy_configs(
            problem_type=problem_type,
            strategy_names=(grid_point.strategy,),
            n_partitions=n_partitions,
            seed=seed,
            ensemble_method=grid_point.ensemble_method,
            chunk_fraction=grid_point.chunk_fraction,
            budget_ratio=grid_point.budget_ratio,
            force_chunking=True,
        )[grid_point.strategy]
        configs[grid_point.config_name] = base_config
    return configs


def make_rmt_strategy_grid(
    strategies: Sequence[str],
    ensemble_methods: Sequence[str],
    chunk_fractions: Sequence[float],
    budget_ratios: Sequence[float],
) -> list[RMTStrategyGridPoint]:
    return [
        RMTStrategyGridPoint(
            strategy=strategy,
            ensemble_method=ensemble_method,
            chunk_fraction=float(chunk_fraction),
            budget_ratio=float(budget_ratio),
        )
        for strategy in strategies
        for ensemble_method in ensemble_methods
        for chunk_fraction in chunk_fractions
        for budget_ratio in budget_ratios
    ]


class RMTRegressionExperimentOrchestrator:
    def __init__(self, config: RMTRegressionExperimentConfig) -> None:
        self.config = config
        self.report_builder = EnsembleReportBuilder()
        self.rmt_report_table_builder = RMTReportTableBuilder()

    def _prepare_runtime(self) -> None:
        if self.config.synthetic_smoke:
            os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
            os.environ.setdefault("OMP_NUM_THREADS", "1")

    def _create_logger(self) -> BenchmarkLogger:
        base_dir = Path(__file__).resolve().parent
        run_id = f"run_rmt_contraction_regression_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        return BenchmarkLogger(run_id=run_id, artifacts_root=base_dir / "results")

    def _create_incremental_recorder(
        self,
        logger: BenchmarkLogger,
    ) -> Callable[[Mapping[str, Any]], None]:
        incremental_path = logger.paths.metrics / "rmt_regression_runs.jsonl"

        def _append_incremental(record: Mapping[str, Any]) -> None:
            with incremental_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(json_ready(dict(record)), ensure_ascii=False) + "\n")

        return _append_incremental

    def _create_runner(self, logger: BenchmarkLogger) -> EnsembleChunkBenchmarkRunner:
        return EnsembleChunkBenchmarkRunner(
            logger=logger,
            cv_folds=2 if self.config.synthetic_smoke else 1,
            seed=self.config.seed,
            show_progress=self.config.show_progress,
            on_record=self._create_incremental_recorder(logger),
        )

    def _load_datasets(self) -> list[RawDatasetBundle]:
        if self.config.synthetic_smoke:
            return [make_synthetic_regression_smoke_dataset(self.config.seed)]

        datasets = load_suite_raw_datasets(
            classification_suite=None,
            regression_suite=self.config.regression_suite,
            classification_tasks=None,
            regression_tasks=self.config.regression_tasks,
            show_progress=self.config.show_progress,
        )
        return [
            cap_openml_dataset(dataset, self.config.max_train_rows, self.config.seed)
            if isinstance(dataset, OpenMLRawDatasetBundle)
            else dataset
            for dataset in datasets
        ]

    def _load_available_datasets(self) -> list[RawDatasetBundle]:
        datasets = self._load_datasets()
        if not datasets:
            raise RuntimeError("No regression datasets available for RMT contraction experiment.")
        return datasets

    def _build_strategy_configs(self) -> dict[str, dict[str, Any]]:
        return make_rmt_experiment_strategy_configs(
            problem_type="regression",
            strategies=self.config.strategies,
            ensemble_methods=self.config.ensemble_methods,
            chunk_fractions=self.config.chunk_fractions,
            budget_ratios=self.config.budget_ratios,
            n_partitions=self.config.n_partitions,
            seed=self.config.seed,
            show_progress=self.config.show_progress,
        )

    def _run_experiment(
        self,
        datasets: Sequence[RawDatasetBundle],
        strategy_configs: Mapping[str, Mapping[str, Any]],
        runner: EnsembleChunkBenchmarkRunner,
    ) -> list[dict[str, Any]]:
        run_records: list[dict[str, Any]] = []
        for dataset in tqdm(
            datasets,
            desc="Run RMT datasets",
            disable=not self.config.show_progress,
            leave=False,
        ):
            model_pool = make_model_pool(
                seed=self.config.seed,
                model_names=self.config.models,
                problem_type="regression",
            )
            run_records.extend(runner.run_dataset(dataset, strategy_configs, model_pool))
        return run_records

    def _build_report_artifacts(self, run_records: Sequence[Mapping[str, Any]], logger: BenchmarkLogger) -> None:
        self.report_builder.build_tables(run_records, logger.paths.metrics)
        self.rmt_report_table_builder.build_report_tables(run_records, logger.paths.metrics, load_reference_metrics())
        logger.create_markdown_report(run_records)

    def _build_run_meta(
        self,
        logger: BenchmarkLogger,
        run_records: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        return {
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

    def _write_run_metadata(self, logger: BenchmarkLogger, run_records: Sequence[Mapping[str, Any]]) -> None:
        run_meta = self._build_run_meta(logger, run_records)
        (logger.paths.root / "run_meta.json").write_text(
            json.dumps(json_ready(run_meta), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _announce_completion(self, logger: BenchmarkLogger) -> Path:
        print(f"RMT contraction regression experiment completed. Artifacts: {logger.paths.root}")
        return logger.paths.root

    def run(self) -> Path:
        self._prepare_runtime()
        logger = self._create_logger()
        runner = self._create_runner(logger)
        datasets = self._load_available_datasets()
        strategy_configs = self._build_strategy_configs()
        run_records = self._run_experiment(datasets, strategy_configs, runner)
        self._build_report_artifacts(run_records, logger)
        self._write_run_metadata(logger, run_records)
        return self._announce_completion(logger)


def run_rmt_contraction_regression_experiment(
    regression_tasks: Sequence[str] | None = None,
    models: Sequence[str] = ("lightgbm",),
    max_train_rows: int | None = 300_000,
    show_progress: bool = True
) -> Path:
    config = RMTRegressionExperimentConfig(
        regression_tasks=regression_tasks or DEFAULT_RMT_REGRESSION_TASKS,
        models=models,
        max_train_rows=max_train_rows,
        show_progress=show_progress,
    )
    return RMTRegressionExperimentOrchestrator(config).run()


if __name__ == "__main__":
    run_rmt_contraction_regression_experiment(
        models=("tabpfn",),
        show_progress=True
    )
