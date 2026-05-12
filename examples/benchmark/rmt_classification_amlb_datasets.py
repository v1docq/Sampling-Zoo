from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass
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

from benchmark_dataset_interfaces import (  # noqa: E402
    cap_openml_dataset,
    make_synthetic_classification_smoke_dataset,
)
from benchmark_datasets import AMLB_OPENML_DATASETS, OpenMLRawDatasetBundle, RawDatasetBundle, load_suite_raw_datasets  # noqa: E402
from benchmark_incremental import IncrementalExperimentSaver  # noqa: E402
from benchmark_logging import BenchmarkLogger  # noqa: E402
from benchmark_models import make_model_pool  # noqa: E402
from benchmark_repo import AMLB_CATEGORY_PROFILES, OPENML_CLASSIFICATION_SUITE  # noqa: E402
from benchmark_runner import EnsembleChunkBenchmarkRunner  # noqa: E402
from rmt_experiment_utils import json_ready  # noqa: E402
from rmt_regression_medium_datasets import (  # noqa: E402
    DEFAULT_BUDGET_RATIOS,
    DEFAULT_CONSTRAINED_GATING_CONFIG,
    DEFAULT_ROUTER_MODES,
    DEFAULT_VIEW_STRATEGIES,
    RMTRegressionExperimentOrchestrator,
    make_rmt_experiment_strategy_configs,
)
from rmt_report_tables import RMTReportTableBuilder  # noqa: E402
from run_big_datasets_ensemble import EnsembleReportBuilder  # noqa: E402
from sampling_zoo.core.experiment.contracts import StrategyGridContract  # noqa: E402
from sampling_zoo.core.experiment.morphisms import build_standard_rmt_experiment_plan, normalize_strategy_grid  # noqa: E402
from sampling_zoo.core.experiment.stages import ExperimentPlan  # noqa: E402


DEFAULT_CLASSIFICATION_TASKS: tuple[str, ...] = AMLB_CATEGORY_PROFILES["amlb_top20_mix"]
DEFAULT_CLASSIFICATION_STRATEGIES: tuple[str, ...] = ("rmt_contraction", "random", "difficulty")
DEFAULT_CLASSIFICATION_ENSEMBLE_METHODS: tuple[str, ...] = ("voting", "routed_weighted")


@dataclass(frozen=True)
class RMTClassificationExperimentConfig:
    classification_suite: int | None = OPENML_CLASSIFICATION_SUITE
    classification_tasks: Sequence[str] | None = DEFAULT_CLASSIFICATION_TASKS
    strategies: Sequence[str] = DEFAULT_CLASSIFICATION_STRATEGIES
    models: Sequence[str] = ("tabpfn",)
    ensemble_methods: Sequence[str] = DEFAULT_CLASSIFICATION_ENSEMBLE_METHODS
    budget_ratios: Sequence[float] = DEFAULT_BUDGET_RATIOS
    view_strategies: Sequence[str] = DEFAULT_VIEW_STRATEGIES
    router_modes: Sequence[str] = DEFAULT_ROUTER_MODES
    n_partitions: int = 5
    max_train_rows: int | None = 100_000
    seed: int = 42
    show_progress: bool = True
    synthetic_smoke: bool = False


def _resolve_classification_tasks(tasks: Sequence[str] | None) -> tuple[str, ...]:
    resolved = []
    for task in tasks or DEFAULT_CLASSIFICATION_TASKS:
        resolved.append(AMLB_OPENML_DATASETS.get(task, task))
    return tuple(dict.fromkeys(resolved))


class RMTClassificationExperimentOrchestrator(RMTRegressionExperimentOrchestrator):
    def __init__(self, config: RMTClassificationExperimentConfig) -> None:
        self.config = config
        self.report_builder = EnsembleReportBuilder()
        self.rmt_report_table_builder = RMTReportTableBuilder()
        self.incremental_saver: IncrementalExperimentSaver | None = None
        self.experiment_plan: ExperimentPlan | None = None

    def _prepare_runtime(self) -> None:
        if self.config.synthetic_smoke:
            os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
            os.environ.setdefault("OMP_NUM_THREADS", "1")

    def _create_logger(self) -> BenchmarkLogger:
        base_dir = Path(__file__).resolve().parent
        run_id = f"run_rmt_contraction_classification_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        return BenchmarkLogger(run_id=run_id, artifacts_root=base_dir / "results")

    def _create_incremental_saver(self, logger: BenchmarkLogger) -> IncrementalExperimentSaver:
        def _build_ensemble_tables(records: Sequence[Mapping[str, Any]]) -> None:
            self.report_builder.build_tables(records, logger.paths.metrics)

        def _build_rmt_tables(records: Sequence[Mapping[str, Any]]) -> None:
            self.rmt_report_table_builder.build_report_tables(records, logger.paths.metrics)

        def _build_markdown_report(records: Sequence[Mapping[str, Any]]) -> None:
            logger.create_markdown_report(records)

        return IncrementalExperimentSaver(
            records_path=logger.paths.metrics / "rmt_classification_runs.jsonl",
            metadata_path=logger.paths.root / "run_meta.json",
            snapshot_hooks=(_build_ensemble_tables, _build_rmt_tables, _build_markdown_report),
            metadata_builder=lambda records, status: self._build_run_meta(logger, records, status),
            json_ready=json_ready,
        )

    def _load_datasets(self) -> list[RawDatasetBundle]:
        if self.config.synthetic_smoke:
            return [make_synthetic_classification_smoke_dataset(self.config.seed, n_classes=3)]

        resolved_tasks = _resolve_classification_tasks(self.config.classification_tasks)
        with tqdm(
            total=2,
            desc="Load OpenML classification datasets",
            disable=not self.config.show_progress,
            leave=False,
            unit="stage",
        ) as load_progress:
            load_progress.set_postfix_str("resolve suite tasks")
            datasets = load_suite_raw_datasets(
                classification_suite=self.config.classification_suite,
                regression_suite=None,
                classification_tasks=resolved_tasks,
                regression_tasks=None,
                show_progress=self.config.show_progress,
            )
            load_progress.update(1)

            load_progress.set_postfix_str("apply stratified row caps")
            prepared_datasets = []
            for dataset in tqdm(
                datasets,
                desc="Prepare classification datasets",
                disable=not self.config.show_progress,
                leave=False,
            ):
                prepared_datasets.append(
                    cap_openml_dataset(dataset, self.config.max_train_rows, self.config.seed)
                    if isinstance(dataset, OpenMLRawDatasetBundle)
                    else dataset
                )
            load_progress.update(1)

        return prepared_datasets

    def _load_available_datasets(self) -> list[RawDatasetBundle]:
        datasets = self._load_datasets()
        if not datasets:
            raise RuntimeError("No classification datasets available for RMT contraction experiment.")
        return datasets

    def _build_experiment_plan(self) -> ExperimentPlan:
        plan = build_standard_rmt_experiment_plan(asdict(self.config))
        self.experiment_plan = plan
        return plan

    def _build_strategy_grid(self) -> StrategyGridContract:
        configs = make_rmt_experiment_strategy_configs(
            problem_type="classification",
            strategies=self.config.strategies,
            ensemble_methods=self.config.ensemble_methods,
            budget_ratios=self.config.budget_ratios,
            view_strategies=self.config.view_strategies,
            n_partitions=self.config.n_partitions,
            seed=self.config.seed,
            show_progress=self.config.show_progress,
            router_modes=self.config.router_modes,
        )
        for config in configs.values():
            if config.get("router") == "constrained_gating":
                config.update(DEFAULT_CONSTRAINED_GATING_CONFIG)
        return normalize_strategy_grid(configs)

    def _run_experiment(
        self,
        datasets: Sequence[RawDatasetBundle],
        strategy_configs: StrategyGridContract | Mapping[str, Mapping[str, Any]],
        runner: EnsembleChunkBenchmarkRunner,
    ) -> list[dict[str, Any]]:
        run_records: list[dict[str, Any]] = []
        model_pool = make_model_pool(
            seed=self.config.seed,
            model_names=self.config.models,
            problem_type="classification",
        )
        for dataset in tqdm(
            datasets,
            desc="Run RMT classification datasets",
            disable=not self.config.show_progress,
            leave=False,
        ):
            run_records.extend(runner.run_dataset(dataset, strategy_configs, model_pool))
        return run_records

    def _build_run_meta(
        self,
        logger: BenchmarkLogger,
        run_records: Sequence[Mapping[str, Any]],
        status: str = "completed",
    ) -> dict[str, Any]:
        return {
            "run_id": logger.run_id,
            "output_dir": str(logger.paths.root),
            "classification_suite": self.config.classification_suite,
            "classification_tasks": list(self.config.classification_tasks or []),
            "resolved_classification_tasks": list(_resolve_classification_tasks(self.config.classification_tasks)),
            "strategies": list(self.config.strategies),
            "models": list(self.config.models),
            "ensemble_methods": list(self.config.ensemble_methods),
            "budget_ratios": list(self.config.budget_ratios),
            "view_strategies": list(self.config.view_strategies),
            "router_modes": list(self.config.router_modes),
            "primary_metrics": {"binary": "roc_auc", "multiclass": "log_loss"},
            "max_train_rows": self.config.max_train_rows,
            "synthetic_smoke": self.config.synthetic_smoke,
            "status": status,
            "records": len(run_records),
            "experiment_plan": None if self.experiment_plan is None else self.experiment_plan.to_dict(),
        }

    def _announce_completion(self, logger: BenchmarkLogger) -> Path:
        print(f"RMT contraction classification experiment completed. Artifacts: {logger.paths.root}")
        return logger.paths.root


def run_rmt_contraction_classification_experiment(
    classification_tasks: Sequence[str] | None = None,
    models: Sequence[str] = ("tabpfn",),
    max_train_rows: int | None = 100_000,
    show_progress: bool = True,
) -> Path:
    config = RMTClassificationExperimentConfig(
        classification_tasks=classification_tasks or DEFAULT_CLASSIFICATION_TASKS,
        models=models,
        max_train_rows=max_train_rows,
        show_progress=show_progress,
    )
    return RMTClassificationExperimentOrchestrator(config).run()


if __name__ == "__main__":
    run_rmt_contraction_classification_experiment()
