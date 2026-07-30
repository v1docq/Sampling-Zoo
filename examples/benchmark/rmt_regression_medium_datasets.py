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

from benchmark_dataset_interfaces import cap_openml_dataset, make_synthetic_regression_smoke_dataset  # noqa: E402
from benchmark_datasets import OpenMLRawDatasetBundle, RawDatasetBundle, load_suite_raw_datasets  # noqa: E402
from benchmark_incremental import IncrementalExperimentSaver  # noqa: E402
from benchmark_logging import BenchmarkLogger  # noqa: E402
from benchmark_models import make_model_pool  # noqa: E402
from benchmark_repo import OPENML_REGRESSION_SUITE  # noqa: E402
from benchmark_runner import EnsembleChunkBenchmarkRunner  # noqa: E402
from benchmark_sampling_strategies import make_chunking_strategy_configs  # noqa: E402
from rmt_experiment_utils import json_ready, load_reference_metrics  # noqa: E402
from rmt_report_tables import EFFICIENCY_DELTAS, RMTReportTableBuilder, build_rmt_report_tables  # noqa: E402
from run_big_datasets_ensemble import EnsembleReportBuilder  # noqa: E402
from sampling_zoo.core.experiment.contracts import StrategyGridContract  # noqa: E402
from sampling_zoo.core.experiment.artifact_manifest import (  # noqa: E402
    ARTIFACT_MANIFEST_SCHEMA_VERSION,
    RunIdentity,
)
from sampling_zoo.core.experiment.artifact_runtime import (  # noqa: E402
    capture_run_identity,
    materialize_experiment_artifact_manifest,
)
from sampling_zoo.core.experiment.morphisms import (  # noqa: E402
    build_standard_rmt_experiment_plan,
    normalize_strategy_grid,
)
from sampling_zoo.core.experiment.stages import ExperimentPlan, ExperimentStageId  # noqa: E402

DEFAULT_RMT_REGRESSION_TASKS: tuple[str, ...] = (
    "diamonds",
    "house_16H",
    "house_sales",
    "elevators",
    "pol",
    "Brazilian_houses",
    "OnlineNewsPopularity",

)
DEFAULT_BUDGET_RATIOS: tuple[float, ...] = (0.1, 0.3, 0.5, 0.75, 0.9)
DEFAULT_ENSEMBLE_METHODS: tuple[str, ...] = ("voting", "routed_weighted")
DEFAULT_VIEW_STRATEGY: tuple[str, ...] = (
    #"subsample",
    "gaussian",
)
DEFAULT_VIEW_STRATEGIES: tuple[str, ...] = DEFAULT_VIEW_STRATEGY
DEFAULT_ROUTER_MODES: tuple[str, ...] = ("spectral", "constrained_gating")
DEFAULT_CONSTRAINED_GATING_CONFIG: dict[str, Any] = {
    "gating_hidden_dim": 64,
    "gating_epochs": 200,
    "gating_lr": 1e-3,
    "gating_kl_weight": 0.10,
    "gating_balance_weight": 0.01,
    "gating_weight_decay": 1e-4,
    "gating_batch_size": 2048,
    "gating_device": "auto",
}
DEFAULT_STRATEGIES: tuple[str, ...] = (
    "rmt_contraction",
    "random",
    "difficulty",
    # "feature_clustering",
)


@dataclass(frozen=True)
class RMTRegressionExperimentConfig:
    regression_suite: int | None = OPENML_REGRESSION_SUITE
    regression_tasks: Sequence[str] | None = DEFAULT_RMT_REGRESSION_TASKS
    strategies: Sequence[str] = DEFAULT_STRATEGIES
    models: Sequence[str] = ("lightgbm",)
    ensemble_methods: Sequence[str] = DEFAULT_ENSEMBLE_METHODS
    budget_ratios: Sequence[float] = DEFAULT_BUDGET_RATIOS
    view_strategies: Sequence[str] = DEFAULT_VIEW_STRATEGIES
    router_modes: Sequence[str] = DEFAULT_ROUTER_MODES
    n_partitions: int = 5
    max_train_rows: int | None = 300_000
    seed: int = 42
    show_progress: bool = True
    synthetic_smoke: bool = False


@dataclass(frozen=True)
class RMTStrategyGridPoint:
    strategy: str
    ensemble_method: str
    budget_ratio: float
    view_strategy: str | None = None
    router: str | None = None

    @property
    def config_name(self) -> str:
        ratio_tag = f"{int(round(self.budget_ratio * 100)):02d}"
        view_tag = f"__view_{self.view_strategy}" if self.view_strategy is not None else ""
        router_tag = f"__router_{self.router}" if self.router is not None else ""
        return f"{self.strategy}{view_tag}__{self.ensemble_method}{router_tag}__budget_{ratio_tag}"


def make_rmt_experiment_strategy_configs(
        problem_type: str,
        strategies: Sequence[str],
        ensemble_methods: Sequence[str],
        budget_ratios: Sequence[float],
        view_strategies: Sequence[str],
        n_partitions: int,
        seed: int,
        show_progress: bool = True,
        router_modes: Sequence[str] = DEFAULT_ROUTER_MODES,
) -> dict[str, dict[str, Any]]:
    configs: dict[str, dict[str, Any]] = {
        "full_dataset": {
            "strategy": "full_dataset",
            "force_direct_model": True,
            "ensemble_method": "full_dataset",
            "budget_ratio": 1.0,
        }
    }

    grid = make_rmt_strategy_grid(
        strategies=strategies,
        ensemble_methods=ensemble_methods,
        budget_ratios=budget_ratios,
        view_strategies=view_strategies,
        router_modes=router_modes,
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
            budget_ratio=grid_point.budget_ratio,
            force_chunking=True,
        )[grid_point.strategy]
        if grid_point.view_strategy is not None:
            base_config["view_strategy"] = grid_point.view_strategy
        if grid_point.router is not None:
            base_config["router"] = grid_point.router
        if grid_point.router == "constrained_gating":
            base_config.update(DEFAULT_CONSTRAINED_GATING_CONFIG)
        configs[grid_point.config_name] = base_config
    # del configs['full_dataset']
    return configs


def make_rmt_strategy_grid(
        strategies: Sequence[str],
        ensemble_methods: Sequence[str],
        budget_ratios: Sequence[float],
        view_strategies: Sequence[str] = DEFAULT_VIEW_STRATEGIES,
        router_modes: Sequence[str] = DEFAULT_ROUTER_MODES,
) -> list[RMTStrategyGridPoint]:
    view_strategies = _normalize_view_strategies(view_strategies)
    router_modes = _normalize_router_modes(router_modes)
    grid: list[RMTStrategyGridPoint] = []
    for strategy in strategies:
        strategy_view_strategies: Sequence[str | None]
        if strategy == "rmt_contraction":
            strategy_view_strategies = tuple(view_strategies)
        else:
            strategy_view_strategies = (None,)
        for view_strategy in strategy_view_strategies:
            for ensemble_method in ensemble_methods:
                strategy_router_modes: Sequence[str | None]
                if strategy == "rmt_contraction" and ensemble_method == "routed_weighted":
                    strategy_router_modes = tuple(router_modes)
                else:
                    strategy_router_modes = (None,)
                for budget_ratio in budget_ratios:
                    for router in strategy_router_modes:
                        grid.append(
                            RMTStrategyGridPoint(
                                strategy=strategy,
                                ensemble_method=ensemble_method,
                                budget_ratio=float(budget_ratio),
                                view_strategy=view_strategy,
                                router=router,
                            )
                        )
    return grid


def _normalize_view_strategies(view_strategies: Sequence[str] | str) -> tuple[str, ...]:
    if isinstance(view_strategies, str):
        return (view_strategies,)
    return tuple(view_strategies)


def _normalize_router_modes(router_modes: Sequence[str] | str) -> tuple[str, ...]:
    if isinstance(router_modes, str):
        return (router_modes,)
    return tuple(router_modes)


class RMTRegressionExperimentOrchestrator:
    def __init__(self, config: RMTRegressionExperimentConfig) -> None:
        self.config = config
        self.report_builder = EnsembleReportBuilder()
        self.rmt_report_table_builder = RMTReportTableBuilder()
        self.incremental_saver: IncrementalExperimentSaver | None = None
        self.experiment_plan: ExperimentPlan | None = None
        self.run_identity: RunIdentity | None = None

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
        self.incremental_saver = self._create_incremental_saver(logger)
        self._configure_artifact_tracking(logger, self.incremental_saver)
        self.incremental_saver.start()
        return self.incremental_saver.record

    def _configure_artifact_tracking(
            self,
            logger: BenchmarkLogger,
            saver: IncrementalExperimentSaver,
    ) -> None:
        run_identity = self._ensure_run_identity(logger)
        metadata_builder = saver.metadata_builder

        def _build_metadata(
                records: Sequence[Mapping[str, Any]],
                status: str,
        ) -> Mapping[str, Any]:
            payload = (
                {"status": status, "records": len(records)}
                if metadata_builder is None
                else metadata_builder(records, status)
            )
            return self._enrich_run_metadata(payload, run_identity)

        def _materialize_manifest(
                records: Sequence[Mapping[str, Any]],
                status: str,
        ) -> None:
            materialize_experiment_artifact_manifest(
                run_dir=logger.paths.root,
                run_identity=run_identity,
                status=status,
                records=records,
            )

        saver.metadata_builder = _build_metadata
        saver.add_lifecycle_hook(_materialize_manifest)

    def _ensure_run_identity(self, logger: BenchmarkLogger) -> RunIdentity:
        current_identity = getattr(self, "run_identity", None)
        if current_identity is not None:
            return current_identity
        effective_config = (
            self.experiment_plan.effective_config
            if self.experiment_plan is not None
            else asdict(self.config)
        )
        self.run_identity = capture_run_identity(
            run_id=logger.run_id,
            effective_config=effective_config,
            repo_root=ROOT_DIR,
            ignored_git_paths=(logger.paths.root,),
        )
        return self.run_identity

    @staticmethod
    def _enrich_run_metadata(
            payload: Mapping[str, Any],
            run_identity: RunIdentity,
    ) -> dict[str, Any]:
        return {
            **dict(payload),
            "artifact_manifest_schema_version": (
                ARTIFACT_MANIFEST_SCHEMA_VERSION
            ),
            "run_identity": run_identity.to_dict(),
        }

    def _create_incremental_saver(self, logger: BenchmarkLogger) -> IncrementalExperimentSaver:
        reference_metrics = load_reference_metrics()

        def _build_ensemble_tables(records: Sequence[Mapping[str, Any]]) -> None:
            self.report_builder.build_tables(records, logger.paths.metrics)

        def _build_rmt_tables(records: Sequence[Mapping[str, Any]]) -> None:
            self.rmt_report_table_builder.build_report_tables(records, logger.paths.metrics, reference_metrics)

        def _build_markdown_report(records: Sequence[Mapping[str, Any]]) -> None:
            logger.create_markdown_report(records)

        return IncrementalExperimentSaver(
            records_path=logger.paths.metrics / "rmt_regression_runs.jsonl",
            metadata_path=logger.paths.root / "run_meta.json",
            snapshot_hooks=(
                _build_ensemble_tables,
                _build_rmt_tables,
                _build_markdown_report,
            ),
            metadata_builder=lambda records, status: self._build_run_meta(logger, records, status),
            json_ready=json_ready,
        )

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

        with tqdm(
            total=2,
            desc="Load OpenML datasets",
            disable=not self.config.show_progress,
            leave=False,
            unit="stage",
        ) as load_progress:
            load_progress.set_postfix_str("resolve suite tasks")
            datasets = load_suite_raw_datasets(
                classification_suite=None,
                regression_suite=self.config.regression_suite,
                classification_tasks=None,
                regression_tasks=self.config.regression_tasks,
                show_progress=self.config.show_progress,
            )
            load_progress.update(1)

            load_progress.set_postfix_str("apply row caps")
            prepared_datasets = []
            for dataset in tqdm(
                datasets,
                desc="Prepare OpenML datasets",
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
            raise RuntimeError("No regression datasets available for RMT contraction experiment.")
        return datasets

    def _build_experiment_plan(self) -> ExperimentPlan:
        plan = build_standard_rmt_experiment_plan(asdict(self.config))
        self.experiment_plan = plan
        return plan

    def _build_strategy_grid(self) -> StrategyGridContract:
        configs = make_rmt_experiment_strategy_configs(
            problem_type="regression",
            strategies=self.config.strategies,
            ensemble_methods=self.config.ensemble_methods,
            budget_ratios=self.config.budget_ratios,
            view_strategies=self.config.view_strategies,
            n_partitions=self.config.n_partitions,
            seed=self.config.seed,
            show_progress=self.config.show_progress,
            router_modes=self.config.router_modes,
        )
        return normalize_strategy_grid(configs)

    def _run_experiment(
            self,
            datasets: Sequence[RawDatasetBundle],
            strategy_configs: StrategyGridContract | Mapping[str, Mapping[str, Any]],
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
        if self.incremental_saver is not None:
            self.incremental_saver.persist_snapshot(run_records, status="running")
            return
        self.report_builder.build_tables(run_records, logger.paths.metrics)
        self.rmt_report_table_builder.build_report_tables(run_records, logger.paths.metrics, load_reference_metrics())
        logger.create_markdown_report(run_records)

    def _build_run_meta(
            self,
            logger: BenchmarkLogger,
            run_records: Sequence[Mapping[str, Any]],
            status: str = "completed",
    ) -> dict[str, Any]:
        return {
            "run_id": logger.run_id,
            "output_dir": str(logger.paths.root),
            "regression_suite": self.config.regression_suite,
            "regression_tasks": list(self.config.regression_tasks or []),
            "strategies": list(self.config.strategies),
            "models": list(self.config.models),
            "ensemble_methods": list(self.config.ensemble_methods),
            "budget_ratios": list(self.config.budget_ratios),
            "view_strategies": list(self.config.view_strategies),
            "max_train_rows": self.config.max_train_rows,
            "synthetic_smoke": self.config.synthetic_smoke,
            "status": status,
            "records": len(run_records),
            "experiment_plan": None if self.experiment_plan is None else self.experiment_plan.to_dict(),
        }

    def _write_run_metadata(self, logger: BenchmarkLogger, run_records: Sequence[Mapping[str, Any]]) -> None:
        if self.incremental_saver is not None:
            self.incremental_saver.finalize(run_records)
            return
        run_meta = self._enrich_run_metadata(
            self._build_run_meta(logger, run_records),
            self._ensure_run_identity(logger),
        )
        (logger.paths.root / "run_meta.json").write_text(
            json.dumps(json_ready(run_meta), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _announce_completion(self, logger: BenchmarkLogger) -> Path:
        print(f"RMT contraction regression experiment completed. Artifacts: {logger.paths.root}")
        return logger.paths.root

    def _execute_experiment_plan(self, plan: ExperimentPlan) -> Path:
        context: dict[str, Any] = {}
        for stage_id in plan.stage_ids():
            if stage_id == ExperimentStageId.PREPARE_RUNTIME:
                self._prepare_runtime()
            elif stage_id == ExperimentStageId.CREATE_LOGGER:
                context["logger"] = self._create_logger()
            elif stage_id == ExperimentStageId.CREATE_RUNNER:
                context["runner"] = self._create_runner(context["logger"])
            elif stage_id == ExperimentStageId.LOAD_DATASETS:
                context["datasets"] = self._load_available_datasets()
            elif stage_id == ExperimentStageId.BUILD_STRATEGY_GRID:
                context["strategy_grid"] = self._build_strategy_grid()
            elif stage_id == ExperimentStageId.RUN_DATASETS:
                context["run_records"] = self._run_experiment(
                    context["datasets"],
                    context["strategy_grid"],
                    context["runner"],
                )
            elif stage_id == ExperimentStageId.BUILD_REPORTS:
                self._build_report_artifacts(context["run_records"], context["logger"])
            elif stage_id == ExperimentStageId.WRITE_METADATA:
                self._write_run_metadata(context["logger"], context["run_records"])
            elif stage_id == ExperimentStageId.FINALIZE:
                context["result_path"] = self._announce_completion(context["logger"])
            else:
                raise RuntimeError(f"Unsupported experiment stage: {stage_id}")
        return context["result_path"]

    def run(self) -> Path:
        plan = self._build_experiment_plan()
        try:
            return self._execute_experiment_plan(plan)
        except Exception as ex:
            if self.incremental_saver is not None:
                self.incremental_saver.mark_failed(ex)
            raise


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
