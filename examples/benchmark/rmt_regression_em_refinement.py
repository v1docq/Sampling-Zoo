from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
BENCHMARK_DIR = Path(__file__).resolve().parent
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

from benchmark_incremental import IncrementalExperimentSaver  # noqa: E402
from benchmark_logging import BenchmarkLogger  # noqa: E402
from rmt_experiment_utils import json_ready, load_reference_metrics  # noqa: E402
from rmt_regression_medium_datasets import (  # noqa: E402
    DEFAULT_BUDGET_RATIOS,
    DEFAULT_ROUTER_MODES,
    DEFAULT_RMT_REGRESSION_TASKS,
    DEFAULT_VIEW_STRATEGIES,
    RMTRegressionExperimentConfig,
    RMTRegressionExperimentOrchestrator,
    make_rmt_experiment_strategy_configs,
)
from sampling_zoo.core.experiment.contracts import StrategyGridContract  # noqa: E402
from sampling_zoo.core.experiment.morphisms import build_standard_rmt_experiment_plan, normalize_strategy_grid  # noqa: E402
from sampling_zoo.core.experiment.stages import ExperimentPlan  # noqa: E402


EM_RETRAINING_CONFIG: dict[str, Any] = {
    "routing_refinement": "em_retraining",
    "em_max_iterations": 2,
    "em_min_improvement": 1e-4,
    "em_assignment_policy": "hard_top1",
    "em_min_partition_size": 32,
    "em_refit_router": True,
    "em_keep_best": True,
}


@dataclass(frozen=True)
class RMTRegressionEMExperimentConfig(RMTRegressionExperimentConfig):
    strategies: Sequence[str] = ("rmt_contraction",)
    ensemble_methods: Sequence[str] = ("routed_weighted",)
    budget_ratios: Sequence[float] = DEFAULT_BUDGET_RATIOS
    view_strategies: Sequence[str] = DEFAULT_VIEW_STRATEGIES
    router_modes: Sequence[str] = DEFAULT_ROUTER_MODES
    regression_tasks: Sequence[str] | None = DEFAULT_RMT_REGRESSION_TASKS
    models: Sequence[str] = ("lightgbm",)
    reference_results_dir: Path | None = None


class RMTRegressionEMExperimentOrchestrator(RMTRegressionExperimentOrchestrator):
    def __init__(self, config: RMTRegressionEMExperimentConfig) -> None:
        super().__init__(config)
        self.config: RMTRegressionEMExperimentConfig = config

    def _create_logger(self) -> BenchmarkLogger:
        base_dir = Path(__file__).resolve().parent
        run_id = f"run_rmt_contraction_regression_em_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        return BenchmarkLogger(run_id=run_id, artifacts_root=base_dir / "results")

    def _build_experiment_plan(self) -> ExperimentPlan:
        payload = asdict(self.config)
        if payload.get("reference_results_dir") is not None:
            payload["reference_results_dir"] = str(payload["reference_results_dir"])
        plan = build_standard_rmt_experiment_plan(payload)
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
        for config in configs.values():
            if config.get("strategy") == "rmt_contraction" and config.get("ensemble_method") == "routed_weighted":
                config.update(EM_RETRAINING_CONFIG)
        return normalize_strategy_grid(configs)

    def _create_incremental_saver(self, logger: BenchmarkLogger) -> IncrementalExperimentSaver:
        reference_metrics = load_reference_metrics()

        def _build_ensemble_tables(records: Sequence[Mapping[str, Any]]) -> None:
            self.report_builder.build_tables(records, logger.paths.metrics)

        def _build_rmt_tables(records: Sequence[Mapping[str, Any]]) -> None:
            self.rmt_report_table_builder.build_report_tables(records, logger.paths.metrics, reference_metrics)
            self._attach_reference_delta(logger)

        def _build_markdown_report(records: Sequence[Mapping[str, Any]]) -> None:
            logger.create_markdown_report(records)

        return IncrementalExperimentSaver(
            records_path=logger.paths.metrics / "rmt_regression_em_runs.jsonl",
            metadata_path=logger.paths.root / "run_meta.json",
            snapshot_hooks=(_build_ensemble_tables, _build_rmt_tables, _build_markdown_report),
            metadata_builder=lambda records, status: self._build_run_meta(logger, records, status),
            json_ready=json_ready,
        )

    def _build_report_artifacts(self, run_records: Sequence[Mapping[str, Any]], logger: BenchmarkLogger) -> None:
        super()._build_report_artifacts(run_records, logger)
        self._attach_reference_delta(logger)

    def _attach_reference_delta(self, logger: BenchmarkLogger) -> None:
        reference_dir = self.config.reference_results_dir
        if reference_dir is None:
            return
        current_path = logger.paths.metrics / "rmt_raw_runs.csv"
        reference_path = self._resolve_reference_raw_runs_path(reference_dir)
        if not current_path.exists() or reference_path is None or not reference_path.exists():
            return

        current = pd.read_csv(current_path)
        reference = pd.read_csv(reference_path)
        if current.empty or reference.empty:
            return
        current = self._ensure_score_columns(current)
        reference = self._ensure_score_columns(reference)
        key_columns = [
            "dataset",
            "model",
            "sampler",
            "ensemble_method",
            "router",
            "view_strategy",
            "budget_ratio",
            "primary_metric",
        ]
        available_keys = [column for column in key_columns if column in current.columns and column in reference.columns]
        if not available_keys:
            return
        ref_scores = reference[available_keys + ["score"]].rename(columns={"score": "non_em_score"})
        merged = current.merge(ref_scores, on=available_keys, how="left")
        merged["delta_vs_non_em_same_config"] = merged["score"] - merged["non_em_score"]
        merged.to_csv(current_path, index=False)

    @staticmethod
    def _resolve_reference_raw_runs_path(reference_dir: Path) -> Path | None:
        reference_dir = Path(reference_dir)
        candidates = [
            reference_dir / "metrics" / "rmt_raw_runs.csv",
            reference_dir / "rmt_raw_runs.csv",
        ]
        return next((path for path in candidates if path.exists()), None)

    @staticmethod
    def _ensure_score_columns(table: pd.DataFrame) -> pd.DataFrame:
        table = table.copy()
        if "primary_metric" not in table.columns:
            table["primary_metric"] = "rmse"
        if "score" not in table.columns:
            table["score"] = pd.to_numeric(table.get("rmse"), errors="coerce")
        return table

    def _build_run_meta(
        self,
        logger: BenchmarkLogger,
        run_records: Sequence[Mapping[str, Any]],
        status: str = "completed",
    ) -> dict[str, Any]:
        meta = super()._build_run_meta(logger, run_records, status)
        meta["routing_refinement"] = dict(EM_RETRAINING_CONFIG)
        meta["reference_results_dir"] = None if self.config.reference_results_dir is None else str(self.config.reference_results_dir)
        return meta

    def _announce_completion(self, logger: BenchmarkLogger) -> Path:
        print(f"RMT regression EM refinement experiment completed. Artifacts: {logger.paths.root}")
        return logger.paths.root


def run_rmt_contraction_regression_em_experiment(
    regression_tasks: Sequence[str] | None = None,
    models: Sequence[str] = ("lightgbm",),
    max_train_rows: int | None = 300_000,
    reference_results_dir: str | Path | None = None,
    show_progress: bool = True,
) -> Path:
    config = RMTRegressionEMExperimentConfig(
        regression_tasks=regression_tasks or DEFAULT_RMT_REGRESSION_TASKS,
        models=models,
        max_train_rows=max_train_rows,
        reference_results_dir=None if reference_results_dir is None else Path(reference_results_dir),
        show_progress=show_progress,
    )
    return RMTRegressionEMExperimentOrchestrator(config).run()


if __name__ == "__main__":
    run_rmt_contraction_regression_em_experiment()
