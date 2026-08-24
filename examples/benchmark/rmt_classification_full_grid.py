"""Staged AMLB-compatible RMT classification benchmark."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

from tqdm.auto import tqdm

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

BENCHMARK_DIR = Path(__file__).resolve().parent
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

from examples.benchmark.benchmark_incremental import (  # noqa: E402
    IncrementalExperimentSaver,
)
from examples.benchmark.benchmark_models import make_model_pool  # noqa: E402
from examples.benchmark.benchmark_repo import (  # noqa: E402
    DEFAULT_BUDGET_RATIOS,
    OPENML_CLASSIFICATION_SUITE,
)
from examples.benchmark.benchmark_runner import (  # noqa: E402
    EnsembleChunkBenchmarkRunner,
)
from examples.benchmark.rmt_classification_full_grid_plan import (  # noqa: E402
    DEFAULT_CLASSIFICATION_SCENARIO_GROUPS,
    RMTClassificationScenarioGroup,
    build_rmt_classification_full_grid,
    model_names_for_classification_groups,
)
from examples.benchmark.rmt_classification_report import (  # noqa: E402
    RMTClassificationReportPlotBuilder,
    RMTClassificationReportTableBuilder,
)
from examples.benchmark.rmt_classification_sampling_gate import (  # noqa: E402
    RMTClassificationGateConfig,
    RMTClassificationGateOrchestrator,
)
from examples.benchmark.rmt_experiment_utils import json_ready  # noqa: E402
from sampling_zoo.core.experiment.contracts import (  # noqa: E402
    ModelStrategyScenarioGridContract,
)
from sampling_zoo.core.experiment.morphisms import (  # noqa: E402
    build_standard_rmt_experiment_plan,
)
from sampling_zoo.core.experiment.resume import ResumePolicy  # noqa: E402
from sampling_zoo.core.experiment.stages import ExperimentPlan  # noqa: E402


DEFAULT_CLASSIFICATION_FULL_GRID_TASKS: tuple[str, ...] = (
    "adult",
    "bank-marketing",
    "covertype",
    "jannis",
)


@dataclass(frozen=True)
class RMTClassificationFullGridConfig(RMTClassificationGateConfig):
    classification_suite: int | None = OPENML_CLASSIFICATION_SUITE
    classification_tasks: Sequence[str] | None = (
        DEFAULT_CLASSIFICATION_FULL_GRID_TASKS
    )
    budget_ratios: Sequence[float] = DEFAULT_BUDGET_RATIOS
    scenario_groups: Sequence[str] = DEFAULT_CLASSIFICATION_SCENARIO_GROUPS
    max_train_rows: int | None = 50_000
    min_samples_per_class: int = 1

    def __post_init__(self) -> None:
        super().__post_init__()
        if int(self.min_samples_per_class) < 1:
            raise ValueError("min_samples_per_class must be positive")
        object.__setattr__(
            self,
            "models",
            model_names_for_classification_groups(self.scenario_groups),
        )


class RMTClassificationFullGridOrchestrator(
    RMTClassificationGateOrchestrator
):
    config: RMTClassificationFullGridConfig

    def __init__(self, config: RMTClassificationFullGridConfig) -> None:
        super().__init__(config)
        self.config = config
        self.scenario_grid: ModelStrategyScenarioGridContract | None = None
        self.classification_report_table_builder = (
            RMTClassificationReportTableBuilder()
        )
        self.classification_report_plot_builder = (
            RMTClassificationReportPlotBuilder()
        )

    def _run_id_prefix(self) -> str:
        return "run_rmt_classification_full_grid"

    def _build_experiment_plan(self) -> ExperimentPlan:
        scenario_grid = self._ensure_scenario_grid()
        effective_config = {
            **asdict(self.config),
            "scenario_grid": scenario_grid.to_dict(),
        }
        plan = build_standard_rmt_experiment_plan(effective_config)
        self.experiment_plan = plan
        return plan

    def _build_strategy_grid(self) -> ModelStrategyScenarioGridContract:
        return self._ensure_scenario_grid()

    def _ensure_scenario_grid(self) -> ModelStrategyScenarioGridContract:
        if self.scenario_grid is None:
            self.scenario_grid = build_rmt_classification_full_grid(
                budget_ratios=self.config.budget_ratios,
                scenario_groups=self.config.scenario_groups,
                n_partitions=self.config.n_partitions,
                seed=self.config.seed,
                min_samples_per_class=self.config.min_samples_per_class,
            )
        return self.scenario_grid

    def _make_model_pool(self) -> dict[str, Any]:
        return make_model_pool(
            seed=self.config.seed,
            model_names=self._ensure_scenario_grid().model_names,
            problem_type="classification",
        )

    def _create_incremental_saver(
        self,
        logger,
    ) -> IncrementalExperimentSaver:
        def _build_ensemble_tables(
            records: Sequence[Mapping[str, Any]],
        ) -> None:
            self.report_builder.build_tables(records, logger.paths.metrics)

        def _build_classification_tables(
            records: Sequence[Mapping[str, Any]],
        ) -> None:
            self.classification_report_table_builder.build_report_tables(
                records,
                logger.paths.metrics,
            )

        def _build_markdown_report(
            records: Sequence[Mapping[str, Any]],
        ) -> None:
            logger.create_markdown_report(records)

        saver = IncrementalExperimentSaver(
            records_path=(
                logger.paths.metrics
                / "rmt_classification_full_grid_runs.jsonl"
            ),
            metadata_path=logger.paths.root / "run_meta.json",
            snapshot_hooks=(
                _build_ensemble_tables,
                _build_classification_tables,
                _build_markdown_report,
            ),
            metadata_builder=lambda records, status: self._build_run_meta(
                logger,
                records,
                status,
            ),
            json_ready=json_ready,
        )

        def _build_final_plots(
            records: Sequence[Mapping[str, Any]],
            status: str,
        ) -> None:
            if status != "completed":
                return
            tables = (
                self.classification_report_table_builder.build_report_tables(
                    records,
                    logger.paths.metrics,
                )
            )
            self.classification_report_plot_builder.build_plots(
                tables["raw"],
                logger.paths.plots,
            )

        return saver.add_lifecycle_hook(_build_final_plots)

    def _run_experiment(
        self,
        datasets,
        strategy_configs: ModelStrategyScenarioGridContract,
        runner: EnsembleChunkBenchmarkRunner,
    ) -> list[dict[str, Any]]:
        run_records: list[dict[str, Any]] = (
            []
            if self.resume_session is None
            else [
                dict(record)
                for record in self.resume_session.plan.retained_records
            ]
        )
        model_pool = self._make_model_pool()
        for dataset in tqdm(
            datasets,
            desc="Run RMT classification full grid",
            disable=not self.config.show_progress,
            leave=False,
        ):
            run_records.extend(
                runner.run_scenario_grid(
                    dataset,
                    strategy_configs,
                    model_pool,
                )
            )
        return run_records

    def _build_run_meta(
        self,
        logger,
        run_records: Sequence[Mapping[str, Any]],
        status: str = "completed",
    ) -> dict[str, Any]:
        metadata = super()._build_run_meta(
            logger,
            run_records,
            status=status,
        )
        metadata.update(
            {
                "scenario_groups": list(self.config.scenario_groups),
                "scenario_grid": self._ensure_scenario_grid().to_dict(),
                "scenario_count": len(
                    self._ensure_scenario_grid().scenarios
                ),
                "primary_metrics": {
                    "binary": "roc_auc",
                    "multiclass": "log_loss",
                },
                "calibration_metrics": [
                    "brier_score",
                    "expected_calibration_error",
                ],
                "min_samples_per_class": int(
                    self.config.min_samples_per_class
                ),
            }
        )
        return metadata

    def _announce_completion(self, logger) -> Path:
        print(
            "RMT classification full-grid experiment completed. "
            f"Artifacts: {logger.paths.root}"
        )
        return logger.paths.root


def run_rmt_classification_full_grid_experiment(
    *,
    classification_tasks: Sequence[str] | None = None,
    scenario_groups: Sequence[str] = DEFAULT_CLASSIFICATION_SCENARIO_GROUPS,
    budget_ratios: Sequence[float] = DEFAULT_BUDGET_RATIOS,
    max_train_rows: int | None = 50_000,
    min_samples_per_class: int = 1,
    show_progress: bool = True,
    resume_from: str | Path | None = None,
    resume_policy: str = ResumePolicy.RETRY_FAILED.value,
    synthetic_smoke: bool = False,
) -> Path:
    config = RMTClassificationFullGridConfig(
        classification_tasks=(
            classification_tasks
            or DEFAULT_CLASSIFICATION_FULL_GRID_TASKS
        ),
        scenario_groups=scenario_groups,
        budget_ratios=budget_ratios,
        max_train_rows=max_train_rows,
        min_samples_per_class=min_samples_per_class,
        show_progress=show_progress,
        resume_from=resume_from,
        resume_policy=resume_policy,
        synthetic_smoke=synthetic_smoke,
    )
    return RMTClassificationFullGridOrchestrator(config).run()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the staged AMLB RMT classification grid."
    )
    parser.add_argument("--task", action="append", dest="tasks")
    parser.add_argument(
        "--scenario-group",
        action="append",
        choices=[group.value for group in RMTClassificationScenarioGroup],
        dest="scenario_groups",
    )
    parser.add_argument("--budget", action="append", type=float, dest="budgets")
    parser.add_argument("--max-train-rows", type=int, default=50_000)
    parser.add_argument("--min-samples-per-class", type=int, default=1)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument(
        "--resume-policy",
        choices=[policy.value for policy in ResumePolicy],
        default=ResumePolicy.RETRY_FAILED.value,
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_rmt_classification_full_grid_experiment(
        classification_tasks=args.tasks,
        scenario_groups=(
            tuple(args.scenario_groups)
            if args.scenario_groups
            else DEFAULT_CLASSIFICATION_SCENARIO_GROUPS
        ),
        budget_ratios=(
            tuple(args.budgets)
            if args.budgets
            else ((0.10,) if args.smoke else DEFAULT_BUDGET_RATIOS)
        ),
        max_train_rows=args.max_train_rows,
        min_samples_per_class=args.min_samples_per_class,
        show_progress=not args.no_progress,
        resume_from=args.resume_from,
        resume_policy=args.resume_policy,
        synthetic_smoke=args.smoke,
    )


if __name__ == "__main__":
    main()
