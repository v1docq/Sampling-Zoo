"""Final non-Cartesian RMT regression benchmark for the server run."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

from tqdm.auto import tqdm

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from examples.benchmark.benchmark_model_profiles import TabPFNFinetuneConfig
from examples.benchmark.benchmark_models import make_model_pool
from examples.benchmark.benchmark_repo import (
    DEFAULT_BUDGET_RATIOS as MAIN_BUDGET_RATIOS,
)
from examples.benchmark.benchmark_repo import OPENML_REGRESSION_SUITE
from examples.benchmark.benchmark_runner import EnsembleChunkBenchmarkRunner
from examples.benchmark.rmt_full_grid_plan import (
    DEFAULT_FULL_GRID_SCENARIO_GROUPS,
    RMTFullGridScenarioGroup,
    build_rmt_regression_full_grid,
    model_names_for_scenario_groups,
)
from examples.benchmark.rmt_regression_medium_datasets import (
    DEFAULT_RMT_REGRESSION_TASKS,
    RMTRegressionExperimentConfig,
    RMTRegressionExperimentOrchestrator,
    load_reference_metrics,
)
from examples.benchmark.rmt_report_plots import RMTReportPlotBuilder
from sampling_zoo.core.experiment.contracts import (
    ModelStrategyScenarioGridContract,
)
from sampling_zoo.core.experiment.morphisms import (
    build_standard_rmt_experiment_plan,
)
from sampling_zoo.core.experiment.resume import ResumePolicy
from sampling_zoo.core.experiment.stages import ExperimentPlan


@dataclass(frozen=True)
class RMTFullGridExperimentConfig(RMTRegressionExperimentConfig):
    regression_suite: int | None = OPENML_REGRESSION_SUITE
    regression_tasks: Sequence[str] | None = DEFAULT_RMT_REGRESSION_TASKS
    strategies: Sequence[str] = (
        "rmt_contraction",
        "random",
        "difficulty",
    )
    models: Sequence[str] = (
        "lightgbm",
        "tabpfn_in_context",
        "tabpfn_finetuned",
    )
    ensemble_methods: Sequence[str] = (
        "full_dataset",
        "voting",
        "routed_weighted",
    )
    budget_ratios: Sequence[float] = MAIN_BUDGET_RATIOS
    view_strategies: Sequence[str] = ("gaussian",)
    router_modes: Sequence[str] = ("spectral",)
    max_train_rows: int | None = 100_000
    scenario_groups: Sequence[str] = DEFAULT_FULL_GRID_SCENARIO_GROUPS
    tabpfn_finetune_config: TabPFNFinetuneConfig | Mapping[str, Any] = field(
        default_factory=lambda: TabPFNFinetuneConfig(time_limit=900)
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        resolved_models = model_names_for_scenario_groups(
            self.scenario_groups
        )
        object.__setattr__(self, "models", resolved_models)


class RMTFullGridExperimentOrchestrator(
    RMTRegressionExperimentOrchestrator
):
    def __init__(self, config: RMTFullGridExperimentConfig) -> None:
        super().__init__(config)
        self.config = config
        self.scenario_grid: ModelStrategyScenarioGridContract | None = None
        self.report_plot_builder = RMTReportPlotBuilder()

    def _run_id_prefix(self) -> str:
        return "run_rmt_regression_full_grid"

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
            self.scenario_grid = build_rmt_regression_full_grid(
                budget_ratios=self.config.budget_ratios,
                scenario_groups=self.config.scenario_groups,
                n_partitions=self.config.n_partitions,
                seed=self.config.seed,
            )
        return self.scenario_grid

    def _make_model_pool(self) -> dict[str, Any]:
        return make_model_pool(
            seed=self.config.seed,
            model_names=self._ensure_scenario_grid().model_names,
            problem_type="regression",
            tabpfn_finetune_config=self.config.tabpfn_finetune_config,
        )

    def _create_incremental_saver(self, logger):
        saver = super()._create_incremental_saver(logger)

        def _build_final_plots(records, status: str) -> None:
            if status != "completed":
                return
            tables = self.rmt_report_table_builder.build_report_tables(
                records,
                logger.paths.metrics,
                load_reference_metrics(),
            )
            self.report_plot_builder.build_plots(
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
            desc="Run final RMT grid datasets",
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
        run_records,
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
            }
        )
        return metadata

    def _announce_completion(self, logger) -> Path:
        print(
            "RMT regression full-grid experiment completed. "
            f"Artifacts: {logger.paths.root}"
        )
        return logger.paths.root


def run_rmt_regression_full_grid_experiment(
    *,
    regression_tasks: Sequence[str] | None = None,
    scenario_groups: Sequence[str] = DEFAULT_FULL_GRID_SCENARIO_GROUPS,
    budget_ratios: Sequence[float] = MAIN_BUDGET_RATIOS,
    max_train_rows: int | None = 100_000,
    show_progress: bool = True,
    resume_from: str | Path | None = None,
    resume_policy: str = ResumePolicy.RETRY_FAILED.value,
    synthetic_smoke: bool = False,
    tabpfn_finetune_config: (
        TabPFNFinetuneConfig | Mapping[str, Any] | None
    ) = None,
) -> Path:
    config = RMTFullGridExperimentConfig(
        regression_tasks=(
            regression_tasks or DEFAULT_RMT_REGRESSION_TASKS
        ),
        scenario_groups=scenario_groups,
        budget_ratios=budget_ratios,
        max_train_rows=max_train_rows,
        show_progress=show_progress,
        resume_from=resume_from,
        resume_policy=resume_policy,
        synthetic_smoke=synthetic_smoke,
        tabpfn_finetune_config=(
            TabPFNFinetuneConfig(time_limit=900)
            if tabpfn_finetune_config is None
            else tabpfn_finetune_config
        ),
    )
    return RMTFullGridExperimentOrchestrator(config).run()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the final non-Cartesian RMT regression grid."
    )
    parser.add_argument("--task", action="append", dest="tasks")
    parser.add_argument(
        "--scenario-group",
        action="append",
        choices=[group.value for group in RMTFullGridScenarioGroup],
        dest="scenario_groups",
    )
    parser.add_argument(
        "--budget",
        action="append",
        type=float,
        dest="budgets",
    )
    parser.add_argument("--max-train-rows", type=int, default=100_000)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--finetune-epochs", type=int, default=30)
    parser.add_argument("--finetune-time-limit", type=int, default=900)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def _run_cli(args: argparse.Namespace) -> Path:
    if args.smoke:
        return run_rmt_regression_full_grid_experiment(
            scenario_groups=("lightgbm_controls",),
            budget_ratios=(0.05,),
            max_train_rows=2_000,
            show_progress=not args.no_progress,
            synthetic_smoke=True,
        )
    return run_rmt_regression_full_grid_experiment(
        regression_tasks=args.tasks,
        scenario_groups=(
            args.scenario_groups or DEFAULT_FULL_GRID_SCENARIO_GROUPS
        ),
        budget_ratios=args.budgets or MAIN_BUDGET_RATIOS,
        max_train_rows=args.max_train_rows,
        show_progress=not args.no_progress,
        resume_from=args.resume_from,
        tabpfn_finetune_config=TabPFNFinetuneConfig(
            epochs=args.finetune_epochs,
            time_limit=args.finetune_time_limit,
        ),
    )


if __name__ == "__main__":
    _run_cli(_parse_args())
