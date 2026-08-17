from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Sequence

from benchmark_logging import BenchmarkLogger
from rmt_regression_medium_datasets import (
    DEFAULT_BUDGET_RATIOS,
    DEFAULT_ENSEMBLE_METHODS,
    DEFAULT_RMT_REGRESSION_TASKS,
    DEFAULT_ROUTER_MODES,
    RMTRegressionExperimentConfig,
    RMTRegressionExperimentOrchestrator,
)
from sampling_zoo.core.experiment.resume import ResumePolicy


RAW_FEATURE_CLUSTERING_STRATEGIES: tuple[str, ...] = ("raw_feature_clustering",)


class RawFeatureClusteringMediumOrchestrator(RMTRegressionExperimentOrchestrator):
    """Medium-regression side experiment without RMT projections/SVD."""

    def _run_id_prefix(self) -> str:
        return "run_raw_feature_clustering_medium"

    def _create_logger(self) -> BenchmarkLogger:
        base_dir = Path(__file__).resolve().parent
        if self.config.resume_from is not None:
            return super()._create_logger()
        run_id = f"{self._run_id_prefix()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        return BenchmarkLogger(run_id=run_id, artifacts_root=base_dir / "results")


def run_raw_feature_clustering_medium_experiment(
    *,
    regression_tasks: Sequence[str] | None = None,
    models: Sequence[str] = ("lightgbm",),
    budget_ratios: Sequence[float] = DEFAULT_BUDGET_RATIOS,
    max_train_rows: int | None = 300_000,
    show_progress: bool = True,
    resume_from: str | Path | None = None,
    resume_policy: str = ResumePolicy.RETRY_FAILED.value,
    synthetic_smoke: bool = False,
) -> Path:
    """
    Run the projection ablation for the medium regression benchmark.

    This uses the same datasets, budgets, ensemble methods, routers, balanced
    cluster selection, resume policy, and reporting path as the current RMT
    medium benchmark. The only strategy is raw_feature_clustering: it clusters
    standardized encoded features directly instead of an RMT random-contraction
    SVD embedding.
    """
    config = RMTRegressionExperimentConfig(
        regression_tasks=regression_tasks or DEFAULT_RMT_REGRESSION_TASKS,
        strategies=RAW_FEATURE_CLUSTERING_STRATEGIES,
        models=models,
        ensemble_methods=DEFAULT_ENSEMBLE_METHODS,
        budget_ratios=budget_ratios,
        router_modes=DEFAULT_ROUTER_MODES,
        max_train_rows=max_train_rows,
        show_progress=show_progress,
        resume_from=resume_from,
        resume_policy=resume_policy,
        synthetic_smoke=synthetic_smoke,
    )
    return RawFeatureClusteringMediumOrchestrator(config).run()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the raw-feature clustering RMT ablation."
    )
    parser.add_argument("--task", action="append", dest="tasks")
    parser.add_argument("--model", action="append", dest="models")
    parser.add_argument("--budget", action="append", type=float, dest="budgets")
    parser.add_argument("--max-train-rows", type=int, default=300_000)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--resume-policy", default=ResumePolicy.RETRY_FAILED.value)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def _run_cli(args: argparse.Namespace) -> Path:
    return run_raw_feature_clustering_medium_experiment(
        regression_tasks=args.tasks,
        models=tuple(args.models or ("lightgbm",)),
        budget_ratios=tuple(args.budgets or DEFAULT_BUDGET_RATIOS),
        max_train_rows=2_000 if args.smoke else args.max_train_rows,
        show_progress=not args.no_progress,
        resume_from=args.resume_from,
        resume_policy=args.resume_policy,
        synthetic_smoke=args.smoke,
    )


if __name__ == "__main__":
    _run_cli(_parse_args())
