from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import math
from numbers import Integral
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

from benchmark_logging import BenchmarkLogger  # noqa: E402
from benchmark_models import make_model_pool  # noqa: E402
from benchmark_repo import OPENML_REGRESSION_SUITE  # noqa: E402
from benchmark_sampling_strategies import make_chunking_strategy_configs  # noqa: E402
from rmt_experiment_utils import budget_ratio_tag  # noqa: E402
from rmt_regression_medium_datasets import (  # noqa: E402
    DEFAULT_RMT_REGRESSION_TASKS,
    RMTRegressionExperimentConfig,
    RMTRegressionExperimentOrchestrator,
)
from sampling_zoo.core.experiment.contracts import StrategyGridContract  # noqa: E402
from sampling_zoo.core.experiment.morphisms import (  # noqa: E402
    build_standard_rmt_experiment_plan,
    normalize_strategy_grid,
)
from sampling_zoo.core.experiment.resume import ResumePolicy  # noqa: E402
from sampling_zoo.core.experiment.stages import ExperimentPlan  # noqa: E402


DEFAULT_PARTITION_SELECTION_METRICS: tuple[str, ...] = (
    "balanced_silhouette",
    "validation_proxy",
    "budget_aware_validation_proxy",
    "downstream_proxy",
)
DEFAULT_MECHANISM_SMOKE_TASKS: tuple[str, ...] = (
    "Brazilian_houses",
    "diamonds",
    "pol",
)
DEFAULT_MECHANISM_SMOKE_BUDGET_RATIOS: tuple[float, ...] = (0.01, 0.05, 0.20)
DEFAULT_PARTITION_ABLATION_BUDGET_RATIOS: tuple[float, ...] = (
    0.01,
    0.03,
    0.05,
    0.10,
    0.20,
)
SUPPORTED_PARTITION_SELECTION_METRICS = frozenset(
    {
        "silhouette",
        "balanced_silhouette",
        "validation_proxy",
        "budget_aware_validation_proxy",
        "downstream_proxy",
    }
)


def normalize_non_empty_names(
    name: str,
    values: Sequence[str] | str,
) -> tuple[str, ...]:
    raw_values = (values,) if isinstance(values, str) else tuple(values)
    normalized = tuple(str(value).strip() for value in raw_values)
    if not normalized or any(not value for value in normalized):
        raise ValueError(f"{name} must contain non-empty values")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must not contain duplicates")
    return normalized


def normalize_partition_selection_metrics(
    values: Sequence[str] | str,
) -> tuple[str, ...]:
    normalized = tuple(
        value.lower()
        for value in normalize_non_empty_names(
            "cluster_selection_metrics",
            values,
        )
    )
    unsupported = tuple(
        value
        for value in normalized
        if value not in SUPPORTED_PARTITION_SELECTION_METRICS
    )
    if unsupported:
        raise ValueError(
            "Unsupported cluster selection metrics: "
            f"{list(unsupported)}. Supported: "
            f"{sorted(SUPPORTED_PARTITION_SELECTION_METRICS)}"
        )
    if len(set(normalized)) != len(normalized):
        raise ValueError("cluster_selection_metrics must not contain duplicates")
    return normalized


def normalize_budget_ratios(values: Sequence[float]) -> tuple[float, ...]:
    normalized = tuple(float(value) for value in values)
    if not normalized:
        raise ValueError("budget_ratios must not be empty")
    if any(not math.isfinite(value) or not 0 < value <= 1 for value in normalized):
        raise ValueError("budget_ratios must contain finite values in (0, 1]")
    if len(set(normalized)) != len(normalized):
        raise ValueError("budget_ratios must not contain duplicates")
    tags = tuple(budget_ratio_tag(value) for value in normalized)
    if len(set(tags)) != len(tags):
        raise ValueError("budget_ratios must have unique percentage tags")
    return normalized


@dataclass(frozen=True)
class RMTPartitionSelectionAblationConfig:
    regression_suite: int | None = OPENML_REGRESSION_SUITE
    regression_tasks: Sequence[str] | None = DEFAULT_RMT_REGRESSION_TASKS
    models: Sequence[str] = ("lightgbm",)
    budget_ratios: Sequence[float] = DEFAULT_PARTITION_ABLATION_BUDGET_RATIOS
    cluster_selection_metrics: Sequence[str] = DEFAULT_PARTITION_SELECTION_METRICS
    n_partitions: int = 5
    validation_proxy_fraction: float = 0.2
    validation_proxy_min_partition_rows: int = 8
    validation_proxy_smoothing: float = 1.0
    min_sampled_rows_per_partition: int = 32
    budget_max_imbalance_ratio: float = 5.0
    budget_min_partition_fraction: float = 0.05
    include_single_partition_candidate: bool = True
    downstream_proxy_shortlist_size: int = 3
    downstream_proxy_n_estimators: int = 32
    downstream_complexity_penalty_weight: float = 0.01
    model_n_jobs: int = 1
    max_train_rows: int | None = 300_000
    seed: int = 42
    show_progress: bool = True
    synthetic_smoke: bool = False
    resume_from: str | Path | None = None
    resume_policy: str = ResumePolicy.RETRY_FAILED.value

    def __post_init__(self) -> None:
        tasks = (
            None
            if self.regression_tasks is None
            else normalize_non_empty_names(
                "regression_tasks",
                self.regression_tasks,
            )
        )
        models = tuple(
            model.lower()
            for model in normalize_non_empty_names(
                "models",
                self.models,
            )
        )
        if len(set(models)) != len(models):
            raise ValueError("models must not contain case-insensitive duplicates")
        if self.n_partitions < 1:
            raise ValueError("n_partitions must be positive")
        if not 0 < float(self.validation_proxy_fraction) < 0.5:
            raise ValueError("validation_proxy_fraction must be in (0, 0.5)")
        if self.validation_proxy_min_partition_rows < 1:
            raise ValueError("validation_proxy_min_partition_rows must be positive")
        if self.min_sampled_rows_per_partition < 1:
            raise ValueError("min_sampled_rows_per_partition must be positive")
        if self.budget_max_imbalance_ratio < 1:
            raise ValueError("budget_max_imbalance_ratio must be at least 1")
        if not 0 <= self.budget_min_partition_fraction <= 1:
            raise ValueError("budget_min_partition_fraction must be in [0, 1]")
        if self.downstream_proxy_shortlist_size < 1:
            raise ValueError("downstream_proxy_shortlist_size must be positive")
        if self.downstream_proxy_n_estimators < 1:
            raise ValueError("downstream_proxy_n_estimators must be positive")
        if self.downstream_complexity_penalty_weight < 0:
            raise ValueError(
                "downstream_complexity_penalty_weight must be non-negative"
            )
        if not math.isfinite(float(self.validation_proxy_smoothing)):
            raise ValueError("validation_proxy_smoothing must be finite")
        if self.validation_proxy_smoothing <= 0:
            raise ValueError("validation_proxy_smoothing must be positive")
        if (
            isinstance(self.model_n_jobs, bool)
            or not isinstance(self.model_n_jobs, Integral)
            or self.model_n_jobs < 1
        ):
            raise ValueError("model_n_jobs must be a positive integer")
        if self.max_train_rows is not None and self.max_train_rows < 1:
            raise ValueError("max_train_rows must be positive when provided")

        object.__setattr__(self, "regression_tasks", tasks)
        object.__setattr__(self, "models", models)
        object.__setattr__(
            self,
            "budget_ratios",
            normalize_budget_ratios(self.budget_ratios),
        )
        object.__setattr__(
            self,
            "cluster_selection_metrics",
            normalize_partition_selection_metrics(self.cluster_selection_metrics),
        )
        object.__setattr__(
            self,
            "resume_policy",
            ResumePolicy.parse(self.resume_policy).value,
        )
        object.__setattr__(self, "model_n_jobs", int(self.model_n_jobs))

    def to_regression_config(self) -> RMTRegressionExperimentConfig:
        return RMTRegressionExperimentConfig(
            regression_suite=self.regression_suite,
            regression_tasks=self.regression_tasks,
            strategies=("rmt_contraction",),
            models=self.models,
            ensemble_methods=("routed_weighted",),
            budget_ratios=self.budget_ratios,
            view_strategies=("gaussian",),
            router_modes=("spectral",),
            n_partitions=self.n_partitions,
            max_train_rows=self.max_train_rows,
            seed=self.seed,
            show_progress=self.show_progress,
            synthetic_smoke=self.synthetic_smoke,
            resume_from=self.resume_from,
            resume_policy=self.resume_policy,
        )


@dataclass(frozen=True)
class RMTPartitionSelectionGridPoint:
    cluster_selection_metric: str
    budget_ratio: float

    @property
    def config_name(self) -> str:
        return (
            "rmt_contraction__view_gaussian__routed_weighted"
            "__router_spectral"
            f"__selection_{self.cluster_selection_metric}"
            f"__budget_{budget_ratio_tag(self.budget_ratio)}"
        )


def make_rmt_partition_selection_grid(
    *,
    cluster_selection_metrics: Sequence[str] = DEFAULT_PARTITION_SELECTION_METRICS,
    budget_ratios: Sequence[float] = DEFAULT_PARTITION_ABLATION_BUDGET_RATIOS,
) -> tuple[RMTPartitionSelectionGridPoint, ...]:
    metrics = normalize_partition_selection_metrics(cluster_selection_metrics)
    budgets = normalize_budget_ratios(budget_ratios)
    return tuple(
        RMTPartitionSelectionGridPoint(
            cluster_selection_metric=metric,
            budget_ratio=budget_ratio,
        )
        for budget_ratio in budgets
        for metric in metrics
    )


def make_rmt_partition_selection_strategy_configs(
    config: RMTPartitionSelectionAblationConfig,
) -> dict[str, dict[str, Any]]:
    configs: dict[str, dict[str, Any]] = {
        "full_dataset": {
            "strategy": "full_dataset",
            "force_direct_model": True,
            "ensemble_method": "full_dataset",
            "budget_ratio": 1.0,
        }
    }
    grid = make_rmt_partition_selection_grid(
        cluster_selection_metrics=config.cluster_selection_metrics,
        budget_ratios=config.budget_ratios,
    )
    for grid_point in tqdm(
        grid,
        desc="Build partition-selection ablation grid",
        disable=not config.show_progress,
        leave=False,
    ):
        strategy_config = make_chunking_strategy_configs(
            problem_type="regression",
            strategy_names=("rmt_contraction",),
            n_partitions=config.n_partitions,
            seed=config.seed,
            ensemble_method="routed_weighted",
            budget_ratio=grid_point.budget_ratio,
            force_chunking=True,
            extra_strategy_params={
                "rmt_contraction": {
                    "view_strategy": "gaussian",
                    "router": "spectral",
                    "cluster_selection_metric": (
                        grid_point.cluster_selection_metric
                    ),
                    "validation_proxy_fraction": (
                        config.validation_proxy_fraction
                    ),
                    "validation_proxy_min_partition_rows": (
                        config.validation_proxy_min_partition_rows
                    ),
                    "validation_proxy_smoothing": (
                        config.validation_proxy_smoothing
                    ),
                    "budget_feasibility_mode": (
                        "hard"
                        if grid_point.cluster_selection_metric
                        in {
                            "budget_aware_validation_proxy",
                            "downstream_proxy",
                        }
                        else "off"
                    ),
                    "min_sampled_rows_per_partition": (
                        config.min_sampled_rows_per_partition
                    ),
                    "budget_max_imbalance_ratio": (
                        config.budget_max_imbalance_ratio
                    ),
                    "budget_min_partition_fraction": (
                        config.budget_min_partition_fraction
                    ),
                    "include_single_partition_candidate": (
                        config.include_single_partition_candidate
                    ),
                    "cluster_ensemble_method": (
                        "best_score"
                        if grid_point.cluster_selection_metric
                        == "downstream_proxy"
                        else "coassociation"
                    ),
                    "downstream_proxy_shortlist_size": (
                        config.downstream_proxy_shortlist_size
                    ),
                    "downstream_proxy_n_estimators": (
                        config.downstream_proxy_n_estimators
                    ),
                    "downstream_complexity_penalty_weight": (
                        config.downstream_complexity_penalty_weight
                    ),
                }
            },
        )["rmt_contraction"]
        configs[grid_point.config_name] = strategy_config
    return configs


class RMTPartitionSelectionAblationOrchestrator(
    RMTRegressionExperimentOrchestrator
):
    def __init__(self, config: RMTPartitionSelectionAblationConfig) -> None:
        self.ablation_config = config
        super().__init__(config.to_regression_config())

    def _build_experiment_plan(self) -> ExperimentPlan:
        effective_config = {
            "experiment_kind": "rmt_partition_selection_ablation",
            **asdict(self.ablation_config),
            "ensemble_methods": ["routed_weighted"],
            "view_strategies": ["gaussian"],
            "router_modes": ["spectral"],
        }
        plan = build_standard_rmt_experiment_plan(effective_config)
        self.experiment_plan = plan
        return plan

    def _create_logger(self) -> BenchmarkLogger:
        if self.ablation_config.resume_from is not None:
            return super()._create_logger()
        run_id = (
            "run_rmt_partition_selection_ablation_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        return BenchmarkLogger(
            run_id=run_id,
            artifacts_root=BENCHMARK_DIR / "results",
        )

    def _build_strategy_grid(self) -> StrategyGridContract:
        return normalize_strategy_grid(
            make_rmt_partition_selection_strategy_configs(
                self.ablation_config
            )
        )

    def _make_model_pool(self) -> dict[str, Any]:
        return make_model_pool(
            seed=self.ablation_config.seed,
            model_names=self.ablation_config.models,
            problem_type="regression",
            n_jobs=self.ablation_config.model_n_jobs,
        )

    def _build_run_meta(
        self,
        logger: BenchmarkLogger,
        run_records: Sequence[Mapping[str, Any]],
        status: str = "completed",
    ) -> dict[str, Any]:
        metadata = super()._build_run_meta(logger, run_records, status)
        metadata.update(
            {
                "experiment_kind": "rmt_partition_selection_ablation",
                "cluster_selection_metrics": list(
                    self.ablation_config.cluster_selection_metrics
                ),
                "validation_proxy_fraction": (
                    self.ablation_config.validation_proxy_fraction
                ),
                "validation_proxy_min_partition_rows": (
                    self.ablation_config.validation_proxy_min_partition_rows
                ),
                "validation_proxy_smoothing": (
                    self.ablation_config.validation_proxy_smoothing
                ),
                "min_sampled_rows_per_partition": (
                    self.ablation_config.min_sampled_rows_per_partition
                ),
                "budget_max_imbalance_ratio": (
                    self.ablation_config.budget_max_imbalance_ratio
                ),
                "budget_min_partition_fraction": (
                    self.ablation_config.budget_min_partition_fraction
                ),
                "downstream_proxy_shortlist_size": (
                    self.ablation_config.downstream_proxy_shortlist_size
                ),
                "downstream_proxy_n_estimators": (
                    self.ablation_config.downstream_proxy_n_estimators
                ),
                "downstream_complexity_penalty_weight": (
                    self.ablation_config.downstream_complexity_penalty_weight
                ),
                "model_n_jobs": self.ablation_config.model_n_jobs,
            }
        )
        return metadata


def run_rmt_partition_selection_ablation(
    regression_tasks: Sequence[str] | None = None,
    models: Sequence[str] = ("lightgbm",),
    budget_ratios: Sequence[float] = DEFAULT_PARTITION_ABLATION_BUDGET_RATIOS,
    cluster_selection_metrics: Sequence[str] = DEFAULT_PARTITION_SELECTION_METRICS,
    max_train_rows: int | None = 300_000,
    model_n_jobs: int = 1,
    seed: int = 42,
    show_progress: bool = True,
    resume_from: str | Path | None = None,
    resume_policy: str = ResumePolicy.RETRY_FAILED.value,
) -> Path:
    config = RMTPartitionSelectionAblationConfig(
        regression_tasks=(
            DEFAULT_RMT_REGRESSION_TASKS
            if regression_tasks is None
            else regression_tasks
        ),
        models=models,
        budget_ratios=budget_ratios,
        cluster_selection_metrics=cluster_selection_metrics,
        max_train_rows=max_train_rows,
        model_n_jobs=model_n_jobs,
        seed=seed,
        show_progress=show_progress,
        resume_from=resume_from,
        resume_policy=resume_policy,
    )
    return RMTPartitionSelectionAblationOrchestrator(config).run()


def run_rmt_partition_selection_mechanism_smoke(
    *,
    regression_tasks: Sequence[str] = DEFAULT_MECHANISM_SMOKE_TASKS,
    models: Sequence[str] = ("lightgbm",),
    max_train_rows: int | None = 300_000,
    seed: int = 42,
    show_progress: bool = True,
) -> Path:
    """Run the first post-refactor gate before the full seven-dataset grid."""

    config = RMTPartitionSelectionAblationConfig(
        regression_tasks=regression_tasks,
        models=models,
        budget_ratios=DEFAULT_MECHANISM_SMOKE_BUDGET_RATIOS,
        cluster_selection_metrics=DEFAULT_PARTITION_SELECTION_METRICS,
        max_train_rows=max_train_rows,
        model_n_jobs=1,
        seed=seed,
        show_progress=show_progress,
    )
    return RMTPartitionSelectionAblationOrchestrator(config).run()


def _run_cli() -> Path:
    parser = argparse.ArgumentParser(
        description="Compare RMT partition-selection objectives."
    )
    parser.add_argument(
        "--mechanism-smoke",
        action="store_true",
        help="Run the three-dataset, three-budget preflight grid.",
    )
    parser.add_argument("--max-train-rows", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    common = {
        "max_train_rows": args.max_train_rows,
        "seed": args.seed,
        "show_progress": not args.no_progress,
    }
    if args.mechanism_smoke:
        return run_rmt_partition_selection_mechanism_smoke(**common)
    return run_rmt_partition_selection_ablation(**common)


if __name__ == "__main__":
    _run_cli()
