from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
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
from rmt_experiment_utils import budget_ratio_tag  # noqa: E402
from rmt_partition_selection_ablation import (  # noqa: E402
    DEFAULT_MECHANISM_SMOKE_BUDGET_RATIOS,
    DEFAULT_MECHANISM_SMOKE_TASKS,
    DEFAULT_PARTITION_ABLATION_BUDGET_RATIOS,
    RMTPartitionSelectionAblationConfig,
    RMTPartitionSelectionAblationOrchestrator,
    make_rmt_partition_selection_strategy_configs,
    normalize_non_empty_names,
)
from rmt_regression_medium_datasets import DEFAULT_RMT_REGRESSION_TASKS  # noqa: E402
from sampling_zoo.core.experiment.contracts import StrategyGridContract  # noqa: E402
from sampling_zoo.core.experiment.morphisms import (  # noqa: E402
    build_standard_rmt_experiment_plan,
    normalize_strategy_grid,
)
from sampling_zoo.core.experiment.resume import ResumePolicy  # noqa: E402
from sampling_zoo.core.experiment.stages import ExperimentPlan  # noqa: E402


DEFAULT_ROW_SELECTION_METHODS: tuple[str, ...] = (
    "hybrid",
    "leverage",
    "maxvol",
    "capped_leverage",
)
SUPPORTED_ROW_SELECTION_METHODS = frozenset(
    {"all", "hybrid", "leverage", "maxvol", "capped_leverage"}
)


def normalize_row_selection_methods(
    values: Sequence[str] | str,
) -> tuple[str, ...]:
    normalized = tuple(
        value.lower()
        for value in normalize_non_empty_names("row_selection_methods", values)
    )
    unsupported = tuple(
        value for value in normalized if value not in SUPPORTED_ROW_SELECTION_METHODS
    )
    if unsupported:
        raise ValueError(
            f"Unsupported row selection methods: {list(unsupported)}. "
            f"Supported: {sorted(SUPPORTED_ROW_SELECTION_METHODS)}"
        )
    if len(set(normalized)) != len(normalized):
        raise ValueError("row_selection_methods must not contain duplicates")
    return normalized


@dataclass(frozen=True)
class RMTRowSelectionAblationConfig(RMTPartitionSelectionAblationConfig):
    cluster_selection_metrics: Sequence[str] = ("balanced_silhouette",)
    row_selection_methods: Sequence[str] = DEFAULT_ROW_SELECTION_METHODS
    leverage_cap_quantile: float = 0.95

    def __post_init__(self) -> None:
        super().__post_init__()
        if not 0 < float(self.leverage_cap_quantile) <= 1:
            raise ValueError("leverage_cap_quantile must be in (0, 1]")
        if tuple(self.cluster_selection_metrics) != ("balanced_silhouette",):
            raise ValueError(
                "row-selection ablation fixes cluster_selection_metrics to "
                "('balanced_silhouette',)"
            )
        object.__setattr__(
            self,
            "row_selection_methods",
            normalize_row_selection_methods(self.row_selection_methods),
        )
        object.__setattr__(
            self,
            "leverage_cap_quantile",
            float(self.leverage_cap_quantile),
        )


def make_rmt_row_selection_strategy_configs(
    config: RMTRowSelectionAblationConfig,
) -> dict[str, dict[str, Any]]:
    base_configs = make_rmt_partition_selection_strategy_configs(config)
    configs = {"full_dataset": base_configs["full_dataset"]}
    rmt_configs = tuple(
        strategy_config
        for name, strategy_config in base_configs.items()
        if name != "full_dataset"
    )
    grid = tuple(
        (strategy_config, method)
        for strategy_config in rmt_configs
        for method in config.row_selection_methods
    )
    for strategy_config, method in tqdm(
        grid,
        desc="Build row-selection ablation grid",
        disable=not config.show_progress,
        leave=False,
    ):
        materialized = dict(strategy_config)
        materialized["selection_method"] = method
        materialized["leverage_cap_quantile"] = config.leverage_cap_quantile
        budget_tag = budget_ratio_tag(materialized["budget_ratio"])
        name = (
            "rmt_contraction__view_gaussian__routed_weighted"
            "__router_spectral__selection_balanced_silhouette"
            f"__rows_{method}__budget_{budget_tag}"
        )
        configs[name] = materialized
    return configs


class RMTRowSelectionAblationOrchestrator(
    RMTPartitionSelectionAblationOrchestrator
):
    def __init__(self, config: RMTRowSelectionAblationConfig) -> None:
        self.row_ablation_config = config
        super().__init__(config)

    def _build_experiment_plan(self) -> ExperimentPlan:
        effective_config = {
            "experiment_kind": "rmt_row_selection_ablation",
            **asdict(self.row_ablation_config),
            "ensemble_methods": ["routed_weighted"],
            "view_strategies": ["gaussian"],
            "router_modes": ["spectral"],
        }
        plan = build_standard_rmt_experiment_plan(effective_config)
        self.experiment_plan = plan
        return plan

    def _create_logger(self) -> BenchmarkLogger:
        if self.row_ablation_config.resume_from is not None:
            return super()._create_logger()
        run_id = (
            "run_rmt_row_selection_ablation_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        return BenchmarkLogger(
            run_id=run_id,
            artifacts_root=BENCHMARK_DIR / "results",
        )

    def _build_strategy_grid(self) -> StrategyGridContract:
        return normalize_strategy_grid(
            make_rmt_row_selection_strategy_configs(self.row_ablation_config)
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
                "experiment_kind": "rmt_row_selection_ablation",
                "row_selection_methods": list(
                    self.row_ablation_config.row_selection_methods
                ),
                "leverage_cap_quantile": (
                    self.row_ablation_config.leverage_cap_quantile
                ),
            }
        )
        return metadata


def run_rmt_row_selection_ablation(
    regression_tasks: Sequence[str] | None = None,
    models: Sequence[str] = ("lightgbm",),
    budget_ratios: Sequence[float] = DEFAULT_PARTITION_ABLATION_BUDGET_RATIOS,
    row_selection_methods: Sequence[str] = DEFAULT_ROW_SELECTION_METHODS,
    leverage_cap_quantile: float = 0.95,
    max_train_rows: int | None = 300_000,
    model_n_jobs: int = 1,
    seed: int = 42,
    show_progress: bool = True,
    resume_from: str | Path | None = None,
    resume_policy: str = ResumePolicy.RETRY_FAILED.value,
) -> Path:
    config = RMTRowSelectionAblationConfig(
        regression_tasks=(
            DEFAULT_RMT_REGRESSION_TASKS
            if regression_tasks is None
            else regression_tasks
        ),
        models=models,
        budget_ratios=budget_ratios,
        row_selection_methods=row_selection_methods,
        leverage_cap_quantile=leverage_cap_quantile,
        max_train_rows=max_train_rows,
        model_n_jobs=model_n_jobs,
        seed=seed,
        show_progress=show_progress,
        resume_from=resume_from,
        resume_policy=resume_policy,
    )
    return RMTRowSelectionAblationOrchestrator(config).run()


def run_rmt_row_selection_mechanism_smoke(
    *,
    regression_tasks: Sequence[str] = DEFAULT_MECHANISM_SMOKE_TASKS,
    models: Sequence[str] = ("lightgbm",),
    max_train_rows: int | None = 300_000,
    seed: int = 42,
    show_progress: bool = True,
) -> Path:
    return run_rmt_row_selection_ablation(
        regression_tasks=regression_tasks,
        models=models,
        budget_ratios=DEFAULT_MECHANISM_SMOKE_BUDGET_RATIOS,
        max_train_rows=max_train_rows,
        model_n_jobs=1,
        seed=seed,
        show_progress=show_progress,
    )


def _run_cli() -> Path:
    parser = argparse.ArgumentParser(
        description="Compare row-selection policies inside fixed RMT partitions."
    )
    parser.add_argument("--mechanism-smoke", action="store_true")
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
        return run_rmt_row_selection_mechanism_smoke(**common)
    return run_rmt_row_selection_ablation(**common)


if __name__ == "__main__":
    _run_cli()
