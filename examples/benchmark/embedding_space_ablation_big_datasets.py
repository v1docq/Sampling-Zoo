from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
from tqdm.auto import tqdm

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
BENCHMARK_DIR = Path(__file__).resolve().parent
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

from benchmark_datasets import RawDatasetBundle  # noqa: E402
from benchmark_runner import EnsembleChunkBenchmarkRunner  # noqa: E402
from embedding_space_ablation_medium_datasets import (  # noqa: E402
    COMPARISON_ENSEMBLE_METHODS,
    COMPARISON_ROUTER_MODES,
    COMPARISON_STRATEGIES,
    FOUNDATION_MODELS,
    EmbeddingSpaceAblationConfig,
    EmbeddingSpaceAblationMediumOrchestrator,
    make_embedding_ablation_strategy_configs,
)
from sampling_zoo.core.experiment.contracts import StrategyGridContract  # noqa: E402
from sampling_zoo.core.experiment.morphisms import normalize_strategy_grid  # noqa: E402
from sampling_zoo.core.experiment.resume import ResumePolicy  # noqa: E402


@dataclass(frozen=True)
class BigDatasetAblationPlan:
    task: str
    budgets: tuple[float, ...]
    cluster_algorithms: tuple[str, ...]
    historical_winner: str
    rationale: str


# Both clusterers are evaluated for every representation and ``best_score``
# selects the winner, matching the mechanism used by the earlier experiments.
# ``historical_winner`` is metadata only and is never used to restrict the
# search.  Every recorded paper winner used two partitions.
DEFAULT_BIG_DATASET_PLANS: tuple[BigDatasetAblationPlan, ...] = (
    BigDatasetAblationPlan(
        task="Airlines_DepDelay_10M",
        budgets=(0.01, 0.05, 0.10),
        cluster_algorithms=("kmeans", "gmm"),
        historical_winner="kmeans",
        rationale="Exact paper grid; 0.10 was the best recorded RMT budget.",
    ),
    BigDatasetAblationPlan(
        task="Buzzinsocialmedia_Twitter",
        budgets=(0.05, 0.10, 0.20),
        cluster_algorithms=("kmeans", "gmm"),
        historical_winner="gmm",
        rationale="Recorded GMM winner; 0.20/0.10 were the model-specific optima.",
    ),
    BigDatasetAblationPlan(
        task="nyc-taxi-green-dec-2016",
        budgets=(0.10, 0.30),
        cluster_algorithms=("kmeans", "gmm"),
        historical_winner="kmeans",
        rationale="0.30 was the TabPFN optimum; omit costly 0.50-0.90 contexts.",
    ),
    BigDatasetAblationPlan(
        task="Yolanda",
        budgets=(0.05, 0.10, 0.20),
        cluster_algorithms=("kmeans", "gmm"),
        historical_winner="gmm",
        rationale="Recorded GMM winner; cap contexts at 48k selected rows.",
    ),
    BigDatasetAblationPlan(
        task="Allstate_Claims_Severity",
        budgets=(0.10, 0.30, 0.50),
        cluster_algorithms=("kmeans", "gmm"),
        historical_winner="kmeans",
        rationale="Exact completed paper grid; 0.50 was the optimum.",
    ),
    BigDatasetAblationPlan(
        task="black_friday",
        budgets=(0.10, 0.30, 0.50),
        cluster_algorithms=("kmeans", "gmm"),
        historical_winner="kmeans",
        rationale="Retain the stable paper range; omit costly 0.75/0.90 runs.",
    ),
)

DEFAULT_BIG_TASKS: tuple[str, ...] = tuple(
    plan.task for plan in DEFAULT_BIG_DATASET_PLANS
)
PAPER_TRAIN_ROWS: dict[str, int] = {
    "Airlines_DepDelay_10M": 240_000,
    "Buzzinsocialmedia_Twitter": 240_000,
    "nyc-taxi-green-dec-2016": 240_000,
    "Yolanda": 240_000,
    "Allstate_Claims_Severity": 135_588,
    "black_friday": 120_110,
}
DEFAULT_BIG_BUDGETS: tuple[float, ...] = tuple(
    sorted({budget for plan in DEFAULT_BIG_DATASET_PLANS for budget in plan.budgets})
)
SUPPORTED_CONTROLLED_CLUSTERERS = frozenset({"kmeans", "gmm"})


@dataclass(frozen=True)
class EmbeddingSpaceAblationBigConfig(EmbeddingSpaceAblationConfig):
    regression_tasks: Sequence[str] | None = DEFAULT_BIG_TASKS
    budget_ratios: Sequence[float] = DEFAULT_BIG_BUDGETS
    controlled_cluster_algorithms: Sequence[str] = ("kmeans", "gmm")
    task_plans: Sequence[BigDatasetAblationPlan] = DEFAULT_BIG_DATASET_PLANS

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.cluster_profile != "controlled":
            raise ValueError("Big ablation requires the controlled profile")
        plans = tuple(self.task_plans)
        if not plans:
            raise ValueError("task_plans must not be empty")
        plan_tasks = tuple(plan.task for plan in plans)
        if len(set(plan_tasks)) != len(plan_tasks):
            raise ValueError("task_plans must not contain duplicate tasks")
        requested_tasks = tuple(self.regression_tasks or ())
        if set(plan_tasks) != set(requested_tasks):
            raise ValueError(
                "task_plans must describe every requested task exactly once"
            )
        for plan in plans:
            algorithms = tuple(plan.cluster_algorithms)
            if not algorithms or len(set(algorithms)) != len(algorithms):
                raise ValueError(
                    f"Invalid clusterer grid for {plan.task}: {algorithms}"
                )
            unsupported = set(algorithms) - SUPPORTED_CONTROLLED_CLUSTERERS
            if unsupported:
                raise ValueError(
                    f"Unsupported clusterers for {plan.task}: "
                    f"{sorted(unsupported)}"
                )
            if plan.historical_winner not in algorithms:
                raise ValueError(
                    f"Historical winner for {plan.task} must be in its grid"
                )
            if not plan.budgets or any(
                not 0.0 < float(budget) <= 1.0
                for budget in plan.budgets
            ):
                raise ValueError(
                    f"Invalid budget grid for {plan.task}: {plan.budgets}"
                )


class EmbeddingSpaceAblationBigOrchestrator(
    EmbeddingSpaceAblationMediumOrchestrator
):
    def __init__(self, config: EmbeddingSpaceAblationBigConfig) -> None:
        super().__init__(config)
        self.config = config
        self._plans_by_task = {
            plan.task: plan for plan in self.config.task_plans
        }

    def _run_id_prefix(self) -> str:
        return "run_embedding_space_ablation_big"

    def _config_for_plan(
        self,
        plan: BigDatasetAblationPlan,
    ) -> EmbeddingSpaceAblationBigConfig:
        return replace(
            self.config,
            regression_tasks=(plan.task,),
            budget_ratios=plan.budgets,
            controlled_cluster_algorithms=plan.cluster_algorithms,
            task_plans=(plan,),
        )

    def _strategy_grid_for_plan(
        self,
        plan: BigDatasetAblationPlan,
    ) -> StrategyGridContract:
        return normalize_strategy_grid(
            make_embedding_ablation_strategy_configs(
                self._config_for_plan(plan)
            )
        )

    def _build_strategy_grid(self) -> StrategyGridContract:
        # The experiment stage contract expects a grid.  Actual grids are
        # dataset-specific and are constructed in ``_run_experiment``.
        return self._strategy_grid_for_plan(tuple(self.config.task_plans)[0])

    @staticmethod
    def _task_name(dataset: RawDatasetBundle) -> str:
        return dataset.name.split("__task_", 1)[0]

    def _run_experiment(
        self,
        datasets: Sequence[RawDatasetBundle],
        strategy_configs: StrategyGridContract
        | Mapping[str, Mapping[str, Any]],
        runner: EnsembleChunkBenchmarkRunner,
    ) -> list[dict[str, Any]]:
        del strategy_configs
        run_records: list[dict[str, Any]] = (
            []
            if self.resume_session is None
            else [
                dict(record)
                for record in self.resume_session.plan.retained_records
            ]
        )
        for dataset in tqdm(
            datasets,
            desc="Run big embedding ablation",
            disable=not self.config.show_progress,
            leave=False,
        ):
            task_name = self._task_name(dataset)
            try:
                plan = self._plans_by_task[task_name]
            except KeyError as ex:
                raise RuntimeError(
                    f"No big-ablation plan for loaded dataset {dataset.name}"
                ) from ex
            print(
                "[big-ablation] "
                f"task={task_name} budgets={list(plan.budgets)} "
                f"clusterers={list(plan.cluster_algorithms)} "
                "selection=best_score",
                flush=True,
            )
            run_records.extend(
                runner.run_dataset(
                    dataset,
                    self._strategy_grid_for_plan(plan),
                    self._make_model_pool(),
                )
            )
        return run_records

    def _comparison_protocol(self) -> dict[str, Any]:
        pair_methods_per_budget = sum(
            len(self.config.router_modes)
            if method == "routed_weighted"
            else 1
            for method in self.config.ensemble_methods
        )
        plans = [
            {
                **asdict(plan),
                "selected_rows_on_paper_split": [
                    int(round(PAPER_TRAIN_ROWS[plan.task] * budget))
                    for budget in plan.budgets
                ],
            }
            for plan in self.config.task_plans
        ]
        return {
            "question": (
                "Does RMT/SVD row representation improve matched-budget "
                "chunking over standardized raw features on large tasks?"
            ),
            "profile": "controlled_task_specific",
            "controlled_factors": {
                "n_partitions": self.config.n_partitions,
                "clusterer_policy": (
                    "KMeans and GMM evaluated in each representation; "
                    "best_score selects the winner"
                ),
                "ensemble_methods": list(self.config.ensemble_methods),
                "router_modes": list(self.config.router_modes),
                "seed": self.config.seed,
                "max_train_rows": self.config.max_train_rows,
            },
            "task_plans": plans,
            "representations": {
                "rmt_contraction": "random-view SVD sample embedding",
                "raw_feature_clustering": (
                    "standardized encoded raw features"
                ),
            },
            "include_direct_baseline": False,
            "expected_leaf_runs": sum(
                len(COMPARISON_STRATEGIES)
                * len(plan.budgets)
                * pair_methods_per_budget
                * len(self.config.models)
                for plan in self.config.task_plans
            ),
            "cache_expectation": (
                "both clusterers are fitted once per representation; "
                "structure is reused across budgets/models; materialized "
                "partitions are budget-specific; trained chunk models are "
                "model-specific and reused between voting and routing"
            ),
        }

    def _validate_completed_comparison(
        self,
        comparison_tables: Mapping[str, pd.DataFrame],
    ) -> None:
        EmbeddingSpaceAblationMediumOrchestrator._validate_completed_comparison(
            comparison_tables
        )
        protocol = self._comparison_protocol()
        expected_runs = int(protocol["expected_leaf_runs"])
        runs = comparison_tables["runs"]
        pairs = comparison_tables["pairs"]
        expected_pairs = expected_runs // len(COMPARISON_STRATEGIES)
        if len(runs) != expected_runs or len(pairs) != expected_pairs:
            raise RuntimeError(
                "Big embedding ablation coverage check failed: "
                f"expected_runs={expected_runs}, actual_runs={len(runs)}, "
                f"expected_pairs={expected_pairs}, actual_pairs={len(pairs)}"
            )
        for plan in self.config.task_plans:
            task_runs = runs[
                runs["dataset"].astype(str).str.startswith(
                    f"{plan.task}__task_"
                )
            ]
            actual_budgets = set(
                pd.to_numeric(
                    task_runs["budget_ratio"], errors="coerce"
                ).dropna()
            )
            expected_budgets = {float(value) for value in plan.budgets}
            selected_algorithms = set(
                task_runs["selected_cluster_algorithm"].dropna()
            )
            expected_algorithm_signature = "|".join(
                plan.cluster_algorithms
            )
            configured_signatures = set(
                task_runs["configured_cluster_algorithms"].dropna()
            )
            evaluated_signatures = set(
                task_runs["evaluated_cluster_algorithms"].dropna()
            )
            if (
                actual_budgets != expected_budgets
                or not selected_algorithms
                or not selected_algorithms.issubset(
                    set(plan.cluster_algorithms)
                )
                or configured_signatures != {
                    expected_algorithm_signature
                }
                or evaluated_signatures != {
                    expected_algorithm_signature
                }
            ):
                raise RuntimeError(
                    "Big embedding ablation task contract failed for "
                    f"{plan.task}: budgets={sorted(actual_budgets)}, "
                    f"selected_clusterers={sorted(selected_algorithms)}, "
                    f"configured={sorted(configured_signatures)}, "
                    f"evaluated={sorted(evaluated_signatures)}, "
                    f"expected={expected_algorithm_signature}"
                )


def _select_task_plans(
    tasks: Sequence[str] | None,
    budget_override: Sequence[float] | None,
    clusterer_override: Sequence[str] | None,
) -> tuple[BigDatasetAblationPlan, ...]:
    requested = set(tasks or DEFAULT_BIG_TASKS)
    unknown = requested - set(DEFAULT_BIG_TASKS)
    if unknown:
        raise ValueError(f"Unknown big tasks: {sorted(unknown)}")
    selected: list[BigDatasetAblationPlan] = []
    for plan in DEFAULT_BIG_DATASET_PLANS:
        if plan.task not in requested:
            continue
        selected.append(
            replace(
                plan,
                budgets=(
                    tuple(float(value) for value in budget_override)
                    if budget_override
                    else plan.budgets
                ),
                cluster_algorithms=(
                    tuple(clusterer_override)
                    if clusterer_override is not None
                    else plan.cluster_algorithms
                ),
                historical_winner=(
                    tuple(clusterer_override)[0]
                    if clusterer_override is not None
                    and plan.historical_winner not in clusterer_override
                    else plan.historical_winner
                ),
            )
        )
    return tuple(selected)


def run_embedding_space_ablation_big(
    *,
    tasks: Sequence[str] | None = None,
    models: Sequence[str] = FOUNDATION_MODELS,
    budget_override: Sequence[float] | None = None,
    clusterer_override: Sequence[str] | None = None,
    max_train_rows: int | None = 300_000,
    seed: int = 42,
    show_progress: bool = True,
    output_dir: str | Path | None = None,
    resume_from: str | Path | None = None,
    resume_policy: str = ResumePolicy.RETRY_FAILED.value,
) -> Path:
    plans = _select_task_plans(
        tasks,
        budget_override,
        clusterer_override,
    )
    config = EmbeddingSpaceAblationBigConfig(
        regression_tasks=tuple(plan.task for plan in plans),
        models=tuple(models),
        strategies=COMPARISON_STRATEGIES,
        ensemble_methods=COMPARISON_ENSEMBLE_METHODS,
        router_modes=COMPARISON_ROUTER_MODES,
        budget_ratios=tuple(
            sorted({budget for plan in plans for budget in plan.budgets})
        ),
        controlled_cluster_algorithms=tuple(
            sorted(
                {
                    algorithm
                    for plan in plans
                    for algorithm in plan.cluster_algorithms
                }
            )
        ),
        task_plans=plans,
        cluster_profile="controlled",
        n_partitions=2,
        max_train_rows=max_train_rows,
        seed=seed,
        show_progress=show_progress,
        include_direct_baseline=False,
        output_dir=output_dir,
        resume_from=resume_from,
        resume_policy=resume_policy,
    )
    return EmbeddingSpaceAblationBigOrchestrator(config).run()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Matched large-dataset ablation: RMT/SVD embedding versus raw "
            "standardized feature clustering."
        )
    )
    parser.add_argument(
        "--task",
        action="append",
        dest="tasks",
        choices=DEFAULT_BIG_TASKS,
    )
    parser.add_argument("--model", action="append", dest="models")
    parser.add_argument(
        "--budget",
        action="append",
        type=float,
        dest="budgets",
        help="Override the task-specific budget grid for every selected task.",
    )
    parser.add_argument(
        "--cluster-algorithm",
        action="append",
        dest="cluster_algorithms",
        choices=sorted(SUPPORTED_CONTROLLED_CLUSTERERS),
        help=(
            "Override the default KMeans+GMM grid; repeat the option to use "
            "multiple clusterers."
        ),
    )
    parser.add_argument("--max-train-rows", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument(
        "--resume-policy",
        default=ResumePolicy.RETRY_FAILED.value,
    )
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument(
        "--print-plan",
        action="store_true",
        help="Print the resolved task plan without starting the experiment.",
    )
    return parser.parse_args()


def _run_cli(args: argparse.Namespace) -> Path | None:
    plans = _select_task_plans(
        args.tasks,
        args.budgets,
        args.cluster_algorithms,
    )
    if args.print_plan:
        print(
            json.dumps(
                [asdict(plan) for plan in plans],
                ensure_ascii=False,
                indent=2,
            )
        )
        return None
    output_dir = args.output_dir
    if output_dir is None and args.resume_from is not None:
        output_dir = args.resume_from
    return run_embedding_space_ablation_big(
        tasks=tuple(plan.task for plan in plans),
        models=tuple(args.models or FOUNDATION_MODELS),
        budget_override=args.budgets,
        clusterer_override=args.cluster_algorithms,
        max_train_rows=args.max_train_rows,
        seed=args.seed,
        show_progress=not args.no_progress,
        output_dir=output_dir,
        resume_from=args.resume_from,
        resume_policy=args.resume_policy,
    )


if __name__ == "__main__":
    result_path = _run_cli(_parse_args())
    if result_path is not None:
        print(result_path)
