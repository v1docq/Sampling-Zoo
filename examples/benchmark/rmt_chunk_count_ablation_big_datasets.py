from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
BENCHMARK_DIR = Path(__file__).resolve().parent
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

from embedding_space_ablation_big_datasets import DEFAULT_BIG_TASKS  # noqa: E402
from rmt_chunk_count_ablation_medium_datasets import (  # noqa: E402
    DEFAULT_CHUNK_COUNT_CLUSTER_ALGORITHMS,
    DEFAULT_CHUNK_COUNT_ENSEMBLES,
    DEFAULT_CHUNK_COUNT_MODELS,
    RMTChunkCountAblationConfig,
    RMTChunkCountAblationMediumOrchestrator,
)
from rmt_experiment_utils import json_ready  # noqa: E402
from sampling_zoo.core.experiment.resume import ResumePolicy  # noqa: E402


DEFAULT_BIG_EXPERT_COUNT_TASKS: tuple[str, ...] = (
    "black_friday",
    "Allstate_Claims_Severity",
    "Yolanda",
    "Buzzinsocialmedia_Twitter",
    "nyc-taxi-green-dec-2016",
    "Airlines_DepDelay_10M",
)
DEFAULT_BIG_EXPERT_COUNT_BUDGETS: tuple[float, ...] = (0.1,)
DEFAULT_BIG_EXPERT_COUNTS: tuple[int, ...] = (2, 4, 8)


class RMTChunkCountAblationBigOrchestrator(
    RMTChunkCountAblationMediumOrchestrator
):
    def _run_id_prefix(self) -> str:
        return "run_rmt_chunk_count_ablation_big"


def make_big_foundation_config(
    *,
    tasks: Sequence[str] = DEFAULT_BIG_EXPERT_COUNT_TASKS,
    models: Sequence[str] = DEFAULT_CHUNK_COUNT_MODELS,
    budgets: Sequence[float] = DEFAULT_BIG_EXPERT_COUNT_BUDGETS,
    chunk_counts: Sequence[int] = DEFAULT_BIG_EXPERT_COUNTS,
    ensemble_methods: Sequence[str] = DEFAULT_CHUNK_COUNT_ENSEMBLES,
    voting_pruning_modes: Sequence[bool] = (True, False),
    cluster_algorithms: Sequence[str] = DEFAULT_CHUNK_COUNT_CLUSTER_ALGORITHMS,
    max_train_rows: int | None = 300_000,
    seed: int = 42,
    show_progress: bool = True,
    output_dir: str | Path | None = None,
    resume_from: str | Path | None = None,
    resume_policy: str = ResumePolicy.RETRY_FAILED.value,
    synthetic_smoke: bool = False,
) -> RMTChunkCountAblationConfig:
    unknown_tasks = set(tasks) - set(DEFAULT_BIG_TASKS)
    if unknown_tasks:
        raise ValueError(f"Unknown big tasks: {sorted(unknown_tasks)}")
    return RMTChunkCountAblationConfig(
        regression_tasks=tuple(tasks),
        models=tuple(models),
        ensemble_methods=tuple(ensemble_methods),
        budget_ratios=tuple(float(value) for value in budgets),
        chunk_counts=tuple(int(value) for value in chunk_counts),
        voting_pruning_modes=tuple(bool(value) for value in voting_pruning_modes),
        cluster_algorithms=tuple(cluster_algorithms),
        max_train_rows=max_train_rows,
        seed=seed,
        show_progress=show_progress,
        include_direct_baseline=False,
        output_dir=output_dir,
        resume_from=resume_from,
        resume_policy=resume_policy,
        synthetic_smoke=synthetic_smoke,
    )


def _plan_summary(config: RMTChunkCountAblationConfig) -> dict[str, object]:
    methods_per_k_budget = sum(
        1 if method == "routed_weighted" else len(config.voting_pruning_modes)
        for method in config.ensemble_methods
    )
    return {
        "profile": "quick_big_foundation_expert_count",
        "tasks": list(config.regression_tasks or ()),
        "models": list(config.models),
        "budgets": list(config.budget_ratios),
        "requested_experts": list(config.chunk_counts),
        "ensemble_methods": list(config.ensemble_methods),
        "voting_pruning": [
            "enabled" if enabled else "disabled"
            for enabled in config.voting_pruning_modes
        ],
        "cluster_algorithms": list(config.cluster_algorithms),
        "leaf_runs": (
            len(config.regression_tasks or ())
            * len(config.models)
            * len(config.budget_ratios)
            * len(config.chunk_counts)
            * methods_per_k_budget
        ),
        "actual_expert_fits_with_cache": (
            len(config.regression_tasks or ())
            * len(config.models)
            * len(config.budget_ratios)
            * sum(config.chunk_counts)
        ),
        "fixed_total_row_budget_across_k": True,
        "direct_baseline": False,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compact per-dataset foundation-model ablation over the number "
            "of RMT clusters and active experts. By default all large tasks "
            "run from lighter to heavier."
        )
    )
    parser.add_argument(
        "--task",
        action="append",
        dest="tasks",
        choices=DEFAULT_BIG_TASKS,
    )
    parser.add_argument("--model", action="append", dest="models")
    parser.add_argument("--budget", action="append", type=float, dest="budgets")
    parser.add_argument(
        "--chunk-count", action="append", type=int, dest="chunk_counts"
    )
    parser.add_argument(
        "--ensemble-method",
        action="append",
        choices=DEFAULT_CHUNK_COUNT_ENSEMBLES,
        dest="ensemble_methods",
    )
    parser.add_argument(
        "--voting-pruning",
        choices=("on", "off", "both"),
        default="both",
    )
    parser.add_argument(
        "--cluster-algorithm",
        action="append",
        choices=DEFAULT_CHUNK_COUNT_CLUSTER_ALGORITHMS,
        dest="cluster_algorithms",
    )
    parser.add_argument("--max-train-rows", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument(
        "--resume-policy", default=ResumePolicy.RETRY_FAILED.value
    )
    parser.add_argument("--print-plan", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def _config_from_args(args: argparse.Namespace) -> RMTChunkCountAblationConfig:
    smoke = bool(args.smoke)
    output_dir = args.output_dir
    if output_dir is None and args.resume_from is not None:
        output_dir = args.resume_from
    return make_big_foundation_config(
        tasks=tuple(args.tasks or DEFAULT_BIG_EXPERT_COUNT_TASKS),
        models=tuple(
            args.models
            or (("lightgbm",) if smoke else DEFAULT_CHUNK_COUNT_MODELS)
        ),
        budgets=tuple(
            args.budgets
            or ((0.3,) if smoke else DEFAULT_BIG_EXPERT_COUNT_BUDGETS)
        ),
        chunk_counts=tuple(
            args.chunk_counts or ((2, 4) if smoke else DEFAULT_BIG_EXPERT_COUNTS)
        ),
        ensemble_methods=tuple(
            args.ensemble_methods or DEFAULT_CHUNK_COUNT_ENSEMBLES
        ),
        voting_pruning_modes=(
            (True, False)
            if args.voting_pruning == "both"
            else (args.voting_pruning == "on",)
        ),
        cluster_algorithms=tuple(
            args.cluster_algorithms or DEFAULT_CHUNK_COUNT_CLUSTER_ALGORITHMS
        ),
        max_train_rows=2_000 if smoke else args.max_train_rows,
        seed=args.seed,
        show_progress=not args.no_progress,
        output_dir=output_dir,
        resume_from=args.resume_from,
        resume_policy=args.resume_policy,
        synthetic_smoke=smoke,
    )


def _run_cli(args: argparse.Namespace) -> Path | None:
    config = _config_from_args(args)
    if args.print_plan:
        print(json.dumps(json_ready(_plan_summary(config)), indent=2))
        return None
    return RMTChunkCountAblationBigOrchestrator(config).run()


if __name__ == "__main__":
    result_path = _run_cli(_parse_args())
    if result_path is not None:
        print(result_path)
