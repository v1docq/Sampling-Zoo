from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
BENCHMARK_DIR = Path(__file__).resolve().parent
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

from benchmark_incremental import IncrementalExperimentSaver  # noqa: E402
from benchmark_logging import BenchmarkLogger  # noqa: E402
from rmt_experiment_utils import json_ready, markdown_table  # noqa: E402
from rmt_regression_medium_datasets import (  # noqa: E402
    DEFAULT_RMT_REGRESSION_TASKS,
    RMTRegressionExperimentConfig,
    RMTRegressionExperimentOrchestrator,
    make_rmt_experiment_strategy_configs,
)
from sampling_zoo.core.experiment.contracts import StrategyGridContract  # noqa: E402
from sampling_zoo.core.experiment.morphisms import normalize_strategy_grid  # noqa: E402
from sampling_zoo.core.experiment.resume import ResumePolicy  # noqa: E402


COMPARISON_STRATEGIES: tuple[str, ...] = (
    "rmt_contraction",
    "raw_feature_clustering",
)
COMPARISON_BUDGETS: tuple[float, ...] = (0.1, 0.5, 0.9)
COMPARISON_ENSEMBLE_METHODS: tuple[str, ...] = (
    "voting",
    "routed_weighted",
)
COMPARISON_ROUTER_MODES: tuple[str, ...] = ("spectral",)
FOUNDATION_MODELS: tuple[str, ...] = ("tabpfn", "tabicl")
CLUSTER_PROFILES: tuple[str, ...] = ("controlled", "paper")
DEFAULT_CONTROLLED_CLUSTER_ALGORITHMS: tuple[str, ...] = ("kmeans",)


@dataclass(frozen=True)
class EmbeddingSpaceAblationConfig(RMTRegressionExperimentConfig):
    """Matched design for the spectral-embedding versus raw-feature ablation."""

    strategies: Sequence[str] = COMPARISON_STRATEGIES
    models: Sequence[str] = FOUNDATION_MODELS
    ensemble_methods: Sequence[str] = COMPARISON_ENSEMBLE_METHODS
    budget_ratios: Sequence[float] = COMPARISON_BUDGETS
    router_modes: Sequence[str] = COMPARISON_ROUTER_MODES
    n_partitions: int = 2
    cluster_profile: str = "controlled"
    controlled_cluster_algorithms: Sequence[str] = (
        DEFAULT_CONTROLLED_CLUSTER_ALGORITHMS
    )
    include_direct_baseline: bool = False
    output_dir: str | Path | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if tuple(self.strategies) != COMPARISON_STRATEGIES:
            raise ValueError(
                "Embedding ablation requires exactly rmt_contraction and "
                "raw_feature_clustering."
            )
        if self.cluster_profile not in CLUSTER_PROFILES:
            raise ValueError(
                f"cluster_profile must be one of: {', '.join(CLUSTER_PROFILES)}"
            )
        if self.n_partitions < 2:
            raise ValueError("n_partitions must be at least 2")
        controlled_algorithms = tuple(self.controlled_cluster_algorithms)
        if not controlled_algorithms:
            raise ValueError(
                "controlled_cluster_algorithms must not be empty"
            )
        if len(set(controlled_algorithms)) != len(controlled_algorithms):
            raise ValueError(
                "controlled_cluster_algorithms must not contain duplicates"
            )
        budgets = tuple(float(value) for value in self.budget_ratios)
        if not budgets or any(value <= 0.0 or value > 1.0 for value in budgets):
            raise ValueError("budget_ratios must contain values in (0, 1]")
        if len(set(budgets)) != len(budgets):
            raise ValueError("budget_ratios must not contain duplicates")
        unsupported_ensembles = set(self.ensemble_methods) - set(
            COMPARISON_ENSEMBLE_METHODS
        )
        if unsupported_ensembles:
            raise ValueError(
                "Embedding ablation supports voting and routed_weighted only"
            )
        if (
            "routed_weighted" in self.ensemble_methods
            and tuple(self.router_modes) != COMPARISON_ROUTER_MODES
        ):
            raise ValueError(
                "Use router_modes=('spectral',) so the ablation does not add "
                "a learned-gating confound."
            )


def make_embedding_ablation_strategy_configs(
    config: EmbeddingSpaceAblationConfig,
) -> dict[str, dict[str, Any]]:
    """Build a strictly matched grid for both row representations."""

    configs = make_rmt_experiment_strategy_configs(
        problem_type="regression",
        strategies=config.strategies,
        ensemble_methods=config.ensemble_methods,
        budget_ratios=config.budget_ratios,
        view_strategies=config.view_strategies,
        n_partitions=config.n_partitions,
        seed=config.seed,
        show_progress=config.show_progress,
        router_modes=config.router_modes,
    )
    if not config.include_direct_baseline:
        configs.pop("full_dataset", None)

    for strategy_config in configs.values():
        if strategy_config.get("strategy") not in COMPARISON_STRATEGIES:
            continue
        strategy_config["ablation_axis"] = "row_representation"
        strategy_config["ablation_profile"] = config.cluster_profile
        if config.cluster_profile == "controlled":
            # Keep the number of chunks fixed through min=max, but route the
            # fit through the selector.  The sampler's ``fixed`` path is an
            # optimized KMeans-only path and therefore cannot compare GMM.
            # Both representations always receive the same candidate list.
            strategy_config.update(
                {
                    "n_partitions": int(config.n_partitions),
                    "partition_selection_method": "auto",
                    "cluster_algorithms": list(
                        config.controlled_cluster_algorithms
                    ),
                    "cluster_ensemble_method": "best_score",
                    "min_partitions": int(config.n_partitions),
                    "max_partitions": int(config.n_partitions),
                }
            )
    return configs


class EmbeddingAblationReportBuilder:
    """Persist matched RMT/raw-feature comparisons without budget selection."""

    REPRESENTATION_NAMES = {
        "rmt_contraction": "rmt_embedding",
        "raw_feature_clustering": "raw_features",
    }

    def build(
        self,
        run_records: Sequence[Mapping[str, Any]],
        output_dir: Path,
    ) -> dict[str, pd.DataFrame]:
        output_dir.mkdir(parents=True, exist_ok=True)
        runs = self._normalize_runs(run_records)
        runs.to_csv(output_dir / "embedding_ablation_runs.csv", index=False)
        pairs = self._build_pairs(runs)
        pairs.to_csv(output_dir / "embedding_ablation_matched_pairs.csv", index=False)
        summary = self._build_summary(pairs)
        summary.to_csv(output_dir / "embedding_ablation_summary.csv", index=False)
        budget_summary = self._build_budget_summary(pairs)
        budget_summary.to_csv(
            output_dir / "embedding_ablation_by_budget.csv",
            index=False,
        )
        dataset_summary = self._build_dataset_summary(pairs)
        dataset_summary.to_csv(
            output_dir / "embedding_ablation_by_dataset.csv",
            index=False,
        )
        self._write_markdown_report(
            runs,
            pairs,
            summary,
            dataset_summary,
            output_dir,
        )
        return {
            "runs": runs,
            "pairs": pairs,
            "summary": summary,
            "budget_summary": budget_summary,
            "dataset_summary": dataset_summary,
        }

    def _normalize_runs(
        self,
        run_records: Sequence[Mapping[str, Any]],
    ) -> pd.DataFrame:
        if not run_records:
            return pd.DataFrame(columns=self._run_columns())
        frame = pd.json_normalize(list(run_records), sep=".")

        def series(name: str, default: Any = None) -> pd.Series:
            if name in frame.columns:
                return frame[name]
            return pd.Series([default] * len(frame), index=frame.index)

        def algorithm_signature(value: Any) -> str | None:
            if isinstance(value, (list, tuple)):
                return "|".join(str(item) for item in value)
            return None if value is None else str(value)

        sampler = series("strategy_params.strategy")
        runs = pd.DataFrame(
            {
                "dataset": series("dataset"),
                "model": series("strategy_params.model"),
                "split_label": series("strategy_params.split_label", "split_1"),
                "seed": pd.to_numeric(series("extra.seed"), errors="coerce"),
                "sampler": sampler,
                "representation": sampler.map(self.REPRESENTATION_NAMES),
                "budget_ratio": pd.to_numeric(
                    series("strategy_params.budget_ratio"), errors="coerce"
                ),
                "ensemble_method": series("strategy_params.ensemble_method"),
                "router": series("strategy_params.router", "none").fillna("none"),
                "rmse": pd.to_numeric(series("model_metrics.rmse"), errors="coerce"),
                "fit_sec": pd.to_numeric(series("timings_sec.fit"), errors="coerce"),
                "inference_sec": pd.to_numeric(
                    series("timings_sec.inference"), errors="coerce"
                ),
                "partitioning_sec": pd.to_numeric(
                    series("extra.runtime_diagnostics.partitioning.total"),
                    errors="coerce",
                ),
                "selected_rows": pd.to_numeric(
                    series("extra.partition_size_contract.selected_rows"),
                    errors="coerce",
                ),
                "selected_n_partitions": pd.to_numeric(
                    series("extra.sampler_diagnostics.selected_n_partitions"),
                    errors="coerce",
                ),
                "selected_cluster_algorithm": series(
                    "extra.sampler_diagnostics.selected_cluster_algorithm"
                ),
                "configured_cluster_algorithms": series(
                    "strategy_params.cluster_algorithms"
                ).map(algorithm_signature),
                "evaluated_cluster_algorithms": series(
                    "extra.sampler_diagnostics.cluster_algorithms"
                ).map(algorithm_signature),
                "cluster_selection_metric": series(
                    "extra.sampler_diagnostics.cluster_selection_metric"
                ),
                "cluster_ensemble_method": series(
                    "extra.sampler_diagnostics.cluster_ensemble_method"
                ),
                "structure_cache": series("extra.cache_usage.structure"),
                "base_partition_cache": series(
                    "extra.cache_usage.base_partitions"
                ),
                "trained_model_cache": series("extra.cache_usage.trained_models"),
                "error": series("extra.error"),
            }
        )
        return runs[runs["sampler"].isin(self.REPRESENTATION_NAMES)].reset_index(
            drop=True
        )

    @staticmethod
    def _run_columns() -> list[str]:
        return [
            "dataset",
            "model",
            "split_label",
            "seed",
            "sampler",
            "representation",
            "budget_ratio",
            "ensemble_method",
            "router",
            "rmse",
            "fit_sec",
            "inference_sec",
            "partitioning_sec",
            "selected_rows",
            "selected_n_partitions",
            "selected_cluster_algorithm",
            "configured_cluster_algorithms",
            "evaluated_cluster_algorithms",
            "cluster_selection_metric",
            "cluster_ensemble_method",
            "structure_cache",
            "base_partition_cache",
            "trained_model_cache",
            "error",
        ]

    @staticmethod
    def _pair_keys() -> list[str]:
        return [
            "dataset",
            "model",
            "split_label",
            "seed",
            "budget_ratio",
            "ensemble_method",
            "router",
        ]

    def _build_pairs(self, runs: pd.DataFrame) -> pd.DataFrame:
        if runs.empty:
            return pd.DataFrame()
        valid = runs.dropna(subset=["rmse", "representation"])
        value_columns = [
            "rmse",
            "fit_sec",
            "inference_sec",
            "partitioning_sec",
            "selected_rows",
            "selected_n_partitions",
            "selected_cluster_algorithm",
        ]
        wide = valid.pivot_table(
            index=self._pair_keys(),
            columns="representation",
            values=value_columns,
            aggfunc="first",
        )
        wide.columns = [f"{metric}__{representation}" for metric, representation in wide.columns]
        pairs = wide.reset_index()
        rmt_col = "rmse__rmt_embedding"
        raw_col = "rmse__raw_features"
        if rmt_col not in pairs:
            pairs[rmt_col] = np.nan
        if raw_col not in pairs:
            pairs[raw_col] = np.nan
        selected_rmt = "selected_rows__rmt_embedding"
        selected_raw = "selected_rows__raw_features"
        if selected_rmt not in pairs:
            pairs[selected_rmt] = np.nan
        if selected_raw not in pairs:
            pairs[selected_raw] = np.nan
        pairs["representations_complete"] = pairs[[rmt_col, raw_col]].notna().all(
            axis=1
        )
        pairs["row_budget_match"] = (
            pairs[[selected_rmt, selected_raw]].notna().all(axis=1)
            & np.isclose(pairs[selected_rmt], pairs[selected_raw], rtol=0.0, atol=0.0)
        )
        pairs["selected_rows_raw_minus_rmt"] = pairs[selected_raw] - pairs[selected_rmt]
        pairs["pair_complete"] = (
            pairs["representations_complete"] & pairs["row_budget_match"]
        )
        pairs["rmse_raw_minus_rmt"] = pairs[raw_col] - pairs[rmt_col]
        pairs["rmt_improvement_vs_raw_pct"] = np.where(
            pairs[raw_col].abs() > 0,
            100.0 * pairs["rmse_raw_minus_rmt"] / pairs[raw_col].abs(),
            np.nan,
        )
        tolerance = 1e-12
        pairs["winner"] = np.select(
            [
                pairs["pair_complete"]
                & (pairs["rmse_raw_minus_rmt"] > tolerance),
                pairs["pair_complete"]
                & (pairs["rmse_raw_minus_rmt"] < -tolerance),
                pairs["pair_complete"],
            ],
            ["rmt_embedding", "raw_features", "tie"],
            default="incomplete",
        )
        return pairs.sort_values(self._pair_keys()).reset_index(drop=True)

    def _build_summary(self, pairs: pd.DataFrame) -> pd.DataFrame:
        if "pair_complete" not in pairs:
            return pd.DataFrame()
        complete = pairs[pairs["pair_complete"]].copy()
        if complete.empty:
            return pd.DataFrame()
        group_keys = ["model", "ensemble_method", "router"]
        summary = (
            complete.groupby(group_keys, dropna=False)
            .agg(
                matched_pairs=("pair_complete", "size"),
                mean_rmt_improvement_pct=("rmt_improvement_vs_raw_pct", "mean"),
                median_rmt_improvement_pct=("rmt_improvement_vs_raw_pct", "median"),
                mean_rmse_raw_minus_rmt=("rmse_raw_minus_rmt", "mean"),
                mean_rmt_fit_sec=("fit_sec__rmt_embedding", "mean"),
                mean_raw_fit_sec=("fit_sec__raw_features", "mean"),
            )
            .reset_index()
        )
        wins = (
            complete.assign(count=1)
            .pivot_table(
                index=group_keys,
                columns="winner",
                values="count",
                aggfunc="sum",
                fill_value=0,
            )
            .reset_index()
        )
        for name in ("rmt_embedding", "raw_features", "tie"):
            if name not in wins:
                wins[name] = 0
        wins = wins.rename(
            columns={
                "rmt_embedding": "rmt_wins",
                "raw_features": "raw_wins",
                "tie": "ties",
            }
        )
        return summary.merge(wins, on=group_keys, how="left").sort_values(group_keys)

    def _build_budget_summary(self, pairs: pd.DataFrame) -> pd.DataFrame:
        if "pair_complete" not in pairs:
            return pd.DataFrame()
        complete = pairs[pairs["pair_complete"]].copy()
        if complete.empty:
            return pd.DataFrame()
        return (
            complete.groupby(
                ["model", "ensemble_method", "router", "budget_ratio"],
                dropna=False,
            )
            .agg(
                matched_pairs=("pair_complete", "size"),
                mean_rmt_improvement_pct=("rmt_improvement_vs_raw_pct", "mean"),
                median_rmt_improvement_pct=("rmt_improvement_vs_raw_pct", "median"),
                rmt_wins=("winner", lambda values: int((values == "rmt_embedding").sum())),
                raw_wins=("winner", lambda values: int((values == "raw_features").sum())),
                ties=("winner", lambda values: int((values == "tie").sum())),
            )
            .reset_index()
            .sort_values(["model", "ensemble_method", "router", "budget_ratio"])
        )

    def _build_dataset_summary(self, pairs: pd.DataFrame) -> pd.DataFrame:
        if "pair_complete" not in pairs:
            return pd.DataFrame()
        complete = pairs[pairs["pair_complete"]].copy()
        if complete.empty:
            return pd.DataFrame()
        return (
            complete.groupby(
                ["dataset", "model", "ensemble_method", "router"],
                dropna=False,
            )
            .agg(
                matched_budgets=("pair_complete", "size"),
                mean_rmt_improvement_pct=("rmt_improvement_vs_raw_pct", "mean"),
                median_rmt_improvement_pct=("rmt_improvement_vs_raw_pct", "median"),
                rmt_wins=("winner", lambda values: int((values == "rmt_embedding").sum())),
                raw_wins=("winner", lambda values: int((values == "raw_features").sum())),
                ties=("winner", lambda values: int((values == "tie").sum())),
            )
            .reset_index()
            .sort_values(["dataset", "model", "ensemble_method", "router"])
        )

    @staticmethod
    def _write_markdown_report(
        runs: pd.DataFrame,
        pairs: pd.DataFrame,
        summary: pd.DataFrame,
        dataset_summary: pd.DataFrame,
        output_dir: Path,
    ) -> None:
        complete_count = int(pairs.get("pair_complete", pd.Series(dtype=bool)).sum())
        lines = [
            "# Embedding-space ablation",
            "",
            "Positive `rmt_improvement_vs_raw_pct` means that clustering/routing in the RMT embedding achieved lower RMSE than the raw-feature counterpart at the same budget and aggregation method.",
            "",
            f"- Raw ablation runs: {len(runs)}",
            f"- Matched complete pairs: {complete_count}",
            f"- Incomplete pairs: {int(len(pairs) - complete_count)}",
            "",
            "## Aggregate matched comparison",
            "",
        ]
        if summary.empty:
            lines.append("No complete RMT/raw-feature pairs are available yet.")
        else:
            lines.append(markdown_table(summary))
        lines.extend(["", "## Matched comparison by dataset", ""])
        if dataset_summary.empty:
            lines.append("No complete per-dataset comparisons are available yet.")
        else:
            lines.append(markdown_table(dataset_summary))
        incomplete = pairs[~pairs.get("pair_complete", False)] if not pairs.empty else pairs
        if not incomplete.empty:
            lines.extend(
                [
                    "",
                    "## Incomplete pairs",
                    "",
                    markdown_table(
                        incomplete,
                        [
                            *EmbeddingAblationReportBuilder._pair_keys(),
                            "representations_complete",
                            "row_budget_match",
                            "selected_rows_raw_minus_rmt",
                        ],
                    ),
                ]
            )
        (output_dir / "embedding_ablation_report.md").write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )


class EmbeddingSpaceAblationMediumOrchestrator(
    RMTRegressionExperimentOrchestrator
):
    def __init__(self, config: EmbeddingSpaceAblationConfig) -> None:
        super().__init__(config)
        self.config = config
        self.embedding_report_builder = EmbeddingAblationReportBuilder()

    def _run_id_prefix(self) -> str:
        return "run_embedding_space_ablation_medium"

    def _create_logger(self) -> BenchmarkLogger:
        if self.config.resume_from is not None:
            return super()._create_logger()
        if self.config.output_dir is not None:
            output_dir = Path(self.config.output_dir).expanduser()
            if not output_dir.is_absolute():
                output_dir = (ROOT_DIR / output_dir).resolve()
            if output_dir.exists() and (
                not output_dir.is_dir() or any(output_dir.iterdir())
            ):
                raise FileExistsError(
                    f"Output directory is not empty: {output_dir}. Use a new "
                    "--output-dir, or continue the existing run with "
                    "--resume-from."
                )
            return BenchmarkLogger(
                run_id=output_dir.name,
                artifacts_root=output_dir.parent,
            )
        run_id = f"{self._run_id_prefix()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        return BenchmarkLogger(run_id=run_id, artifacts_root=BENCHMARK_DIR / "results")

    def _build_strategy_grid(self) -> StrategyGridContract:
        return normalize_strategy_grid(
            make_embedding_ablation_strategy_configs(self.config)
        )

    def _create_incremental_saver(
        self,
        logger: BenchmarkLogger,
    ) -> IncrementalExperimentSaver:
        saver = super()._create_incremental_saver(logger)
        saver.records_path = logger.paths.metrics / "embedding_ablation_runs.jsonl"

        def _build_embedding_tables(
            records: Sequence[Mapping[str, Any]],
        ) -> None:
            self.embedding_report_builder.build(records, logger.paths.metrics)

        saver.snapshot_hooks = (*saver.snapshot_hooks, _build_embedding_tables)
        return saver

    def _build_report_artifacts(
        self,
        run_records: Sequence[Mapping[str, Any]],
        logger: BenchmarkLogger,
    ) -> None:
        super()._build_report_artifacts(run_records, logger)
        comparison_tables = self.embedding_report_builder.build(
            run_records,
            logger.paths.metrics,
        )
        (logger.paths.root / "embedding_ablation_protocol.json").write_text(
            json.dumps(
                json_ready(self._comparison_protocol()),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        self._validate_completed_comparison(comparison_tables)

    @staticmethod
    def _validate_completed_comparison(
        comparison_tables: Mapping[str, pd.DataFrame],
    ) -> None:
        runs = comparison_tables["runs"]
        pairs = comparison_tables["pairs"]
        pair_keys = EmbeddingAblationReportBuilder._pair_keys()
        expected_pairs = runs[pair_keys].drop_duplicates()
        duplicate_sides = int(
            runs.duplicated([*pair_keys, "representation"]).sum()
        )
        failed_sides = int(
            runs["rmse"].isna().sum()
            + runs["error"].notna().sum()
        )
        complete_pairs = int(
            pairs.get("pair_complete", pd.Series(dtype=bool)).sum()
        )
        expected_count = int(len(expected_pairs))
        if (
            duplicate_sides
            or failed_sides
            or len(pairs) != expected_count
            or complete_pairs != expected_count
        ):
            raise RuntimeError(
                "Embedding ablation integrity check failed: "
                f"expected_pairs={expected_count}, reported_pairs={len(pairs)}, "
                f"complete_pairs={complete_pairs}, "
                f"duplicate_representation_sides={duplicate_sides}, "
                f"failed_representation_sides={failed_sides}. See "
                "metrics/embedding_ablation_matched_pairs.csv."
            )

    def _build_run_meta(
        self,
        logger: BenchmarkLogger,
        run_records: Sequence[Mapping[str, Any]],
        status: str = "completed",
    ) -> dict[str, Any]:
        return {
            **super()._build_run_meta(logger, run_records, status),
            "embedding_ablation": self._comparison_protocol(),
        }

    def _comparison_protocol(self) -> dict[str, Any]:
        chunk_methods_per_budget = sum(
            1
            for ensemble in self.config.ensemble_methods
            for _router in (
                self.config.router_modes
                if ensemble == "routed_weighted"
                else (None,)
            )
        )
        leaf_runs_per_dataset_model = (
            len(self.config.strategies)
            * len(self.config.budget_ratios)
            * chunk_methods_per_budget
            + (1 if self.config.include_direct_baseline else 0)
        )
        return {
            "question": "Does RMT/SVD row representation improve matched-budget chunking over standardized raw features?",
            "profile": self.config.cluster_profile,
            "controlled_factors": {
                "n_partitions": self.config.n_partitions,
                "clusterer": (
                    {
                        "algorithms": list(
                            self.config.controlled_cluster_algorithms
                        ),
                        "selection": (
                            "fixed"
                            if len(self.config.controlled_cluster_algorithms)
                            == 1
                            else "best_score"
                        ),
                    }
                    if self.config.cluster_profile == "controlled"
                    else "paper_auto_selection"
                ),
                "budgets": list(self.config.budget_ratios),
                "ensemble_methods": list(self.config.ensemble_methods),
                "router_modes": list(self.config.router_modes),
                "seed": self.config.seed,
            },
            "representations": {
                "rmt_contraction": "random-view SVD sample embedding",
                "raw_feature_clustering": "standardized encoded raw features",
            },
            "include_direct_baseline": self.config.include_direct_baseline,
            "leaf_runs_per_dataset_model": leaf_runs_per_dataset_model,
            "expensive_structure_fits_per_dataset_split": len(
                self.config.strategies
            ),
            "unique_trained_chunk_sets_per_dataset_model": (
                len(self.config.strategies) * len(self.config.budget_ratios)
            ),
            "cache_expectation": (
                "structure is reused across budgets/models; trained chunk models "
                "are reused across voting and routed aggregation"
            ),
        }


def run_embedding_space_ablation_medium(
    *,
    regression_tasks: Sequence[str] | None = None,
    models: Sequence[str] = FOUNDATION_MODELS,
    budget_ratios: Sequence[float] = COMPARISON_BUDGETS,
    cluster_profile: str = "controlled",
    n_partitions: int = 2,
    max_train_rows: int | None = 300_000,
    seed: int = 42,
    show_progress: bool = True,
    include_direct_baseline: bool = False,
    output_dir: str | Path | None = None,
    resume_from: str | Path | None = None,
    resume_policy: str = ResumePolicy.RETRY_FAILED.value,
    synthetic_smoke: bool = False,
) -> Path:
    config = EmbeddingSpaceAblationConfig(
        regression_tasks=regression_tasks or DEFAULT_RMT_REGRESSION_TASKS,
        models=models,
        budget_ratios=budget_ratios,
        cluster_profile=cluster_profile,
        n_partitions=n_partitions,
        max_train_rows=max_train_rows,
        seed=seed,
        show_progress=show_progress,
        include_direct_baseline=include_direct_baseline,
        output_dir=output_dir,
        resume_from=resume_from,
        resume_policy=resume_policy,
        synthetic_smoke=synthetic_smoke,
    )
    return EmbeddingSpaceAblationMediumOrchestrator(config).run()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Matched medium-dataset ablation: RMT/SVD embedding versus raw "
            "standardized feature clustering."
        )
    )
    parser.add_argument("--task", action="append", dest="tasks")
    parser.add_argument("--model", action="append", dest="models")
    parser.add_argument("--budget", action="append", type=float, dest="budgets")
    parser.add_argument("--cluster-profile", choices=CLUSTER_PROFILES, default="controlled")
    parser.add_argument("--n-partitions", type=int, default=2)
    parser.add_argument("--max-train-rows", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-direct-baseline", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--resume-policy", default=ResumePolicy.RETRY_FAILED.value)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def _run_cli(args: argparse.Namespace) -> Path:
    smoke = bool(args.smoke)
    output_dir = args.output_dir
    if output_dir is None and args.resume_from is not None:
        # ``output_dir`` participates in the stored experiment configuration.
        # Reusing the resume path preserves the configuration hash for runs
        # started with an explicit output directory.
        output_dir = args.resume_from
    return run_embedding_space_ablation_medium(
        regression_tasks=args.tasks,
        models=tuple(args.models or (("lightgbm",) if smoke else FOUNDATION_MODELS)),
        budget_ratios=tuple(args.budgets or ((0.5,) if smoke else COMPARISON_BUDGETS)),
        cluster_profile=args.cluster_profile,
        n_partitions=args.n_partitions,
        max_train_rows=2_000 if smoke else args.max_train_rows,
        seed=args.seed,
        show_progress=not args.no_progress,
        include_direct_baseline=args.include_direct_baseline,
        output_dir=output_dir,
        resume_from=args.resume_from,
        resume_policy=args.resume_policy,
        synthetic_smoke=smoke,
    )


if __name__ == "__main__":
    result_path = _run_cli(_parse_args())
    print(result_path)
