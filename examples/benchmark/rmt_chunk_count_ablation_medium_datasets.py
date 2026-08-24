from __future__ import annotations

import argparse
import json
import math
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


DEFAULT_CHUNK_COUNTS: tuple[int, ...] = (2, 4, 8, 16)
DEFAULT_CHUNK_COUNT_BUDGETS: tuple[float, ...] = (0.1, 0.3)
DEFAULT_CHUNK_COUNT_MODELS: tuple[str, ...] = ("tabpfn", "tabicl")
DEFAULT_CHUNK_COUNT_ENSEMBLES: tuple[str, ...] = (
    "voting",
    "routed_weighted",
)
DEFAULT_CHUNK_COUNT_CLUSTER_ALGORITHMS: tuple[str, ...] = ("kmeans", "gmm")
DEFAULT_VOTING_PRUNING_MODES: tuple[bool, ...] = (True,)
QUICK_FOUNDATION_TASKS: tuple[str, ...] = (
    "Brazilian_houses",
    "pol",
    "elevators",
    "house_16H",
    "house_sales",
    "OnlineNewsPopularity",
    "diamonds",
)
QUICK_FOUNDATION_BUDGETS: tuple[float, ...] = (0.3,)
QUICK_FOUNDATION_CHUNK_COUNTS: tuple[int, ...] = (2, 4)
SUPPORTED_CHUNK_COUNT_CLUSTER_ALGORITHMS = frozenset({"kmeans", "gmm"})
REFERENCE_CHUNK_COUNT = 2


@dataclass(frozen=True)
class RMTChunkCountAblationConfig(RMTRegressionExperimentConfig):
    """Matched-total-budget ablation over the number of RMT experts."""

    strategies: Sequence[str] = ("rmt_contraction",)
    models: Sequence[str] = DEFAULT_CHUNK_COUNT_MODELS
    ensemble_methods: Sequence[str] = DEFAULT_CHUNK_COUNT_ENSEMBLES
    budget_ratios: Sequence[float] = DEFAULT_CHUNK_COUNT_BUDGETS
    router_modes: Sequence[str] = ("spectral",)
    chunk_counts: Sequence[int] = DEFAULT_CHUNK_COUNTS
    cluster_algorithms: Sequence[str] = DEFAULT_CHUNK_COUNT_CLUSTER_ALGORITHMS
    voting_pruning_modes: Sequence[bool] = DEFAULT_VOTING_PRUNING_MODES
    include_direct_baseline: bool = False
    output_dir: str | Path | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if tuple(self.strategies) != ("rmt_contraction",):
            raise ValueError("Chunk-count ablation supports rmt_contraction only")

        chunk_counts = tuple(int(value) for value in self.chunk_counts)
        if not chunk_counts or any(value < 2 for value in chunk_counts):
            raise ValueError("chunk_counts must contain integers >= 2")
        if len(set(chunk_counts)) != len(chunk_counts):
            raise ValueError("chunk_counts must not contain duplicates")
        if REFERENCE_CHUNK_COUNT not in chunk_counts:
            raise ValueError("chunk_counts must include the k=2 reference arm")

        budgets = tuple(float(value) for value in self.budget_ratios)
        if not budgets or any(
            not math.isfinite(value) or not 0.0 < value <= 1.0
            for value in budgets
        ):
            raise ValueError("budget_ratios must contain finite values in (0, 1]")
        if len(set(budgets)) != len(budgets):
            raise ValueError("budget_ratios must not contain duplicates")

        algorithms = tuple(str(value).lower() for value in self.cluster_algorithms)
        if not algorithms or len(set(algorithms)) != len(algorithms):
            raise ValueError("cluster_algorithms must be non-empty and unique")
        unsupported_algorithms = set(algorithms) - SUPPORTED_CHUNK_COUNT_CLUSTER_ALGORITHMS
        if unsupported_algorithms:
            raise ValueError(
                "Chunk-count ablation supports only kmeans and gmm; got "
                f"{sorted(unsupported_algorithms)}"
            )

        unsupported_ensembles = set(self.ensemble_methods) - set(
            DEFAULT_CHUNK_COUNT_ENSEMBLES
        )
        if unsupported_ensembles:
            raise ValueError("Use voting and/or routed_weighted")
        if (
            "routed_weighted" in self.ensemble_methods
            and tuple(self.router_modes) != ("spectral",)
        ):
            raise ValueError(
                "Use router_modes=('spectral',) to avoid a learned-gating confound"
            )

        pruning_modes = tuple(bool(value) for value in self.voting_pruning_modes)
        if not pruning_modes or len(set(pruning_modes)) != len(pruning_modes):
            raise ValueError("voting_pruning_modes must be non-empty and unique")

        object.__setattr__(self, "chunk_counts", chunk_counts)
        object.__setattr__(self, "budget_ratios", budgets)
        object.__setattr__(self, "cluster_algorithms", algorithms)
        object.__setattr__(self, "voting_pruning_modes", pruning_modes)


def make_chunk_count_ablation_strategy_configs(
    config: RMTChunkCountAblationConfig,
) -> dict[str, dict[str, Any]]:
    """Build fixed-k arms while selecting K-means versus GMM at every k."""

    configs: dict[str, dict[str, Any]] = {}
    if config.include_direct_baseline:
        configs["full_dataset"] = {
            "strategy": "full_dataset",
            "force_direct_model": True,
            "ensemble_method": "full_dataset",
            "budget_ratio": 1.0,
        }

    for chunk_count in config.chunk_counts:
        fixed_k_configs = make_rmt_experiment_strategy_configs(
            problem_type="regression",
            strategies=config.strategies,
            ensemble_methods=config.ensemble_methods,
            budget_ratios=config.budget_ratios,
            view_strategies=config.view_strategies,
            n_partitions=chunk_count,
            seed=config.seed,
            show_progress=config.show_progress,
            router_modes=config.router_modes,
        )
        fixed_k_configs.pop("full_dataset", None)
        for base_name, strategy_config in fixed_k_configs.items():
            # min=max forces the requested k, while the auto selector still
            # compares K-means and GMM by the same balanced-silhouette score.
            strategy_config.update(
                {
                    "ablation_axis": "n_partitions",
                    "ablation_profile": "fixed_k_matched_total_budget",
                    "n_partitions": int(chunk_count),
                    "partition_selection_method": "auto",
                    "cluster_algorithms": list(config.cluster_algorithms),
                    "cluster_ensemble_method": "best_score",
                    "min_partitions": int(chunk_count),
                    "max_partitions": int(chunk_count),
                    # The benchmark runner historically caps an ensemble at
                    # ten chunks.  This ablation must actually retain every
                    # requested expert, including the k=16 arm.
                    "chunks_percent": 100.0,
                }
            )
            strategy_name = f"{base_name}__k_{chunk_count:02d}"
            if strategy_config["ensemble_method"] == "voting":
                for pruning_enabled in config.voting_pruning_modes:
                    if pruning_enabled:
                        # Keep the historical run identity intact so old
                        # pruning-enabled outputs remain resumable.
                        configs[strategy_name] = dict(strategy_config)
                    else:
                        configs[f"{strategy_name}__pruning_off"] = {
                            **strategy_config,
                            "validation_pruning": False,
                        }
            else:
                configs[strategy_name] = strategy_config
    return configs


class ChunkCountAblationReportBuilder:
    """Create downstream-k curves and compare them with silhouette selection."""

    def build(
        self,
        run_records: Sequence[Mapping[str, Any]],
        output_dir: Path,
    ) -> dict[str, pd.DataFrame]:
        output_dir.mkdir(parents=True, exist_ok=True)
        runs = self._normalize_runs(run_records)
        comparisons = self._compare_with_reference(runs)
        pruning_effect = self._build_pruning_effect(runs)
        best_k = self._build_best_k(comparisons)
        summary = self._build_summary(comparisons)
        algorithm_summary = self._build_algorithm_summary(runs)

        tables = {
            "runs": runs,
            "comparisons": comparisons,
            "pruning_effect": pruning_effect,
            "best_k": best_k,
            "summary": summary,
            "algorithm_summary": algorithm_summary,
        }
        filenames = {
            "runs": "chunk_count_runs.csv",
            "comparisons": "chunk_count_vs_k2.csv",
            "pruning_effect": "chunk_count_pruning_effect.csv",
            "best_k": "chunk_count_best_k.csv",
            "summary": "chunk_count_summary.csv",
            "algorithm_summary": "chunk_count_cluster_algorithms.csv",
        }
        for name, table in tables.items():
            table.to_csv(output_dir / filenames[name], index=False)
        self._write_markdown_report(tables, output_dir)
        return tables

    @staticmethod
    def _series(frame: pd.DataFrame, name: str, default: Any = None) -> pd.Series:
        if name in frame.columns:
            return frame[name]
        return pd.Series([default] * len(frame), index=frame.index)

    def _normalize_runs(
        self,
        run_records: Sequence[Mapping[str, Any]],
    ) -> pd.DataFrame:
        if not run_records:
            return pd.DataFrame(columns=self._run_columns())
        frame = pd.json_normalize(list(run_records), sep=".")
        series = lambda name, default=None: self._series(frame, name, default)
        post_budget_prefix = (
            "extra.sampler_diagnostics.post_budget_chunk_sizes."
        )
        post_budget_columns = [
            column
            for column in frame.columns
            if column.startswith(post_budget_prefix)
        ]
        if post_budget_columns:
            post_budget_sizes = frame[post_budget_columns].apply(
                pd.to_numeric, errors="coerce"
            )
            min_expert_train_rows = post_budget_sizes.min(axis=1, skipna=True)
            max_expert_train_rows = post_budget_sizes.max(axis=1, skipna=True)
        else:
            min_expert_train_rows = pd.Series(np.nan, index=frame.index)
            max_expert_train_rows = pd.Series(np.nan, index=frame.index)
        cluster_counts = series(
            "extra.sampler_diagnostics."
            "partition_selection_selected_candidate.components.counts"
        )
        min_source_cluster_rows = cluster_counts.map(
            lambda value: (
                min(value)
                if isinstance(value, (list, tuple)) and value
                else np.nan
            )
        )
        runs = pd.DataFrame(
            {
                "dataset": series("dataset"),
                "model": series("strategy_params.model"),
                "split_label": series("strategy_params.split_label", "split_1"),
                "seed": pd.to_numeric(series("extra.seed"), errors="coerce"),
                "sampler": series("strategy_params.strategy"),
                "budget_ratio": pd.to_numeric(
                    series("strategy_params.budget_ratio"), errors="coerce"
                ),
                "ensemble_method": series("strategy_params.ensemble_method"),
                "voting_pruning": pd.Series(
                    np.where(
                        series("strategy_params.ensemble_method").eq("voting"),
                        series("strategy_params.validation_pruning", True).map(
                            lambda value: "enabled" if bool(value) else "disabled"
                        ),
                        "not_applicable",
                    ),
                    index=frame.index,
                ),
                "router": series("strategy_params.router", "none").fillna("none"),
                "requested_k": pd.to_numeric(
                    series("strategy_params.n_partitions"), errors="coerce"
                ),
                "realized_k": pd.to_numeric(
                    series("extra.sampler_diagnostics.selected_n_partitions"),
                    errors="coerce",
                ),
                "selected_partition_count": pd.to_numeric(
                    series("sample_stats.selected_partition_count"), errors="coerce"
                ),
                "active_expert_count": pd.to_numeric(
                    series("sample_stats.active_model_count"), errors="coerce"
                ),
                "selected_rows": pd.to_numeric(
                    series("sample_stats.selected_rows"), errors="coerce"
                ),
                "model_fit_rows_total": pd.to_numeric(
                    series("sample_stats.model_fit_rows_total"), errors="coerce"
                ),
                "active_model_rows": pd.to_numeric(
                    series("sample_stats.active_model_rows"), errors="coerce"
                ),
                "min_expert_train_rows": min_expert_train_rows,
                "max_expert_train_rows": max_expert_train_rows,
                "min_source_cluster_rows": min_source_cluster_rows,
                "rmse": pd.to_numeric(
                    series("model_metrics.rmse"), errors="coerce"
                ),
                "fit_sec": pd.to_numeric(
                    series("timings_sec.fit"), errors="coerce"
                ),
                "inference_sec": pd.to_numeric(
                    series("timings_sec.inference"), errors="coerce"
                ),
                "partitioning_sec": pd.to_numeric(
                    series("extra.runtime_diagnostics.partitioning.total"),
                    errors="coerce",
                ),
                "selected_cluster_algorithm": series(
                    "extra.sampler_diagnostics.selected_cluster_algorithm"
                ),
                "cluster_selection_score": pd.to_numeric(
                    series(
                        "extra.sampler_diagnostics."
                        "partition_selection_selected_candidate.score"
                    ),
                    errors="coerce",
                ),
                "cluster_candidate_valid": series(
                    "extra.sampler_diagnostics."
                    "partition_selection_selected_candidate.valid"
                ),
                "cluster_imbalance_ratio": pd.to_numeric(
                    series(
                        "extra.sampler_diagnostics."
                        "partition_selection_selected_candidate.components."
                        "imbalance_ratio"
                    ),
                    errors="coerce",
                ),
                "cluster_min_fraction": pd.to_numeric(
                    series(
                        "extra.sampler_diagnostics."
                        "partition_selection_selected_candidate.components."
                        "min_cluster_fraction"
                    ),
                    errors="coerce",
                ),
                "cluster_constraint_violations": series(
                    "extra.sampler_diagnostics."
                    "partition_selection_selected_candidate.components."
                    "constraint_violations"
                ).map(
                    lambda value: (
                        "|".join(str(item) for item in value)
                        if isinstance(value, (list, tuple))
                        else value
                    )
                ),
                "structure_cache": series("extra.cache_usage.structure"),
                "base_partition_cache": series(
                    "extra.cache_usage.base_partitions"
                ),
                "trained_model_cache": series("extra.cache_usage.trained_models"),
                "local_calibration_status": series(
                    "extra.validation_diagnostics.local_calibration.status",
                    "not_applicable",
                ),
                "local_calibration_baseline_metric": pd.to_numeric(
                    series(
                        "extra.validation_diagnostics.local_calibration."
                        "baseline_metric"
                    ),
                    errors="coerce",
                ),
                "local_calibration_candidate_metric": pd.to_numeric(
                    series(
                        "extra.validation_diagnostics.local_calibration."
                        "calibrated_metric"
                    ),
                    errors="coerce",
                ),
                "validation_selection_policy": series(
                    "extra.validation_diagnostics.selection_policy"
                ),
                "validation_rmse_before_pruning": pd.to_numeric(
                    series(
                        "extra.validation_diagnostics."
                        "full_ensemble_metrics_before_pruning.rmse"
                    ),
                    errors="coerce",
                ),
                "validation_rmse_after_pruning": pd.to_numeric(
                    series(
                        "extra.validation_diagnostics."
                        "ensemble_metrics_after_pruning.rmse"
                    ),
                    errors="coerce",
                ),
                "error": series("extra.error"),
            }
        )
        runs = runs[runs["sampler"] == "rmt_contraction"].copy()
        runs["active_fraction_of_requested"] = (
            runs["active_expert_count"] / runs["requested_k"]
        )
        runs["foundation_min_rows_compatible"] = (
            runs["min_expert_train_rows"].notna()
            & runs["min_expert_train_rows"].ge(2)
        )
        return runs.sort_values(self._coordinate_keys()).reset_index(drop=True)

    @staticmethod
    def _run_columns() -> list[str]:
        return [
            "dataset",
            "model",
            "split_label",
            "seed",
            "sampler",
            "budget_ratio",
            "ensemble_method",
            "voting_pruning",
            "router",
            "requested_k",
            "realized_k",
            "selected_partition_count",
            "active_expert_count",
            "selected_rows",
            "model_fit_rows_total",
            "active_model_rows",
            "min_expert_train_rows",
            "max_expert_train_rows",
            "min_source_cluster_rows",
            "rmse",
            "fit_sec",
            "inference_sec",
            "partitioning_sec",
            "selected_cluster_algorithm",
            "cluster_selection_score",
            "cluster_candidate_valid",
            "cluster_imbalance_ratio",
            "cluster_min_fraction",
            "cluster_constraint_violations",
            "structure_cache",
            "base_partition_cache",
            "trained_model_cache",
            "local_calibration_status",
            "local_calibration_baseline_metric",
            "local_calibration_candidate_metric",
            "validation_selection_policy",
            "validation_rmse_before_pruning",
            "validation_rmse_after_pruning",
            "error",
            "active_fraction_of_requested",
            "foundation_min_rows_compatible",
        ]

    @staticmethod
    def _cell_keys() -> list[str]:
        return [
            "dataset",
            "model",
            "split_label",
            "seed",
            "budget_ratio",
            "ensemble_method",
            "voting_pruning",
            "router",
        ]

    @classmethod
    def _coordinate_keys(cls) -> list[str]:
        return [*cls._cell_keys(), "requested_k"]

    def _compare_with_reference(self, runs: pd.DataFrame) -> pd.DataFrame:
        if runs.empty:
            return pd.DataFrame()
        reference = runs[runs["requested_k"] == REFERENCE_CHUNK_COUNT][
            [
                *self._cell_keys(),
                "rmse",
                "selected_rows",
                "active_expert_count",
            ]
        ].rename(
            columns={
                "rmse": "rmse_k2",
                "selected_rows": "selected_rows_k2",
                "active_expert_count": "active_expert_count_k2",
            }
        )
        compared = runs.merge(reference, on=self._cell_keys(), how="left")
        compared["reference_complete"] = compared[["rmse", "rmse_k2"]].notna().all(
            axis=1
        )
        compared["row_budget_matches_k2"] = (
            compared[["selected_rows", "selected_rows_k2"]].notna().all(axis=1)
            & np.isclose(
                compared["selected_rows"],
                compared["selected_rows_k2"],
                rtol=0.0,
                atol=0.0,
            )
        )
        compared["rmse_change_vs_k2_pct"] = np.where(
            compared["rmse_k2"].abs() > 0,
            100.0 * (compared["rmse"] - compared["rmse_k2"])
            / compared["rmse_k2"].abs(),
            np.nan,
        )
        tolerance = 1e-12
        compared["outcome_vs_k2"] = np.select(
            [
                compared["reference_complete"]
                & compared["row_budget_matches_k2"]
                & (compared["rmse_change_vs_k2_pct"] < -tolerance),
                compared["reference_complete"]
                & compared["row_budget_matches_k2"]
                & (compared["rmse_change_vs_k2_pct"] > tolerance),
                compared["reference_complete"]
                & compared["row_budget_matches_k2"],
            ],
            ["better", "worse", "tie"],
            default="incomplete",
        )
        return compared.sort_values(self._coordinate_keys()).reset_index(drop=True)

    def _build_pruning_effect(self, runs: pd.DataFrame) -> pd.DataFrame:
        if runs.empty:
            return pd.DataFrame()
        voting = runs[runs["ensemble_method"].eq("voting")]
        pair_keys = [
            "dataset",
            "model",
            "split_label",
            "seed",
            "budget_ratio",
            "requested_k",
        ]
        enabled = voting[voting["voting_pruning"].eq("enabled")][
            [
                *pair_keys,
                "rmse",
                "active_expert_count",
                "validation_rmse_before_pruning",
                "validation_rmse_after_pruning",
            ]
        ].rename(
            columns={
                "rmse": "pruned_rmse",
                "active_expert_count": "pruned_active_experts",
                "validation_rmse_before_pruning": "validation_rmse_all_experts",
                "validation_rmse_after_pruning": "validation_rmse_pruned",
            }
        )
        disabled = voting[voting["voting_pruning"].eq("disabled")][
            [*pair_keys, "rmse", "active_expert_count"]
        ].rename(
            columns={
                "rmse": "all_experts_rmse",
                "active_expert_count": "all_experts_active_count",
            }
        )
        paired = enabled.merge(disabled, on=pair_keys, how="outer")
        if paired.empty:
            return paired
        paired["pair_complete"] = paired[
            ["pruned_rmse", "all_experts_rmse"]
        ].notna().all(axis=1)
        paired["pruning_rmse_change_vs_all_experts_pct"] = np.where(
            paired["all_experts_rmse"].abs() > 0,
            100.0
            * (paired["pruned_rmse"] - paired["all_experts_rmse"])
            / paired["all_experts_rmse"].abs(),
            np.nan,
        )
        paired["experts_removed"] = (
            paired["all_experts_active_count"]
            - paired["pruned_active_experts"]
        )
        return paired.sort_values(pair_keys).reset_index(drop=True)

    def _build_best_k(self, comparisons: pd.DataFrame) -> pd.DataFrame:
        if comparisons.empty:
            return pd.DataFrame()
        rows: list[dict[str, Any]] = []
        for coordinate, group in comparisons.groupby(self._cell_keys(), dropna=False):
            valid = group[
                group["rmse"].notna() & group["row_budget_matches_k2"]
            ]
            if valid.empty:
                continue
            best = valid.loc[valid["rmse"].idxmin()]
            valid_geometry = valid[valid["cluster_candidate_valid"].eq(True)]
            best_valid_geometry = (
                None
                if valid_geometry.empty
                else valid_geometry.loc[valid_geometry["rmse"].idxmin()]
            )
            foundation_compatible = valid[
                valid["foundation_min_rows_compatible"].eq(True)
            ]
            best_foundation_compatible = (
                None
                if foundation_compatible.empty
                else foundation_compatible.loc[
                    foundation_compatible["rmse"].idxmin()
                ]
            )
            scored = valid[valid["cluster_selection_score"].notna()]
            silhouette_best = (
                None
                if scored.empty
                else scored.loc[scored["cluster_selection_score"].idxmax()]
            )
            row = dict(zip(self._cell_keys(), coordinate))
            row.update(
                {
                    "best_downstream_k": int(best["requested_k"]),
                    "best_rmse": float(best["rmse"]),
                    "k2_rmse": float(best["rmse_k2"]),
                    "best_improvement_vs_k2_pct": float(
                        -best["rmse_change_vs_k2_pct"]
                    ),
                    "best_active_expert_count": int(best["active_expert_count"]),
                    "best_cluster_algorithm": best[
                        "selected_cluster_algorithm"
                    ],
                    "best_valid_geometry_k": (
                        np.nan
                        if best_valid_geometry is None
                        else int(best_valid_geometry["requested_k"])
                    ),
                    "best_valid_geometry_rmse": (
                        np.nan
                        if best_valid_geometry is None
                        else float(best_valid_geometry["rmse"])
                    ),
                    "max_foundation_compatible_k": (
                        np.nan
                        if foundation_compatible.empty
                        else int(foundation_compatible["requested_k"].max())
                    ),
                    "best_foundation_compatible_k": (
                        np.nan
                        if best_foundation_compatible is None
                        else int(best_foundation_compatible["requested_k"])
                    ),
                    "best_foundation_compatible_rmse": (
                        np.nan
                        if best_foundation_compatible is None
                        else float(best_foundation_compatible["rmse"])
                    ),
                    "silhouette_best_k": (
                        np.nan
                        if silhouette_best is None
                        else int(silhouette_best["requested_k"])
                    ),
                    "silhouette_best_score": (
                        np.nan
                        if silhouette_best is None
                        else float(silhouette_best["cluster_selection_score"])
                    ),
                    "silhouette_selected_algorithm": (
                        None
                        if silhouette_best is None
                        else silhouette_best["selected_cluster_algorithm"]
                    ),
                    "silhouette_matches_downstream": (
                        False
                        if silhouette_best is None
                        else int(silhouette_best["requested_k"])
                        == int(best["requested_k"])
                    ),
                }
            )
            rows.append(row)
        return pd.DataFrame(rows).sort_values(self._cell_keys()).reset_index(drop=True)

    @staticmethod
    def _build_summary(comparisons: pd.DataFrame) -> pd.DataFrame:
        if comparisons.empty:
            return pd.DataFrame()
        valid = comparisons[
            comparisons["reference_complete"]
            & comparisons["row_budget_matches_k2"]
        ]
        if valid.empty:
            return pd.DataFrame()
        keys = [
            "model",
            "budget_ratio",
            "ensemble_method",
            "voting_pruning",
            "router",
            "requested_k",
        ]
        return (
            valid.groupby(keys, dropna=False)
            .agg(
                datasets=("dataset", "nunique"),
                mean_rmse_change_vs_k2_pct=("rmse_change_vs_k2_pct", "mean"),
                median_rmse_change_vs_k2_pct=("rmse_change_vs_k2_pct", "median"),
                wins_vs_k2=("outcome_vs_k2", lambda x: int((x == "better").sum())),
                losses_vs_k2=("outcome_vs_k2", lambda x: int((x == "worse").sum())),
                ties_vs_k2=("outcome_vs_k2", lambda x: int((x == "tie").sum())),
                mean_active_experts=("active_expert_count", "mean"),
                mean_active_fraction=("active_fraction_of_requested", "mean"),
                geometry_valid_rate=(
                    "cluster_candidate_valid",
                    lambda values: float(pd.Series(values).eq(True).mean()),
                ),
                foundation_compatible_rate=(
                    "foundation_min_rows_compatible", "mean"
                ),
                mean_fit_sec=("fit_sec", "mean"),
                mean_inference_sec=("inference_sec", "mean"),
            )
            .reset_index()
            .sort_values(keys)
        )

    @staticmethod
    def _build_algorithm_summary(runs: pd.DataFrame) -> pd.DataFrame:
        if runs.empty:
            return pd.DataFrame()
        valid = runs.dropna(subset=["requested_k", "selected_cluster_algorithm"])
        if valid.empty:
            return pd.DataFrame()
        return (
            valid.groupby(
                ["requested_k", "selected_cluster_algorithm"], dropna=False
            )
            .agg(
                selections=("dataset", "size"),
                datasets=("dataset", "nunique"),
                mean_selection_score=("cluster_selection_score", "mean"),
                valid_selections=(
                    "cluster_candidate_valid",
                    lambda values: int(pd.Series(values).eq(True).sum()),
                ),
            )
            .reset_index()
            .sort_values(["requested_k", "selected_cluster_algorithm"])
        )

    @staticmethod
    def _write_markdown_report(
        tables: Mapping[str, pd.DataFrame],
        output_dir: Path,
    ) -> None:
        runs = tables["runs"]
        comparisons = tables["comparisons"]
        best_k = tables["best_k"]
        summary = tables["summary"]
        pruning_effect = tables["pruning_effect"]
        complete = int(
            (
                comparisons.get("reference_complete", pd.Series(dtype=bool))
                & comparisons.get("row_budget_matches_k2", pd.Series(dtype=bool))
            ).sum()
        )
        non_two = int(
            (best_k.get("best_downstream_k", pd.Series(dtype=float)) > 2).sum()
        )
        matches = int(
            best_k.get(
                "silhouette_matches_downstream", pd.Series(dtype=bool)
            ).sum()
        )
        lines = [
            "# RMT chunk-count / expert-count ablation",
            "",
            "All k arms use the same total post-partition row budget. Negative `rmse_change_vs_k2_pct` means that the requested k improves over k=2. Voting may prune trained experts; routed_weighted keeps the routed experts active.",
            "",
            f"- Completed RMT runs: {int(runs['rmse'].notna().sum()) if 'rmse' in runs else 0}",
            f"- Matched run-to-k=2 comparisons: {complete}",
            f"- Downstream cells whose best k is greater than 2: {non_two}/{len(best_k)}",
            f"- Silhouette/downstream best-k agreements: {matches}/{len(best_k)}",
            "",
            "## Best k by dataset/model/budget/aggregation",
            "",
            "No complete cells are available yet." if best_k.empty else markdown_table(best_k),
            "",
            "## Aggregate change relative to k=2",
            "",
            "No complete comparisons are available yet." if summary.empty else markdown_table(summary),
            "",
            "## Voting pruning effect at fixed k",
            "",
            "Negative `pruning_rmse_change_vs_all_experts_pct` means that validation pruning improves test RMSE relative to retaining every trained expert.",
            "",
            (
                "Run with `--voting-pruning both` to create matched pairs."
                if pruning_effect.empty
                else markdown_table(pruning_effect)
            ),
        ]
        (output_dir / "chunk_count_ablation_report.md").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )


class RMTChunkCountAblationMediumOrchestrator(
    RMTRegressionExperimentOrchestrator
):
    def __init__(self, config: RMTChunkCountAblationConfig) -> None:
        super().__init__(config)
        self.config = config
        self.chunk_count_report_builder = ChunkCountAblationReportBuilder()

    def _run_id_prefix(self) -> str:
        return "run_rmt_chunk_count_ablation_medium"

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
                    "--output-dir, or continue it with --resume-from."
                )
            return BenchmarkLogger(
                run_id=output_dir.name,
                artifacts_root=output_dir.parent,
            )
        run_id = f"{self._run_id_prefix()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        return BenchmarkLogger(run_id=run_id, artifacts_root=BENCHMARK_DIR / "results")

    def _build_strategy_grid(self) -> StrategyGridContract:
        return normalize_strategy_grid(
            make_chunk_count_ablation_strategy_configs(self.config)
        )

    def _create_incremental_saver(
        self,
        logger: BenchmarkLogger,
    ) -> IncrementalExperimentSaver:
        saver = super()._create_incremental_saver(logger)
        saver.records_path = logger.paths.metrics / "chunk_count_ablation_runs.jsonl"

        def _build_chunk_count_tables(
            records: Sequence[Mapping[str, Any]],
        ) -> None:
            self.chunk_count_report_builder.build(records, logger.paths.metrics)

        saver.snapshot_hooks = (*saver.snapshot_hooks, _build_chunk_count_tables)
        return saver

    def _build_report_artifacts(
        self,
        run_records: Sequence[Mapping[str, Any]],
        logger: BenchmarkLogger,
    ) -> None:
        super()._build_report_artifacts(run_records, logger)
        tables = self.chunk_count_report_builder.build(
            run_records, logger.paths.metrics
        )
        (logger.paths.root / "chunk_count_ablation_protocol.json").write_text(
            json.dumps(
                json_ready(self._comparison_protocol()),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        self._validate_completed_ablation(tables)

    def _validate_completed_ablation(
        self,
        tables: Mapping[str, pd.DataFrame],
    ) -> None:
        runs = tables["runs"]
        if runs.empty:
            raise RuntimeError("Chunk-count ablation produced no RMT runs")
        coordinate_keys = ChunkCountAblationReportBuilder._coordinate_keys()
        duplicate_runs = int(runs.duplicated(coordinate_keys).sum())
        failed_runs = int(runs["rmse"].isna().sum() + runs["error"].notna().sum())
        wrong_realized_k = int(
            (~np.isclose(runs["requested_k"], runs["realized_k"])).sum()
        )
        wrong_partition_count = int(
            (~np.isclose(runs["requested_k"], runs["selected_partition_count"])).sum()
        )
        routed = runs[runs["ensemble_method"] == "routed_weighted"]
        wrong_routed_active_k = int(
            (~np.isclose(routed["requested_k"], routed["active_expert_count"])).sum()
        )
        unpruned_voting = runs[
            (runs["ensemble_method"] == "voting")
            & (runs["voting_pruning"] == "disabled")
        ]
        wrong_unpruned_active_k = int(
            (~np.isclose(
                unpruned_voting["requested_k"],
                unpruned_voting["active_expert_count"],
            )).sum()
        )
        budget_mismatches = int(
            runs.groupby(ChunkCountAblationReportBuilder._cell_keys(), dropna=False)[
                "selected_rows"
            ].nunique(dropna=False).gt(1).sum()
        )
        expected_per_dataset_model_split = (
            len(self.config.chunk_counts)
            * len(self.config.budget_ratios)
            * sum(
                len(self.config.router_modes)
                if ensemble == "routed_weighted"
                else len(self.config.voting_pruning_modes)
                for ensemble in self.config.ensemble_methods
            )
        )
        group_sizes = runs.groupby(
            ["dataset", "model", "split_label", "seed"], dropna=False
        ).size()
        incomplete_groups = int((group_sizes != expected_per_dataset_model_split).sum())
        if any(
            (
                duplicate_runs,
                failed_runs,
                wrong_realized_k,
                wrong_partition_count,
                wrong_routed_active_k,
                wrong_unpruned_active_k,
                budget_mismatches,
                incomplete_groups,
            )
        ):
            raise RuntimeError(
                "Chunk-count ablation integrity check failed: "
                f"duplicate_runs={duplicate_runs}, failed_runs={failed_runs}, "
                f"wrong_realized_k={wrong_realized_k}, "
                f"wrong_partition_count={wrong_partition_count}, "
                f"wrong_routed_active_k={wrong_routed_active_k}, "
                f"wrong_unpruned_active_k={wrong_unpruned_active_k}, "
                f"row_budget_mismatched_cells={budget_mismatches}, "
                f"incomplete_dataset_model_groups={incomplete_groups}. See "
                "metrics/chunk_count_runs.csv."
            )

    def _build_run_meta(
        self,
        logger: BenchmarkLogger,
        run_records: Sequence[Mapping[str, Any]],
        status: str = "completed",
    ) -> dict[str, Any]:
        return {
            **super()._build_run_meta(logger, run_records, status),
            "chunk_count_ablation": self._comparison_protocol(),
        }

    def _comparison_protocol(self) -> dict[str, Any]:
        methods_per_budget = sum(
            len(self.config.router_modes)
            if ensemble == "routed_weighted"
            else len(self.config.voting_pruning_modes)
            for ensemble in self.config.ensemble_methods
        )
        expert_fits_per_dataset_model = (
            sum(self.config.chunk_counts) * len(self.config.budget_ratios)
        )
        return {
            "question": (
                "Does downstream regression quality continue to improve beyond "
                "two RMT chunks/experts, and does balanced silhouette choose the "
                "same k as downstream RMSE?"
            ),
            "fixed_total_row_budget_across_k": True,
            "reference_k": REFERENCE_CHUNK_COUNT,
            "chunk_counts": list(self.config.chunk_counts),
            "budgets": list(self.config.budget_ratios),
            "cluster_algorithms_at_each_k": list(self.config.cluster_algorithms),
            "cluster_algorithm_selection": "best balanced_silhouette at fixed k",
            "ensemble_methods": list(self.config.ensemble_methods),
            "voting_pruning_modes": [
                "enabled" if enabled else "disabled"
                for enabled in self.config.voting_pruning_modes
            ],
            "router_modes": list(self.config.router_modes),
            "seed": self.config.seed,
            "leaf_runs_per_dataset_model": (
                len(self.config.chunk_counts)
                * len(self.config.budget_ratios)
                * methods_per_budget
                + (1 if self.config.include_direct_baseline else 0)
            ),
            "expensive_structure_fits_per_dataset_split": len(
                self.config.chunk_counts
            ),
            "trained_expert_fits_per_dataset_model": expert_fits_per_dataset_model,
            "cache_expectation": (
                "for each fixed k, RMT structure is reused across budgets/models; "
                "trained chunk models are reused between pruning modes, voting, "
                "and routed_weighted"
            ),
            "interpretation": {
                "voting": "all chunks are trained, then validation pruning may reduce active_expert_count",
                "routed_weighted": "all routed chunk experts remain active",
            },
        }


def run_rmt_chunk_count_ablation_medium(
    *,
    regression_tasks: Sequence[str] | None = None,
    models: Sequence[str] = DEFAULT_CHUNK_COUNT_MODELS,
    budget_ratios: Sequence[float] = DEFAULT_CHUNK_COUNT_BUDGETS,
    chunk_counts: Sequence[int] = DEFAULT_CHUNK_COUNTS,
    cluster_algorithms: Sequence[str] = DEFAULT_CHUNK_COUNT_CLUSTER_ALGORITHMS,
    voting_pruning_modes: Sequence[bool] = DEFAULT_VOTING_PRUNING_MODES,
    ensemble_methods: Sequence[str] = DEFAULT_CHUNK_COUNT_ENSEMBLES,
    max_train_rows: int | None = 300_000,
    seed: int = 42,
    show_progress: bool = True,
    include_direct_baseline: bool = False,
    output_dir: str | Path | None = None,
    resume_from: str | Path | None = None,
    resume_policy: str = ResumePolicy.RETRY_FAILED.value,
    synthetic_smoke: bool = False,
) -> Path:
    config = RMTChunkCountAblationConfig(
        regression_tasks=regression_tasks or DEFAULT_RMT_REGRESSION_TASKS,
        models=models,
        budget_ratios=budget_ratios,
        chunk_counts=chunk_counts,
        cluster_algorithms=cluster_algorithms,
        voting_pruning_modes=voting_pruning_modes,
        ensemble_methods=ensemble_methods,
        max_train_rows=max_train_rows,
        seed=seed,
        show_progress=show_progress,
        include_direct_baseline=include_direct_baseline,
        output_dir=output_dir,
        resume_from=resume_from,
        resume_policy=resume_policy,
        synthetic_smoke=synthetic_smoke,
    )
    return RMTChunkCountAblationMediumOrchestrator(config).run()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Matched-total-budget medium regression ablation over the number "
            "of RMT chunks/experts."
        )
    )
    parser.add_argument("--task", action="append", dest="tasks")
    parser.add_argument("--model", action="append", dest="models")
    parser.add_argument(
        "--ensemble-method",
        action="append",
        choices=DEFAULT_CHUNK_COUNT_ENSEMBLES,
        dest="ensemble_methods",
    )
    parser.add_argument("--budget", action="append", type=float, dest="budgets")
    parser.add_argument(
        "--chunk-count", action="append", type=int, dest="chunk_counts"
    )
    parser.add_argument(
        "--cluster-algorithm", action="append", dest="cluster_algorithms"
    )
    parser.add_argument(
        "--voting-pruning",
        choices=("on", "off", "both"),
        default=None,
        help=(
            "Forward validation pruning for voting. 'both' adds a matched "
            "all-experts control while reusing trained chunk models. The "
            "quick-foundation profile defaults to 'both'; the full profile "
            "defaults to 'on'."
        ),
    )
    parser.add_argument("--max-train-rows", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-direct-baseline", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--resume-policy", default=ResumePolicy.RETRY_FAILED.value)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--quick-foundation",
        action="store_true",
        help=(
            "Use the short TabPFN+TabICL profile: Brazilian_houses and "
            "then all remaining medium tasks from lighter to heavier, budget "
            "0.3, k=2/4, routed plus voting pruning on/off."
        ),
    )
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def _run_cli(args: argparse.Namespace) -> Path:
    smoke = bool(args.smoke)
    quick_foundation = bool(args.quick_foundation)
    output_dir = args.output_dir
    if output_dir is None and args.resume_from is not None:
        output_dir = args.resume_from
    return run_rmt_chunk_count_ablation_medium(
        regression_tasks=(
            args.tasks
            or (QUICK_FOUNDATION_TASKS if quick_foundation else None)
        ),
        models=tuple(args.models or (("lightgbm",) if smoke else DEFAULT_CHUNK_COUNT_MODELS)),
        budget_ratios=tuple(
            args.budgets
            or (
                (0.3,)
                if smoke
                else (
                    QUICK_FOUNDATION_BUDGETS
                    if quick_foundation
                    else DEFAULT_CHUNK_COUNT_BUDGETS
                )
            )
        ),
        chunk_counts=tuple(
            args.chunk_counts
            or (
                (2, 4)
                if smoke
                else (
                    QUICK_FOUNDATION_CHUNK_COUNTS
                    if quick_foundation
                    else DEFAULT_CHUNK_COUNTS
                )
            )
        ),
        cluster_algorithms=tuple(
            args.cluster_algorithms or DEFAULT_CHUNK_COUNT_CLUSTER_ALGORITHMS
        ),
        voting_pruning_modes=(
            (True, False)
            if (args.voting_pruning or ("both" if quick_foundation else "on"))
            == "both"
            else (
                (args.voting_pruning or "on") == "on",
            )
        ),
        ensemble_methods=tuple(
            args.ensemble_methods or DEFAULT_CHUNK_COUNT_ENSEMBLES
        ),
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
