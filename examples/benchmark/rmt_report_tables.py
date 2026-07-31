from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from rmt_experiment_utils import task_key, value_series


EFFICIENCY_DELTAS: tuple[float, ...] = (0.01, 0.03, 0.05)


@dataclass(frozen=True)
class RMTReportTableBuilder:
    efficiency_deltas: Sequence[float] = EFFICIENCY_DELTAS

    def build_report_tables(
        self,
        run_records: Sequence[Mapping[str, Any]],
        output_dir: Path,
        reference_metrics: pd.DataFrame | None = None,
    ) -> dict[str, pd.DataFrame]:
        normalized = self._normalize_records(run_records)
        if normalized.empty:
            return self._write_empty_tables(output_dir)

        raw = self._build_raw_runs_table(normalized)
        raw = self._attach_rmse_baseline(raw)
        raw = self._attach_reference_metrics(raw, reference_metrics)
        raw = self._attach_rmse_drop(raw)
        self._write_table(raw, output_dir / "rmt_raw_runs.csv")

        efficiency = self._build_efficiency_table(raw)
        self._write_table(efficiency, output_dir / "sample_efficiency_curve.csv")

        minimal_budget = self._build_minimal_budget_table(efficiency)
        self._write_table(minimal_budget, output_dir / "minimal_effective_budget.csv")
        return {"raw": raw, "efficiency": efficiency, "minimal_budget": minimal_budget}

    @staticmethod
    def _normalize_records(run_records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
        return pd.json_normalize(list(run_records), sep=".")

    @staticmethod
    def _write_empty_tables(output_dir: Path) -> dict[str, pd.DataFrame]:
        empty = pd.DataFrame()
        empty.to_csv(output_dir / "rmt_raw_runs.csv", index=False)
        return {"raw": empty, "efficiency": empty, "minimal_budget": empty}

    @staticmethod
    def _write_table(table: pd.DataFrame, path: Path) -> None:
        table.to_csv(path, index=False)

    def _build_raw_runs_table(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "dataset": value_series(df, "dataset"),
                "task_key": value_series(df, "dataset").map(task_key),
                "model": value_series(df, "strategy_params.model"),
                "sampler": value_series(df, "strategy_params.strategy"),
                "ensemble_method": value_series(df, "strategy_params.ensemble_method"),
                "router": value_series(df, "strategy_params.router", default=None),
                "view_strategy": value_series(df, "strategy_params.view_strategy", default=None),
                "n_views": self._numeric_series(df, "extra.sampler_diagnostics.n_views"),
                "n_views_policy": value_series(df, "extra.sampler_diagnostics.n_views_policy", default=None),
                "embedding_mode": value_series(df, "extra.sampler_diagnostics.embedding_mode", default=None),
                "partition_selection_method": value_series(
                    df,
                    "extra.sampler_diagnostics.partition_selection_method",
                    default=None,
                ),
                "selected_cluster_algorithm": value_series(
                    df,
                    "extra.sampler_diagnostics.selected_cluster_algorithm",
                    default=None,
                ),
                "cluster_selection_metric": value_series(
                    df,
                    "extra.sampler_diagnostics.cluster_selection_metric",
                    default=None,
                ),
                "cluster_ensemble_method": value_series(
                    df,
                    "extra.sampler_diagnostics.cluster_ensemble_method",
                    default=None,
                ),
                "selected_n_partitions": self._numeric_series(
                    df,
                    "extra.sampler_diagnostics.selected_n_partitions",
                ),
                "validation_proxy_baseline_loss": self._numeric_series(
                    df,
                    (
                        "extra.sampler_diagnostics."
                        "partition_selection_selected_candidate.components."
                        "validation_proxy.baseline_loss"
                    ),
                ),
                "validation_proxy_candidate_loss": self._numeric_series(
                    df,
                    (
                        "extra.sampler_diagnostics."
                        "partition_selection_selected_candidate.components."
                        "validation_proxy.candidate_loss"
                    ),
                ),
                "validation_proxy_relative_gain": self._numeric_series(
                    df,
                    (
                        "extra.sampler_diagnostics."
                        "partition_selection_selected_candidate.components."
                        "validation_proxy.relative_gain"
                    ),
                ),
                "validation_proxy_fallback_fraction": self._numeric_series(
                    df,
                    (
                        "extra.sampler_diagnostics."
                        "partition_selection_selected_candidate.components."
                        "validation_proxy.fallback_validation_fraction"
                    ),
                ),
                "null_model_status": value_series(
                    df,
                    "extra.sampler_diagnostics.null_model_status",
                    default=None,
                ),
                "null_primary_policy": value_series(
                    df,
                    "extra.sampler_diagnostics.null_primary_policy",
                    default=None,
                ),
                "null_empirical_bulk_edge": self._numeric_series(
                    df,
                    "extra.sampler_diagnostics.null_empirical_bulk_edge",
                ),
                "rank_by_null_edge": self._numeric_series(
                    df,
                    "extra.sampler_diagnostics.rank_by_null_edge",
                ),
                "rank_by_stability": self._numeric_series(
                    df,
                    "extra.sampler_diagnostics.rank_by_stability",
                ),
                "null_max_outlier_excess": self._numeric_series(
                    df,
                    "extra.sampler_diagnostics.null_max_outlier_excess",
                ),
                "null_successful_resamples": self._numeric_series(
                    df,
                    "extra.sampler_diagnostics.null_successful_resamples",
                ),
                "subspace_stability_status": value_series(
                    df,
                    "extra.sampler_diagnostics.subspace_stability_status",
                    default=None,
                ),
                "subspace_comparison_rank": self._numeric_series(
                    df,
                    "extra.sampler_diagnostics.subspace_comparison_rank",
                ),
                "subspace_rank_source": value_series(
                    df,
                    "extra.sampler_diagnostics.subspace_rank_source",
                    default=None,
                ),
                "rank_by_subspace_stability": self._numeric_series(
                    df,
                    "extra.sampler_diagnostics.rank_by_subspace_stability",
                ),
                "subspace_max_angle_quantile_degrees": self._numeric_series(
                    df,
                    "extra.sampler_diagnostics.subspace_max_angle_quantile_degrees",
                ),
                "subspace_normalized_projection_distance_quantile": (
                    self._numeric_series(
                        df,
                        (
                            "extra.sampler_diagnostics."
                            "subspace_normalized_projection_distance_quantile"
                        ),
                    )
                ),
                "subspace_stability_frequency": self._numeric_series(
                    df,
                    "extra.sampler_diagnostics.subspace_stability_frequency",
                ),
                "subspace_successful_resamples": self._numeric_series(
                    df,
                    "extra.sampler_diagnostics.subspace_successful_resamples",
                ),
                "budget_ratio": self._numeric_series(df, "strategy_params.budget_ratio"),
                "total_train_rows": self._numeric_series(df, "sample_stats.sample_size"),
                "rmse": self._numeric_series(df, "model_metrics.rmse"),
                "fit_time": self._numeric_series(df, "timings_sec.fit"),
                "inference_time": self._numeric_series(df, "timings_sec.inference"),
                "leverage_entropy": self._numeric_series(df, "extra.sampler_diagnostics.leverage_entropy"),
                "effective_sample_count": self._numeric_series(df, "extra.sampler_diagnostics.effective_sample_count"),
                "chunk_size_imbalance_ratio": self._numeric_series(
                    df,
                    "extra.partition_diagnostics.chunk_size_imbalance.max_to_min_ratio",
                ),
                "chunk_size_cv": self._numeric_series(
                    df,
                    "extra.partition_diagnostics.chunk_size_imbalance.coefficient_of_variation",
                ),
                "target_mean_abs_drift_avg": self._numeric_series(
                    df,
                    "extra.partition_diagnostics.target_drift_summary.mean_abs_drift_avg",
                ),
                "target_mean_std_units_avg": self._numeric_series(
                    df,
                    "extra.partition_diagnostics.target_drift_summary.mean_std_units_avg",
                ),
                "target_quantile_l1_drift_avg": self._numeric_series(
                    df,
                    "extra.partition_diagnostics.target_drift_summary.quantile_l1_drift_avg",
                ),
                "validation_mean_max_routing_proba": self._numeric_series(
                    df,
                    "extra.validation_diagnostics.routing.mean_max_probability",
                ),
                "validation_mean_routing_entropy": self._numeric_series(
                    df,
                    "extra.validation_diagnostics.routing.mean_normalized_entropy",
                ),
                "router_status": value_series(
                    df,
                    "extra.validation_diagnostics.router.status",
                    default=None,
                ),
                "router_training_rmse": self._numeric_series(
                    df,
                    "extra.validation_diagnostics.router.training_rmse",
                ),
                "router_prior_rmse": self._numeric_series(
                    df,
                    "extra.validation_diagnostics.router.prior_rmse",
                ),
                "router_rmse_delta_vs_prior": self._numeric_series(
                    df,
                    "extra.validation_diagnostics.router.rmse_delta_vs_prior",
                ),
                "router_head_status": value_series(
                    df,
                    "extra.validation_diagnostics.router_head.status",
                    default=None,
                ),
                "router_head_training_accuracy": self._numeric_series(
                    df,
                    "extra.validation_diagnostics.router_head.training_accuracy",
                ),
                "routing_refinement_status": value_series(
                    df,
                    "extra.validation_diagnostics.routing_refinement.status",
                    default=None,
                ),
                "routing_refinement_stop_reason": value_series(
                    df,
                    "extra.validation_diagnostics.routing_refinement.stop_reason",
                    default=None,
                ),
                "routing_refinement_best_iteration": self._numeric_series(
                    df,
                    "extra.validation_diagnostics.routing_refinement.best_iteration",
                ),
                "routing_refinement_metric_improvement": self._numeric_series(
                    df,
                    "extra.validation_diagnostics.routing_refinement.metric_improvement",
                ),
                "routing_refinement_final_imbalance": self._numeric_series(
                    df,
                    "extra.validation_diagnostics.routing_refinement.final_imbalance_ratio",
                ),
                "test_mean_max_routing_proba": self._numeric_series(
                    df,
                    "extra.test_routing_diagnostics.mean_max_probability",
                ),
                "test_mean_routing_entropy": self._numeric_series(
                    df,
                    "extra.test_routing_diagnostics.mean_normalized_entropy",
                ),
                "initial_singular_values": value_series(
                    df,
                    "extra.sampler_diagnostics.initial_singular_values",
                    default=None,
                ),
                "singular_values": value_series(df, "extra.sampler_diagnostics.singular_values", default=None),
                "chunk_sizes": value_series(df, "extra.sampler_diagnostics.chunk_sizes", default=None),
            }
        )

    @staticmethod
    def _numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
        return pd.to_numeric(value_series(df, column), errors="coerce")

    def _attach_rmse_baseline(self, raw: pd.DataFrame) -> pd.DataFrame:
        baseline = self._full_dataset_baseline(raw)
        if baseline.empty:
            baseline = self._best_observed_baseline(raw)
        return raw.merge(baseline, on=["dataset", "model"], how="left")

    @staticmethod
    def _full_dataset_baseline(raw: pd.DataFrame) -> pd.DataFrame:
        return (
            raw[raw["sampler"] == "full_dataset"]
            .dropna(subset=["rmse"])
            .groupby(["dataset", "model"], as_index=False)["rmse"]
            .min()
            .rename(columns={"rmse": "rmse_ref"})
        )

    @staticmethod
    def _best_observed_baseline(raw: pd.DataFrame) -> pd.DataFrame:
        return (
            raw.dropna(subset=["rmse"])
            .groupby(["dataset", "model"], as_index=False)["rmse"]
            .min()
            .rename(columns={"rmse": "rmse_ref"})
        )

    @staticmethod
    def _attach_reference_metrics(raw: pd.DataFrame, reference_metrics: pd.DataFrame | None) -> pd.DataFrame:
        if reference_metrics is None or reference_metrics.empty or "Task" not in reference_metrics.columns:
            return raw
        refs = reference_metrics.copy()
        refs["task_key"] = refs["Task"].astype(str)
        if "foundational" not in refs.columns:
            return raw
        refs["foundational_rmse_ref"] = pd.to_numeric(refs["foundational"], errors="coerce")
        return raw.merge(refs[["task_key", "foundational_rmse_ref"]], on="task_key", how="left")

    @staticmethod
    def _attach_rmse_drop(raw: pd.DataFrame) -> pd.DataFrame:
        raw = raw.copy()
        raw["rmse_drop"] = (raw["rmse"] - raw["rmse_ref"]) / raw["rmse_ref"].replace(0, np.nan)
        return raw

    @staticmethod
    def _build_efficiency_table(raw: pd.DataFrame) -> pd.DataFrame:
        return (
            raw[raw["sampler"] != "full_dataset"]
            .groupby(
                [
                    "dataset",
                    "model",
                    "sampler",
                    "ensemble_method",
                    "router",
                    "view_strategy",
                    "partition_selection_method",
                    "selected_cluster_algorithm",
                    "cluster_selection_metric",
                    "cluster_ensemble_method",
                    "null_model_status",
                    "null_primary_policy",
                    "subspace_stability_status",
                    "subspace_rank_source",
                    "budget_ratio",
                ],
                as_index=False,
                dropna=False,
            )
            .agg(
                {
                    "n_views": "mean",
                    "selected_n_partitions": "mean",
                    "validation_proxy_baseline_loss": "mean",
                    "validation_proxy_candidate_loss": "mean",
                    "validation_proxy_relative_gain": "mean",
                    "validation_proxy_fallback_fraction": "mean",
                    "null_empirical_bulk_edge": "mean",
                    "rank_by_null_edge": "mean",
                    "rank_by_stability": "mean",
                    "null_max_outlier_excess": "mean",
                    "null_successful_resamples": "mean",
                    "subspace_comparison_rank": "mean",
                    "rank_by_subspace_stability": "mean",
                    "subspace_max_angle_quantile_degrees": "mean",
                    "subspace_normalized_projection_distance_quantile": "mean",
                    "subspace_stability_frequency": "mean",
                    "subspace_successful_resamples": "mean",
                    "total_train_rows": "mean",
                    "rmse": "mean",
                    "rmse_ref": "mean",
                    "rmse_drop": "mean",
                    "fit_time": "mean",
                    "inference_time": "mean",
                    "leverage_entropy": "mean",
                    "effective_sample_count": "mean",
                    "chunk_size_imbalance_ratio": "mean",
                    "chunk_size_cv": "mean",
                    "target_mean_abs_drift_avg": "mean",
                    "target_mean_std_units_avg": "mean",
                    "target_quantile_l1_drift_avg": "mean",
                    "validation_mean_max_routing_proba": "mean",
                    "validation_mean_routing_entropy": "mean",
                    "router_training_rmse": "mean",
                    "router_prior_rmse": "mean",
                    "router_rmse_delta_vs_prior": "mean",
                    "router_head_training_accuracy": "mean",
                    "routing_refinement_best_iteration": "mean",
                    "routing_refinement_metric_improvement": "mean",
                    "routing_refinement_final_imbalance": "mean",
                    "test_mean_max_routing_proba": "mean",
                    "test_mean_routing_entropy": "mean",
                }
            )
            .sort_values([
                "dataset",
                "sampler",
                "view_strategy",
                "partition_selection_method",
                "selected_cluster_algorithm",
                "cluster_selection_metric",
                "cluster_ensemble_method",
                "null_model_status",
                "null_primary_policy",
                "subspace_stability_status",
                "subspace_rank_source",
                "ensemble_method",
                "router",
                "budget_ratio",
            ])
        )

    def _build_minimal_budget_table(self, efficiency: pd.DataFrame) -> pd.DataFrame:
        minimal_rows: list[dict[str, Any]] = []
        for delta in self.efficiency_deltas:
            eligible = efficiency[efficiency["rmse"] <= efficiency["rmse_ref"] * (1.0 + delta)].copy()
            if eligible.empty:
                continue
            eligible = eligible.sort_values(["budget_ratio"])
            grouped = eligible.groupby(
                [
                    "dataset",
                    "model",
                    "sampler",
                    "ensemble_method",
                    "router",
                    "view_strategy",
                    "partition_selection_method",
                    "selected_cluster_algorithm",
                    "cluster_selection_metric",
                    "cluster_ensemble_method",
                    "null_model_status",
                    "null_primary_policy",
                    "subspace_stability_status",
                    "subspace_rank_source",
                ],
                as_index=False,
                dropna=False,
            ).first()
            grouped["delta"] = delta
            minimal_rows.extend(grouped.to_dict(orient="records"))
        return pd.DataFrame(minimal_rows)


def build_rmt_report_tables(
    run_records: Sequence[Mapping[str, Any]],
    output_dir: Path,
    reference_metrics: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    return RMTReportTableBuilder().build_report_tables(run_records, output_dir, reference_metrics)
