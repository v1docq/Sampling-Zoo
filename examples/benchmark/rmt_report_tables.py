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

        partition_comparison = self._build_partition_selection_comparison(
            raw
        )
        self._write_table(
            partition_comparison,
            output_dir / "partition_selection_comparison.csv",
        )

        minimal_budget = self._build_minimal_budget_table(efficiency)
        self._write_table(minimal_budget, output_dir / "minimal_effective_budget.csv")
        return {
            "raw": raw,
            "efficiency": efficiency,
            "partition_selection_comparison": partition_comparison,
            "minimal_budget": minimal_budget,
        }

    @staticmethod
    def _normalize_records(run_records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
        return pd.json_normalize(list(run_records), sep=".")

    @staticmethod
    def _write_empty_tables(output_dir: Path) -> dict[str, pd.DataFrame]:
        empty = pd.DataFrame()
        empty.to_csv(output_dir / "rmt_raw_runs.csv", index=False)
        empty.to_csv(output_dir / "sample_efficiency_curve.csv", index=False)
        empty.to_csv(
            output_dir / "partition_selection_comparison.csv",
            index=False,
        )
        empty.to_csv(output_dir / "minimal_effective_budget.csv", index=False)
        return {
            "raw": empty,
            "efficiency": empty,
            "partition_selection_comparison": empty,
            "minimal_budget": empty,
        }

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
                "downstream_proxy_global_baseline_loss": self._numeric_series(
                    df,
                    (
                        "extra.sampler_diagnostics."
                        "partition_selection_selected_candidate.components."
                        "downstream_proxy.global_baseline_loss"
                    ),
                ),
                "downstream_proxy_concatenated_loss": self._numeric_series(
                    df,
                    (
                        "extra.sampler_diagnostics."
                        "partition_selection_selected_candidate.components."
                        "downstream_proxy.concatenated_loss"
                    ),
                ),
                "downstream_proxy_candidate_loss": self._numeric_series(
                    df,
                    (
                        "extra.sampler_diagnostics."
                        "partition_selection_selected_candidate.components."
                        "downstream_proxy.candidate_loss"
                    ),
                ),
                "downstream_proxy_gain_vs_global": self._numeric_series(
                    df,
                    (
                        "extra.sampler_diagnostics."
                        "partition_selection_selected_candidate.components."
                        "downstream_proxy.relative_gain_vs_global"
                    ),
                ),
                "downstream_proxy_gain_vs_concatenated": self._numeric_series(
                    df,
                    (
                        "extra.sampler_diagnostics."
                        "partition_selection_selected_candidate.components."
                        "downstream_proxy.relative_gain_vs_concatenated"
                    ),
                ),
                "downstream_proxy_budget_reference_rows": self._numeric_series(
                    df,
                    (
                        "extra.sampler_diagnostics."
                        "partition_selection_selected_candidate.components."
                        "downstream_proxy.budget_reference_rows"
                    ),
                ),
                "downstream_proxy_train_rows": self._numeric_series(
                    df,
                    (
                        "extra.sampler_diagnostics."
                        "partition_selection_selected_candidate.components."
                        "downstream_proxy.proxy_train_rows"
                    ),
                ),
                "downstream_proxy_requested_budget_size": self._numeric_series(
                    df,
                    (
                        "extra.sampler_diagnostics."
                        "partition_selection_selected_candidate.components."
                        "downstream_proxy.requested_budget_size"
                    ),
                ),
                "downstream_proxy_selected_budget_size": self._numeric_series(
                    df,
                    (
                        "extra.sampler_diagnostics."
                        "partition_selection_selected_candidate.components."
                        "downstream_proxy.selected_budget_size"
                    ),
                ),
                "downstream_proxy_budget_violations": value_series(
                    df,
                    (
                        "extra.sampler_diagnostics."
                        "partition_selection_selected_candidate.components."
                        "downstream_proxy.budget_violations"
                    ),
                    default=None,
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
                "sampler_fit_time": self._numeric_series(
                    df,
                    "extra.sampler_diagnostics.runtime.total_seconds",
                ),
                "preprocessing_time": self._numeric_series(
                    df,
                    (
                        "extra.sampler_diagnostics.runtime."
                        "stage_seconds.preprocessing"
                    ),
                ),
                "partition_selection_time": self._numeric_series(
                    df,
                    (
                        "extra.sampler_diagnostics.runtime."
                        "stage_seconds.partition_selection_and_sampling"
                    ),
                ),
                "chunk_training_time": self._numeric_series(
                    df,
                    (
                        "extra.runtime_contract.stage_seconds."
                        "training.chunk_training_and_validation"
                    ),
                ),
                "routing_finalization_time": self._numeric_series(
                    df,
                    (
                        "extra.runtime_contract.stage_seconds."
                        "training.routing_and_finalization"
                    ),
                ),
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
                "pre_budget_chunk_sizes": value_series(
                    df,
                    "extra.sampler_diagnostics.pre_budget_chunk_sizes",
                    default=None,
                ),
                "post_budget_chunk_sizes": value_series(
                    df,
                    "extra.sampler_diagnostics.post_budget_chunk_sizes",
                    default=None,
                ),
                "budget_feasible": value_series(
                    df,
                    "extra.sampler_diagnostics.partition_budget_plan.feasible",
                    default=None,
                ),
                "budget_violations": value_series(
                    df,
                    "extra.sampler_diagnostics.partition_budget_plan.violations",
                    default=None,
                ),
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

    @staticmethod
    def _build_partition_selection_comparison(
        raw: pd.DataFrame,
    ) -> pd.DataFrame:
        key_columns = [
            "dataset",
            "model",
            "sampler",
            "ensemble_method",
            "router",
            "view_strategy",
            "partition_selection_method",
            "budget_ratio",
        ]
        metric_column = "cluster_selection_metric"
        requested_measures = [
            "rmse",
            "rmse_drop",
            "fit_time",
            "inference_time",
            "sampler_fit_time",
            "partition_selection_time",
            "chunk_training_time",
            "total_train_rows",
            "selected_n_partitions",
            "chunk_size_imbalance_ratio",
            "validation_mean_max_routing_proba",
            "validation_mean_routing_entropy",
            "validation_proxy_relative_gain",
            "downstream_proxy_candidate_loss",
            "downstream_proxy_gain_vs_global",
            "downstream_proxy_gain_vs_concatenated",
        ]
        required_columns = {*key_columns, metric_column, "rmse"}
        if raw.empty or not required_columns.issubset(raw.columns):
            return pd.DataFrame(columns=key_columns)

        selected = raw[raw[metric_column].notna()].copy()
        if selected.empty:
            return pd.DataFrame(columns=key_columns)

        measure_columns = [
            column for column in requested_measures if column in selected.columns
        ]
        group_columns = [*key_columns, metric_column]
        grouped = selected.groupby(
            group_columns,
            as_index=False,
            dropna=False,
        )
        aggregated = grouped[measure_columns].mean().merge(
            grouped.size().rename(columns={"size": "run_count"}),
            on=group_columns,
            how="left",
            validate="one_to_one",
        )
        method_names = (
            selected.groupby(group_columns, as_index=False, dropna=False)[
                "cluster_ensemble_method"
            ].first()
            if "cluster_ensemble_method" in selected.columns
            else None
        )
        metric_values = sorted(
            aggregated[metric_column].astype(str).unique().tolist(),
            key=lambda value: (value != "balanced_silhouette", value),
        )
        comparison = None
        result_columns = [*measure_columns, "run_count"]
        for metric in metric_values:
            metric_frame = aggregated[
                aggregated[metric_column].astype(str) == metric
            ].drop(columns=[metric_column])
            metric_frame = metric_frame.rename(
                columns={column: f"{column}_{metric}" for column in result_columns}
            )
            if method_names is not None:
                method_frame = method_names[
                    method_names[metric_column].astype(str) == metric
                ].drop(columns=[metric_column]).rename(
                    columns={
                        "cluster_ensemble_method": (
                            f"cluster_ensemble_method_{metric}"
                        )
                    }
                )
                metric_frame = metric_frame.merge(
                    method_frame,
                    on=key_columns,
                    how="left",
                    validate="one_to_one",
                )
            comparison = (
                metric_frame
                if comparison is None
                else comparison.merge(
                    metric_frame,
                    on=key_columns,
                    how="outer",
                    validate="one_to_one",
                )
            )
        if comparison is None:
            return pd.DataFrame(columns=key_columns)

        baseline_metric = "balanced_silhouette"
        if baseline_metric in metric_values:
            for metric in metric_values:
                if metric == baseline_metric:
                    continue
                for measure in measure_columns:
                    comparison[
                        f"{measure}_delta_{metric}_minus_{baseline_metric}"
                    ] = (
                        comparison[f"{measure}_{metric}"]
                        - comparison[f"{measure}_{baseline_metric}"]
                    )
                left_count = f"run_count_{baseline_metric}"
                right_count = f"run_count_{metric}"
                comparison[f"paired_run_count_{metric}"] = comparison[
                    [left_count, right_count]
                ].fillna(0).min(axis=1).astype(int)
                comparison[f"pair_complete_{metric}"] = (
                    comparison[left_count].notna()
                    & comparison[right_count].notna()
                    & (comparison[left_count] == comparison[right_count])
                )

        if "validation_proxy" in metric_values:
            baseline_count = f"run_count_{baseline_metric}"
            if baseline_count not in comparison.columns:
                comparison[baseline_count] = np.nan
                for measure in measure_columns:
                    comparison[f"{measure}_{baseline_metric}"] = np.nan
                    comparison[
                        f"{measure}_delta_validation_proxy_minus_{baseline_metric}"
                    ] = np.nan
            if "paired_run_count_validation_proxy" not in comparison.columns:
                validation_count = "run_count_validation_proxy"
                comparison["paired_run_count_validation_proxy"] = comparison[
                    [baseline_count, validation_count]
                ].fillna(0).min(axis=1).astype(int)
                comparison["pair_complete_validation_proxy"] = False
            comparison["paired_run_count"] = comparison[
                "paired_run_count_validation_proxy"
            ]
            comparison["pair_complete"] = comparison[
                "pair_complete_validation_proxy"
            ]
        return comparison.sort_values(
            ["dataset", "model", "budget_ratio"],
            ignore_index=True,
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
