from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from sampling_zoo.core.metrics.eval_metrics import metric_direction, metric_drop
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
        raw = self._attach_primary_metric(raw)
        raw = self._attach_score_baseline(raw)
        raw = self._attach_rmse_baseline(raw)
        raw = self._attach_reference_metrics(raw, reference_metrics)
        raw = self._attach_score_drop(raw)
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
                "problem_type": value_series(df, "extra.problem_type", default=None),
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
                "budget_ratio": self._numeric_series(df, "strategy_params.budget_ratio"),
                "total_train_rows": self._numeric_series(df, "sample_stats.sample_size"),
                "class_coverage_count": self._numeric_series(df, "sample_stats.class_coverage_count"),
                "rmse": self._numeric_series(df, "model_metrics.rmse"),
                "accuracy": self._numeric_series(df, "model_metrics.accuracy"),
                "f1_macro": self._numeric_series(df, "model_metrics.f1_macro"),
                "f1_weighted": self._numeric_series(df, "model_metrics.f1_weighted"),
                "roc_auc": self._numeric_series(df, "model_metrics.roc_auc"),
                "log_loss": self._numeric_series(df, "model_metrics.log_loss"),
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
                "chunks_with_missing_classes": self._numeric_series(
                    df,
                    "extra.partition_diagnostics.class_balance_summary.chunks_with_missing_classes",
                ),
                "single_class_chunks": self._numeric_series(
                    df,
                    "extra.partition_diagnostics.class_balance_summary.single_class_chunks",
                ),
                "class_distribution_drift_l1_avg": self._numeric_series(
                    df,
                    "extra.partition_diagnostics.class_balance_summary.class_distribution_drift_l1_avg",
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
                "router_training_log_loss": self._numeric_series(
                    df,
                    "extra.validation_diagnostics.router.training_log_loss",
                ),
                "router_prior_log_loss": self._numeric_series(
                    df,
                    "extra.validation_diagnostics.router.prior_log_loss",
                ),
                "router_log_loss_delta_vs_prior": self._numeric_series(
                    df,
                    "extra.validation_diagnostics.router.log_loss_delta_vs_prior",
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
                "singular_values": value_series(df, "extra.sampler_diagnostics.singular_values", default=None),
                "chunk_sizes": value_series(df, "extra.sampler_diagnostics.chunk_sizes", default=None),
            }
        )

    @staticmethod
    def _numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
        return pd.to_numeric(value_series(df, column), errors="coerce")

    def _attach_primary_metric(self, raw: pd.DataFrame) -> pd.DataFrame:
        raw = raw.copy()
        raw["primary_metric"] = raw.apply(self._infer_primary_metric, axis=1)
        raw["score"] = raw.apply(
            lambda row: row.get(row["primary_metric"], np.nan)
            if isinstance(row.get("primary_metric"), str)
            else np.nan,
            axis=1,
        )
        raw["metric_direction"] = raw["primary_metric"].map(self._safe_metric_direction)
        return raw

    @staticmethod
    def _infer_primary_metric(row: pd.Series) -> str:
        if row.get("problem_type") == "classification":
            if row.get("class_coverage_count") == 2 or pd.notna(row.get("roc_auc")):
                return "roc_auc"
            return "log_loss"
        return "rmse"

    @staticmethod
    def _safe_metric_direction(metric: str) -> str | None:
        try:
            return metric_direction(metric)
        except ValueError:
            return None

    def _attach_score_baseline(self, raw: pd.DataFrame) -> pd.DataFrame:
        baseline = self._full_dataset_score_baseline(raw)
        if baseline.empty:
            baseline = self._best_observed_score_baseline(raw)
        return raw.merge(baseline, on=["dataset", "model", "primary_metric"], how="left")

    @staticmethod
    def _full_dataset_score_baseline(raw: pd.DataFrame) -> pd.DataFrame:
        return RMTReportTableBuilder._best_by_direction(raw[raw["sampler"] == "full_dataset"])

    @staticmethod
    def _best_observed_score_baseline(raw: pd.DataFrame) -> pd.DataFrame:
        return RMTReportTableBuilder._best_by_direction(raw)

    @staticmethod
    def _best_by_direction(raw: pd.DataFrame) -> pd.DataFrame:
        if raw.empty:
            return pd.DataFrame(columns=["dataset", "model", "primary_metric", "score_ref"])
        rows = []
        for keys, group in raw.dropna(subset=["score"]).groupby(["dataset", "model", "primary_metric"], dropna=False):
            metric = keys[2]
            direction = RMTReportTableBuilder._safe_metric_direction(metric)
            if direction == "higher":
                best = group["score"].max()
            elif direction == "lower":
                best = group["score"].min()
            else:
                best = np.nan
            rows.append({
                "dataset": keys[0],
                "model": keys[1],
                "primary_metric": metric,
                "score_ref": best,
            })
        if not rows:
            return pd.DataFrame(columns=["dataset", "model", "primary_metric", "score_ref"])
        return pd.DataFrame(rows)

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
    def _attach_score_drop(raw: pd.DataFrame) -> pd.DataFrame:
        raw = raw.copy()
        drops = []
        for _, row in raw.iterrows():
            metric = row.get("primary_metric")
            score = row.get("score")
            reference = row.get("score_ref")
            if pd.isna(score) or pd.isna(reference):
                drops.append(np.nan)
                continue
            try:
                drops.append(metric_drop(str(metric), float(score), float(reference)))
            except ValueError:
                drops.append(np.nan)
        raw["score_drop"] = drops
        return raw

    @staticmethod
    def _build_efficiency_table(raw: pd.DataFrame) -> pd.DataFrame:
        return (
            raw[raw["sampler"] != "full_dataset"]
            .groupby(
                [
                    "dataset",
                    "problem_type",
                    "model",
                    "primary_metric",
                    "sampler",
                    "ensemble_method",
                    "router",
                    "view_strategy",
                    "partition_selection_method",
                    "selected_cluster_algorithm",
                    "cluster_selection_metric",
                    "cluster_ensemble_method",
                    "budget_ratio",
                ],
                as_index=False,
                dropna=False,
            )
            .agg(
                {
                    "n_views": "mean",
                    "selected_n_partitions": "mean",
                    "total_train_rows": "mean",
                    "class_coverage_count": "mean",
                    "score": "mean",
                    "score_ref": "mean",
                    "score_drop": "mean",
                    "rmse": "mean",
                    "rmse_ref": "mean",
                    "rmse_drop": "mean",
                    "accuracy": "mean",
                    "f1_macro": "mean",
                    "f1_weighted": "mean",
                    "roc_auc": "mean",
                    "log_loss": "mean",
                    "fit_time": "mean",
                    "inference_time": "mean",
                    "leverage_entropy": "mean",
                    "effective_sample_count": "mean",
                    "chunk_size_imbalance_ratio": "mean",
                    "chunk_size_cv": "mean",
                    "target_mean_abs_drift_avg": "mean",
                    "target_mean_std_units_avg": "mean",
                    "target_quantile_l1_drift_avg": "mean",
                    "chunks_with_missing_classes": "mean",
                    "single_class_chunks": "mean",
                    "class_distribution_drift_l1_avg": "mean",
                    "validation_mean_max_routing_proba": "mean",
                    "validation_mean_routing_entropy": "mean",
                    "router_training_rmse": "mean",
                    "router_prior_rmse": "mean",
                    "router_rmse_delta_vs_prior": "mean",
                    "router_training_log_loss": "mean",
                    "router_prior_log_loss": "mean",
                    "router_log_loss_delta_vs_prior": "mean",
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
                "ensemble_method",
                "router",
                "budget_ratio",
            ])
        )

    def _build_minimal_budget_table(self, efficiency: pd.DataFrame) -> pd.DataFrame:
        minimal_rows: list[dict[str, Any]] = []
        for delta in self.efficiency_deltas:
            eligible = efficiency[self._within_allowed_degradation(efficiency, delta)].copy()
            if eligible.empty:
                continue
            eligible = eligible.sort_values(["budget_ratio"])
            grouped = eligible.groupby(
                [
                    "dataset",
                    "problem_type",
                    "model",
                    "primary_metric",
                    "sampler",
                    "ensemble_method",
                    "router",
                    "view_strategy",
                    "partition_selection_method",
                    "selected_cluster_algorithm",
                    "cluster_selection_metric",
                    "cluster_ensemble_method",
                ],
                as_index=False,
                dropna=False,
            ).first()
            grouped["delta"] = delta
            minimal_rows.extend(grouped.to_dict(orient="records"))
        return pd.DataFrame(minimal_rows)

    @staticmethod
    def _within_allowed_degradation(efficiency: pd.DataFrame, delta: float) -> pd.Series:
        if efficiency.empty:
            return pd.Series(dtype=bool)
        degradation = efficiency["score_drop"].copy()
        rmse_mask = efficiency["primary_metric"].eq("rmse")
        if "rmse_drop" in efficiency.columns:
            degradation.loc[rmse_mask] = efficiency.loc[rmse_mask, "rmse_drop"]
        return degradation.le(delta).fillna(False)


def build_rmt_report_tables(
    run_records: Sequence[Mapping[str, Any]],
    output_dir: Path,
    reference_metrics: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    return RMTReportTableBuilder().build_report_tables(run_records, output_dir, reference_metrics)
