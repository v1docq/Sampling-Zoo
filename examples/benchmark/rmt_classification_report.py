"""Probability-first tables and plots for RMT classification experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from examples.benchmark.rmt_experiment_utils import task_key, value_series


DEFAULT_CLASSIFICATION_DELTAS: tuple[float, ...] = (0.01, 0.03, 0.05)


@dataclass(frozen=True)
class RMTClassificationReportTableBuilder:
    """Build classification tables without mixing ROC AUC and log loss."""

    efficiency_deltas: Sequence[float] = DEFAULT_CLASSIFICATION_DELTAS

    def build_report_tables(
        self,
        run_records: Sequence[Mapping[str, Any]],
        output_dir: Path,
    ) -> dict[str, pd.DataFrame]:
        output_dir.mkdir(parents=True, exist_ok=True)
        normalized = pd.json_normalize(list(run_records), sep=".")
        if normalized.empty:
            return self._write_empty_tables(output_dir)

        raw = self._build_raw_runs_table(normalized)
        raw = self._attach_matched_baseline(raw)
        self._write(raw, output_dir / "classification_raw_runs.csv")

        efficiency = self._build_efficiency_table(raw)
        self._write(
            efficiency,
            output_dir / "classification_efficiency_curve.csv",
        )
        minimal_budget = self._build_minimal_budget_table(efficiency)
        self._write(
            minimal_budget,
            output_dir / "classification_minimal_effective_budget.csv",
        )
        class_safety = self._build_class_safety_table(raw)
        self._write(
            class_safety,
            output_dir / "classification_class_safety.csv",
        )
        return {
            "raw": raw,
            "efficiency": efficiency,
            "minimal_budget": minimal_budget,
            "class_safety": class_safety,
        }

    def _build_raw_runs_table(self, frame: pd.DataFrame) -> pd.DataFrame:
        primary_metric = value_series(
            frame,
            "extra.primary_metric",
            default=None,
        )
        roc_auc = self._numeric(frame, "model_metrics.roc_auc")
        log_loss = self._numeric(frame, "model_metrics.log_loss")
        score = pd.Series(
            np.where(primary_metric.eq("roc_auc"), roc_auc, log_loss),
            index=frame.index,
            dtype=float,
        )
        raw = pd.DataFrame(
            {
                "dataset": value_series(frame, "dataset"),
                "model": value_series(frame, "strategy_params.model"),
                "split_label": value_series(
                    frame,
                    "strategy_params.split_label",
                    default="single_split",
                ),
                "experiment_scenario": value_series(
                    frame,
                    "strategy_params.experiment_scenario",
                    default=None,
                ),
                "scenario_group": value_series(
                    frame,
                    "strategy_params.scenario_group",
                    default=None,
                ),
                "scenario_family": value_series(
                    frame,
                    "strategy_params.scenario_family",
                    default=None,
                ),
                "sampler": value_series(frame, "strategy_params.strategy"),
                "ensemble_method": value_series(
                    frame,
                    "strategy_params.ensemble_method",
                ),
                "partition_model_mode": value_series(
                    frame,
                    "strategy_params.partition_model_mode",
                    default=None,
                ),
                "router": value_series(
                    frame,
                    "strategy_params.router",
                    default=None,
                ),
                "row_selection_method": value_series(
                    frame,
                    "strategy_params.selection_method",
                    default=None,
                ),
                "class_allocation_policy": value_series(
                    frame,
                    "strategy_params.class_allocation_policy",
                    default=None,
                ),
                "class_coverage_policy": value_series(
                    frame,
                    "strategy_params.class_coverage_policy",
                    default=None,
                ),
                "budget_ratio": self._numeric(
                    frame,
                    "strategy_params.budget_ratio",
                ),
                "total_train_rows": self._numeric(
                    frame,
                    "sample_stats.sample_size",
                ),
                "model_fit_rows_total": self._numeric(
                    frame,
                    "sample_stats.model_fit_rows_total",
                ),
                "class_coverage_rows_added": self._numeric(
                    frame,
                    "sample_stats.class_coverage_rows_added",
                ),
                "n_classes": self._numeric(frame, "extra.n_classes"),
                "primary_metric": primary_metric,
                "score": score,
                "accuracy": self._numeric(frame, "model_metrics.accuracy"),
                "f1_macro": self._numeric(frame, "model_metrics.f1_macro"),
                "f1_weighted": self._numeric(
                    frame,
                    "model_metrics.f1_weighted",
                ),
                "roc_auc": roc_auc,
                "log_loss": log_loss,
                "brier_score": self._numeric(
                    frame,
                    "model_metrics.brier_score",
                ),
                "expected_calibration_error": self._numeric(
                    frame,
                    "model_metrics.expected_calibration_error",
                ),
                "fit_time": self._numeric(frame, "timings_sec.fit"),
                "inference_time": self._numeric(
                    frame,
                    "timings_sec.inference",
                ),
                "selected_n_partitions": self._numeric(
                    frame,
                    "extra.sampler_diagnostics.selected_n_partitions",
                ),
                "budget_feasible": value_series(
                    frame,
                    "extra.sampler_diagnostics.partition_budget_plan.feasible",
                    default=None,
                ),
                "budget_violations": value_series(
                    frame,
                    "extra.sampler_diagnostics.partition_budget_plan.violations",
                    default=None,
                ),
                "class_coverage_guaranteed": value_series(
                    frame,
                    "extra.sampler_diagnostics.class_coverage_guaranteed",
                    default=None,
                ),
                "chunks_with_missing_classes": self._numeric(
                    frame,
                    (
                        "extra.partition_diagnostics.class_balance_summary."
                        "chunks_with_missing_classes"
                    ),
                ),
                "single_class_chunks": self._numeric(
                    frame,
                    (
                        "extra.partition_diagnostics.class_balance_summary."
                        "single_class_chunks"
                    ),
                ),
                "class_distribution_drift": self._numeric(
                    frame,
                    (
                        "extra.partition_diagnostics.class_balance_summary."
                        "class_distribution_drift_avg"
                    ),
                ),
                "validation_routing_entropy": self._numeric(
                    frame,
                    (
                        "extra.validation_diagnostics.routing."
                        "mean_normalized_entropy"
                    ),
                ),
                "test_routing_entropy": self._numeric(
                    frame,
                    "extra.test_routing_diagnostics.mean_normalized_entropy",
                ),
                "error_code": value_series(
                    frame,
                    "extra.error_code",
                    default=None,
                ),
                "error": value_series(frame, "extra.error", default=None),
            }
        )
        raw.insert(1, "task_key", raw["dataset"].map(task_key))
        raw["metric_direction"] = np.where(
            raw["primary_metric"].eq("roc_auc"),
            "higher_is_better",
            "lower_is_better",
        )
        return raw

    @staticmethod
    def _attach_matched_baseline(raw: pd.DataFrame) -> pd.DataFrame:
        keys = ["dataset", "model", "split_label", "primary_metric"]
        baseline = (
            raw[
                raw["sampler"].eq("full_dataset")
                & raw["score"].notna()
            ]
            .groupby(keys, as_index=False, dropna=False)["score"]
            .first()
            .rename(columns={"score": "score_ref"})
        )
        enriched = raw.merge(baseline, on=keys, how="left")
        enriched["score_ref_source"] = np.where(
            enriched["score_ref"].notna(),
            "full_dataset",
            None,
        )
        enriched["score_drop"] = np.where(
            enriched["primary_metric"].eq("roc_auc"),
            enriched["score_ref"] - enriched["score"],
            enriched["score"] - enriched["score_ref"],
        )
        return enriched

    @staticmethod
    def _build_efficiency_table(raw: pd.DataFrame) -> pd.DataFrame:
        selected = raw[
            ~raw["sampler"].eq("full_dataset")
            & raw["score"].notna()
        ]
        group_columns = [
            "dataset",
            "task_key",
            "model",
            "scenario_group",
            "scenario_family",
            "sampler",
            "ensemble_method",
            "partition_model_mode",
            "router",
            "row_selection_method",
            "class_allocation_policy",
            "primary_metric",
            "metric_direction",
            "budget_ratio",
        ]
        if selected.empty:
            return pd.DataFrame(columns=group_columns)
        return (
            selected.groupby(group_columns, as_index=False, dropna=False)
            .agg(
                runs=("score", "size"),
                score=("score", "mean"),
                score_std=("score", "std"),
                score_ref=("score_ref", "mean"),
                score_drop=("score_drop", "mean"),
                brier_score=("brier_score", "mean"),
                expected_calibration_error=(
                    "expected_calibration_error",
                    "mean",
                ),
                fit_time=("fit_time", "mean"),
                inference_time=("inference_time", "mean"),
                total_train_rows=("total_train_rows", "mean"),
                model_fit_rows_total=("model_fit_rows_total", "mean"),
                class_coverage_rows_added=(
                    "class_coverage_rows_added",
                    "mean",
                ),
                chunks_with_missing_classes=(
                    "chunks_with_missing_classes",
                    "mean",
                ),
                single_class_chunks=("single_class_chunks", "mean"),
                class_distribution_drift=(
                    "class_distribution_drift",
                    "mean",
                ),
                validation_routing_entropy=(
                    "validation_routing_entropy",
                    "mean",
                ),
            )
            .sort_values(
                ["dataset", "model", "scenario_family", "budget_ratio"]
            )
        )

    def _build_minimal_budget_table(
        self,
        efficiency: pd.DataFrame,
    ) -> pd.DataFrame:
        if efficiency.empty:
            return pd.DataFrame()
        keys = [
            "dataset",
            "task_key",
            "model",
            "scenario_group",
            "scenario_family",
            "primary_metric",
        ]
        rows: list[dict[str, Any]] = []
        for group_key, group in efficiency.groupby(keys, dropna=False):
            ordered = group.sort_values("budget_ratio")
            identity = dict(zip(keys, group_key))
            for delta in self.efficiency_deltas:
                accepted = ordered[
                    ordered["score_drop"].notna()
                    & ordered["score_drop"].le(float(delta) + 1e-12)
                ]
                rows.append(
                    {
                        **identity,
                        "allowed_absolute_drop": float(delta),
                        "minimal_budget_ratio": (
                            float(accepted.iloc[0]["budget_ratio"])
                            if not accepted.empty
                            else np.nan
                        ),
                    }
                )
        return pd.DataFrame(rows)

    @staticmethod
    def _build_class_safety_table(raw: pd.DataFrame) -> pd.DataFrame:
        columns = [
            "dataset",
            "model",
            "experiment_scenario",
            "scenario_family",
            "budget_ratio",
            "class_allocation_policy",
            "class_coverage_guaranteed",
            "budget_feasible",
            "budget_violations",
            "class_coverage_rows_added",
            "chunks_with_missing_classes",
            "single_class_chunks",
            "class_distribution_drift",
        ]
        return raw.loc[raw["sampler"].eq("rmt_contraction"), columns].copy()

    @staticmethod
    def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
        return pd.to_numeric(value_series(frame, column), errors="coerce")

    @staticmethod
    def _write(table: pd.DataFrame, path: Path) -> None:
        table.to_csv(path, index=False)

    @staticmethod
    def _write_empty_tables(output_dir: Path) -> dict[str, pd.DataFrame]:
        names = {
            "raw": "classification_raw_runs.csv",
            "efficiency": "classification_efficiency_curve.csv",
            "minimal_budget": "classification_minimal_effective_budget.csv",
            "class_safety": "classification_class_safety.csv",
        }
        tables = {name: pd.DataFrame() for name in names}
        for name, filename in names.items():
            tables[name].to_csv(output_dir / filename, index=False)
        return tables


@dataclass(frozen=True)
class RMTClassificationReportPlotBuilder:
    """Render primary quality, calibration, and class-safety diagnostics."""

    def build_plots(self, raw: pd.DataFrame, output_dir: Path) -> list[Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        if raw.empty:
            return []
        paths: list[Path] = []
        for (dataset, model), group in raw.groupby(
            ["dataset", "model"],
            dropna=False,
        ):
            paths.extend(
                self._build_dataset_model_plots(
                    group,
                    output_dir,
                    str(dataset),
                    str(model),
                )
            )
        return paths

    def _build_dataset_model_plots(
        self,
        frame: pd.DataFrame,
        output_dir: Path,
        dataset: str,
        model: str,
    ) -> list[Path]:
        safe_name = _safe_filename(f"{dataset}__{model}")
        paths = [
            self._plot_quality(
                frame,
                output_dir / f"classification_quality__{safe_name}.png",
                dataset,
                model,
            ),
            self._plot_calibration(
                frame,
                output_dir / f"classification_calibration__{safe_name}.png",
                dataset,
                model,
            ),
            self._plot_class_safety(
                frame,
                output_dir / f"classification_class_safety__{safe_name}.png",
                dataset,
                model,
            ),
        ]
        return [path for path in paths if path is not None]

    @staticmethod
    def _plot_quality(
        frame: pd.DataFrame,
        path: Path,
        dataset: str,
        model: str,
    ) -> Path | None:
        selected = frame[
            ~frame["sampler"].eq("full_dataset")
            & frame["score"].notna()
        ]
        if selected.empty:
            return None
        metric = str(selected["primary_metric"].dropna().iloc[0])
        figure, axes = plt.subplots(1, 2, figsize=(13, 4.8))
        for family, curve in selected.groupby("scenario_family", dropna=False):
            curve = curve.sort_values("budget_ratio")
            label = str(family)
            axes[0].plot(
                100.0 * curve["budget_ratio"],
                curve["score"],
                marker="o",
                label=label,
            )
            axes[1].plot(
                100.0 * curve["budget_ratio"],
                curve["score_drop"],
                marker="o",
                label=label,
            )
        reference = selected["score_ref"].dropna()
        if not reference.empty:
            axes[0].axhline(
                float(reference.iloc[0]),
                color="#252A34",
                linestyle=":",
                label="full_dataset",
            )
        axes[0].set_title(f"Primary metric: {metric}")
        axes[0].set_ylabel(metric)
        axes[1].set_title("Absolute degradation vs matched baseline")
        axes[1].set_ylabel("score_drop (lower is better)")
        for axis in axes:
            axis.set_xlabel("Training budget, %")
            axis.grid(alpha=0.25)
        axes[1].legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            fontsize=7,
        )
        figure.suptitle(f"{dataset} / {model}", fontweight="bold")
        figure.tight_layout()
        figure.savefig(path, dpi=170, bbox_inches="tight")
        plt.close(figure)
        return path

    @staticmethod
    def _plot_calibration(
        frame: pd.DataFrame,
        path: Path,
        dataset: str,
        model: str,
    ) -> Path | None:
        selected = frame[
            ~frame["sampler"].eq("full_dataset")
            & frame["brier_score"].notna()
        ]
        if selected.empty:
            return None
        figure, axes = plt.subplots(1, 2, figsize=(12, 4.4))
        for family, curve in selected.groupby("scenario_family", dropna=False):
            curve = curve.sort_values("budget_ratio")
            label = str(family)
            x = 100.0 * curve["budget_ratio"]
            axes[0].plot(x, curve["brier_score"], marker="o", label=label)
            axes[1].plot(
                x,
                curve["expected_calibration_error"],
                marker="o",
                label=label,
            )
        axes[0].set_title("Brier score")
        axes[1].set_title("Expected calibration error")
        for axis in axes:
            axis.set_xlabel("Training budget, %")
            axis.set_ylabel("Lower is better")
            axis.grid(alpha=0.25)
        axes[1].legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            fontsize=7,
        )
        figure.suptitle(f"Calibration: {dataset} / {model}", fontweight="bold")
        figure.tight_layout()
        figure.savefig(path, dpi=170, bbox_inches="tight")
        plt.close(figure)
        return path

    @staticmethod
    def _plot_class_safety(
        frame: pd.DataFrame,
        path: Path,
        dataset: str,
        model: str,
    ) -> Path | None:
        selected = frame[
            frame["sampler"].eq("rmt_contraction")
            & frame["budget_ratio"].notna()
        ]
        if selected.empty:
            return None
        figure, axes = plt.subplots(1, 2, figsize=(12, 4.4))
        for family, curve in selected.groupby("scenario_family", dropna=False):
            curve = curve.sort_values("budget_ratio")
            label = str(family)
            x = 100.0 * curve["budget_ratio"]
            axes[0].plot(
                x,
                curve["class_distribution_drift"],
                marker="o",
                label=label,
            )
            axes[1].plot(
                x,
                curve["single_class_chunks"].fillna(0.0),
                marker="o",
                label=label,
            )
        axes[0].set_title("Mean class-distribution drift")
        axes[1].set_title("Single-class chunks")
        for axis in axes:
            axis.set_xlabel("Training budget, %")
            axis.grid(alpha=0.25)
        axes[1].legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            fontsize=7,
        )
        figure.suptitle(f"Class safety: {dataset} / {model}", fontweight="bold")
        figure.tight_layout()
        figure.savefig(path, dpi=170, bbox_inches="tight")
        plt.close(figure)
        return path


def _safe_filename(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in value
    )
