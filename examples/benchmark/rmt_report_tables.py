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
                "chunk_fraction": self._chunk_fraction_series(df),
                "budget_ratio": self._numeric_series(df, "strategy_params.budget_ratio"),
                "total_train_rows": self._numeric_series(df, "sample_stats.sample_size"),
                "rmse": self._numeric_series(df, "model_metrics.rmse"),
                "fit_time": self._numeric_series(df, "timings_sec.fit"),
                "inference_time": self._numeric_series(df, "timings_sec.inference"),
                "leverage_entropy": self._numeric_series(df, "extra.sampler_diagnostics.leverage_entropy"),
                "effective_sample_count": self._numeric_series(df, "extra.sampler_diagnostics.effective_sample_count"),
                "singular_values": value_series(df, "extra.sampler_diagnostics.singular_values", default=None),
                "chunk_sizes": value_series(df, "extra.sampler_diagnostics.chunk_sizes", default=None),
            }
        )

    @staticmethod
    def _numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
        return pd.to_numeric(value_series(df, column), errors="coerce")

    @staticmethod
    def _chunk_fraction_series(df: pd.DataFrame) -> pd.Series:
        return pd.to_numeric(
            value_series(df, "strategy_params.chunk_fraction").fillna(
                value_series(df, "strategy_params.experiment_chunk_fraction")
            ),
            errors="coerce",
        )

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
            .groupby(["dataset", "model", "sampler", "ensemble_method", "chunk_fraction", "budget_ratio"], as_index=False)
            .agg(
                {
                    "total_train_rows": "mean",
                    "rmse": "mean",
                    "rmse_ref": "mean",
                    "rmse_drop": "mean",
                    "fit_time": "mean",
                    "inference_time": "mean",
                    "leverage_entropy": "mean",
                    "effective_sample_count": "mean",
                }
            )
            .sort_values(["dataset", "sampler", "ensemble_method", "chunk_fraction", "budget_ratio"])
        )

    def _build_minimal_budget_table(self, efficiency: pd.DataFrame) -> pd.DataFrame:
        minimal_rows: list[dict[str, Any]] = []
        for delta in self.efficiency_deltas:
            eligible = efficiency[efficiency["rmse"] <= efficiency["rmse_ref"] * (1.0 + delta)].copy()
            if eligible.empty:
                continue
            eligible = eligible.sort_values(["budget_ratio", "chunk_fraction"])
            grouped = eligible.groupby(["dataset", "model", "sampler", "ensemble_method"], as_index=False).first()
            grouped["delta"] = delta
            minimal_rows.extend(grouped.to_dict(orient="records"))
        return pd.DataFrame(minimal_rows)


def build_rmt_report_tables(
    run_records: Sequence[Mapping[str, Any]],
    output_dir: Path,
    reference_metrics: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    return RMTReportTableBuilder().build_report_tables(run_records, output_dir, reference_metrics)
