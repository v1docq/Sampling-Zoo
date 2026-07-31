"""Artifact construction for the RMT spiked synthetic benchmark."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


SUMMARY_GROUP_COLUMNS = (
    "noise_distribution",
    "view_strategy",
    "backend",
    "snr",
    "true_rank",
)
BACKEND_PAIR_COLUMNS = (
    "noise_distribution",
    "view_strategy",
    "snr",
    "true_rank",
    "seed",
)


class SpikedBenchmarkArtifactBuilder:
    """Build stable tables and a reader-facing report from leaf-run records."""

    def build(
        self,
        records: Sequence[Mapping[str, Any]],
        output_root: str | Path,
    ) -> dict[str, Path]:
        root = Path(output_root)
        metrics_dir = root / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        raw = self._build_raw_table(records)
        summary = self._build_signal_summary(raw)
        null_summary = self._build_null_summary(raw)
        backend_agreement = self._build_backend_agreement(raw)
        paths = self._write_tables(
            metrics_dir=metrics_dir,
            raw=raw,
            summary=summary,
            null_summary=null_summary,
            backend_agreement=backend_agreement,
        )
        paths["report"] = self._write_report(
            root=root,
            raw=raw,
            summary=summary,
            null_summary=null_summary,
            backend_agreement=backend_agreement,
        )
        return paths

    @staticmethod
    def _build_raw_table(
        records: Sequence[Mapping[str, Any]],
    ) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for record in records:
            row = dict(record)
            diagnostics = row.pop("diagnostics", {})
            row["diagnostics_available"] = bool(diagnostics)
            extra = row.pop("extra", {})
            row["extra_json"] = json.dumps(
                extra,
                ensure_ascii=False,
                sort_keys=True,
                default=_json_default,
            )
            rows.append(row)
        return pd.DataFrame(rows)

    @staticmethod
    def _build_signal_summary(raw: pd.DataFrame) -> pd.DataFrame:
        if raw.empty:
            return pd.DataFrame()
        successful = raw[
            (raw["status"] == "completed")
            & (pd.to_numeric(raw["true_rank"], errors="coerce") > 0)
        ].copy()
        if successful.empty:
            return pd.DataFrame()
        aggregations = {
            "runs": ("grid_key", "count"),
            "empirical_snr_mean": ("empirical_snr", "mean"),
            "standardized_empirical_snr_mean": (
                "standardized_empirical_snr",
                "mean",
            ),
            "selected_rank_mean": ("explained_estimated_rank", "mean"),
            "selected_rank_std": ("explained_estimated_rank", "std"),
            "selected_rank_abs_error_mean": (
                "explained_rank_absolute_error",
                "mean",
            ),
            "null_rank_mean": ("null_edge_estimated_rank", "mean"),
            "null_rank_abs_error_mean": (
                "null_edge_rank_absolute_error",
                "mean",
            ),
            "view_stability_rank_mean": (
                "view_stability_estimated_rank",
                "mean",
            ),
            "subspace_stability_rank_mean": (
                "subspace_stability_estimated_rank",
                "mean",
            ),
            "selected_subspace_recall_mean": (
                "selected_subspace_recall",
                "mean",
            ),
            "selected_subspace_precision_mean": (
                "selected_subspace_precision",
                "mean",
            ),
            "selected_subspace_f1_mean": (
                "selected_subspace_f1",
                "mean",
            ),
            "selected_subspace_max_angle_mean": (
                "selected_subspace_max_principal_angle_degrees",
                "mean",
            ),
            "fit_time_mean_sec": ("fit_time_sec", "mean"),
        }
        return (
            successful.groupby(
                list(SUMMARY_GROUP_COLUMNS),
                dropna=False,
                sort=True,
            )
            .agg(**aggregations)
            .reset_index()
        )

    @staticmethod
    def _build_null_summary(raw: pd.DataFrame) -> pd.DataFrame:
        if raw.empty:
            return pd.DataFrame()
        successful = raw[
            (raw["status"] == "completed")
            & (pd.to_numeric(raw["true_rank"], errors="coerce") == 0)
        ].copy()
        if successful.empty:
            return pd.DataFrame()
        group_columns = [
            "noise_distribution",
            "view_strategy",
            "backend",
        ]
        return (
            successful.groupby(group_columns, dropna=False, sort=True)
            .agg(
                runs=("grid_key", "count"),
                explained_false_positive_rate=(
                    "explained_rank_false_positive",
                    "mean",
                ),
                null_edge_false_positive_rate=(
                    "null_edge_rank_false_positive",
                    "mean",
                ),
                view_stability_false_positive_rate=(
                    "view_stability_rank_false_positive",
                    "mean",
                ),
                subspace_stability_false_positive_rate=(
                    "subspace_stability_rank_false_positive",
                    "mean",
                ),
                fit_time_mean_sec=("fit_time_sec", "mean"),
            )
            .reset_index()
        )

    @staticmethod
    def _build_backend_agreement(raw: pd.DataFrame) -> pd.DataFrame:
        if raw.empty or "backend" not in raw:
            return pd.DataFrame()
        successful = raw[raw["status"] == "completed"].copy()
        available = set(successful["backend"].astype(str))
        if not {"numpy", "torch"}.issubset(available):
            return pd.DataFrame()
        metrics = (
            "explained_estimated_rank",
            "null_edge_estimated_rank",
            "view_stability_estimated_rank",
            "subspace_stability_estimated_rank",
            "selected_subspace_recall",
            "selected_subspace_f1",
        )
        left = successful[successful["backend"] == "numpy"][
            [*BACKEND_PAIR_COLUMNS, *metrics]
        ]
        right = successful[successful["backend"] == "torch"][
            [*BACKEND_PAIR_COLUMNS, *metrics]
        ]
        paired = left.merge(
            right,
            on=list(BACKEND_PAIR_COLUMNS),
            how="inner",
            suffixes=("_numpy", "_torch"),
        )
        if paired.empty:
            return paired
        for metric in metrics:
            paired[f"{metric}_absolute_delta"] = (
                pd.to_numeric(paired[f"{metric}_numpy"], errors="coerce")
                - pd.to_numeric(paired[f"{metric}_torch"], errors="coerce")
            ).abs()
        return paired.sort_values(list(BACKEND_PAIR_COLUMNS)).reset_index(
            drop=True
        )

    def _write_tables(
        self,
        *,
        metrics_dir: Path,
        raw: pd.DataFrame,
        summary: pd.DataFrame,
        null_summary: pd.DataFrame,
        backend_agreement: pd.DataFrame,
    ) -> dict[str, Path]:
        paths = {
            "raw_csv": metrics_dir / "rmt_spiked_raw_runs.csv",
            "summary_csv": metrics_dir / "rmt_spiked_summary_by_snr.csv",
            "null_summary_csv": metrics_dir / "rmt_spiked_null_controls.csv",
            "backend_agreement_csv": (
                metrics_dir / "rmt_spiked_backend_agreement.csv"
            ),
        }
        self._atomic_write_csv(paths["raw_csv"], raw)
        self._atomic_write_csv(paths["summary_csv"], summary)
        self._atomic_write_csv(paths["null_summary_csv"], null_summary)
        self._atomic_write_csv(
            paths["backend_agreement_csv"],
            backend_agreement,
        )
        return paths

    def _write_report(
        self,
        *,
        root: Path,
        raw: pd.DataFrame,
        summary: pd.DataFrame,
        null_summary: pd.DataFrame,
        backend_agreement: pd.DataFrame,
    ) -> Path:
        report_path = root / "report.md"
        completed = (
            int((raw["status"] == "completed").sum())
            if not raw.empty and "status" in raw
            else 0
        )
        failed = (
            int((raw["status"] == "failed").sum())
            if not raw.empty and "status" in raw
            else 0
        )
        skipped = (
            int((raw["status"] == "skipped").sum())
            if not raw.empty and "status" in raw
            else 0
        )
        lines = [
            "# RMT spiked synthetic validation",
            "",
            "This report checks the spectral engine on controlled low-rank "
            "matrices before any production rank policy is changed.",
            "",
            "## Run status",
            "",
            f"- Leaf runs: {len(raw)}",
            f"- Completed: {completed}",
            f"- Failed: {failed}",
            f"- Skipped: {skipped}",
            "",
            "## Signal recovery by SNR",
            "",
            _markdown_table(
                summary,
                (
                    "noise_distribution",
                    "view_strategy",
                    "backend",
                    "snr",
                    "true_rank",
                    "selected_rank_mean",
                    "null_rank_mean",
                    "subspace_stability_rank_mean",
                    "selected_subspace_recall_mean",
                    "selected_subspace_f1_mean",
                ),
            ),
            "",
            "## Rank-zero controls",
            "",
            _markdown_table(
                null_summary,
                (
                    "noise_distribution",
                    "view_strategy",
                    "backend",
                    "runs",
                    "explained_false_positive_rate",
                    "null_edge_false_positive_rate",
                    "view_stability_false_positive_rate",
                    "subspace_stability_false_positive_rate",
                ),
            ),
            "",
            "## Matrix/Torch agreement",
            "",
            _backend_agreement_text(backend_agreement),
            "",
            "## Interpretation gates",
            "",
            "- The selected rank should increase only when signal becomes "
            "detectable; explained variance is expected to over-rank null data.",
            "- Null-edge and stability false-positive rates are evaluated on "
            "rank-zero controls and must be interpreted at the configured "
            "finite-resample quantile.",
            "- Selected-subspace recall measures how much of the true sample "
            "span is recovered and penalizes under-ranking.",
            "- Backend deltas separate numerical backend disagreement from "
            "statistical variability because paired runs reuse the same data seed.",
            "",
            "Production rank selection remains unchanged by this benchmark.",
        ]
        _atomic_write_text(report_path, "\n".join(lines) + "\n")
        return report_path

    @staticmethod
    def _atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)


def _backend_agreement_text(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No paired NumPy/Torch runs are available."
    delta_columns = [
        column for column in frame.columns if column.endswith("_absolute_delta")
    ]
    rows = [
        {
            "metric": column.removesuffix("_absolute_delta"),
            "mean_absolute_delta": float(
                pd.to_numeric(frame[column], errors="coerce").mean()
            ),
            "max_absolute_delta": float(
                pd.to_numeric(frame[column], errors="coerce").max()
            ),
            "paired_runs": len(frame),
        }
        for column in delta_columns
    ]
    return _markdown_table(
        pd.DataFrame(rows),
        (
            "metric",
            "mean_absolute_delta",
            "max_absolute_delta",
            "paired_runs",
        ),
    )


def _markdown_table(
    frame: pd.DataFrame,
    columns: Sequence[str],
) -> str:
    selected = [column for column in columns if column in frame.columns]
    if frame.empty or not selected:
        return "No completed records are available."
    headers = [column.replace("_", " ") for column in selected]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in selected) + "|",
    ]
    for _, row in frame.loc[:, selected].iterrows():
        lines.append(
            "| "
            + " | ".join(_format_markdown_value(row[column]) for column in selected)
            + " |"
        )
    return "\n".join(lines)


def _format_markdown_value(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "-"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4f}"
    return str(value).replace("|", "\\|")


def _atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
