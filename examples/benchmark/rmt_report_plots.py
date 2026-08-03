"""Diagnostic plots for the final RMT regression benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Sequence

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


@dataclass(frozen=True)
class RMTReportPlotBuilder:
    degradation_guides: Sequence[float] = (0.0, 0.01, 0.03, 0.05)
    dpi: int = 160

    def build_plots(
        self,
        raw_runs: pd.DataFrame,
        output_dir: Path,
    ) -> tuple[Path, ...]:
        output_dir.mkdir(parents=True, exist_ok=True)
        prepared = self._prepare_runs(raw_runs)
        if prepared.empty:
            return ()

        paths: list[Path] = []
        for (dataset, model), group in prepared.groupby(
            ["dataset", "model"],
            dropna=False,
        ):
            paths.extend(
                self._build_dataset_model_plots(
                    group,
                    dataset=str(dataset),
                    model=str(model),
                    output_dir=output_dir,
                )
            )
        return tuple(paths)

    @staticmethod
    def _prepare_runs(raw_runs: pd.DataFrame) -> pd.DataFrame:
        required = {
            "dataset",
            "model",
            "scenario_family",
            "budget_ratio",
            "rmse_drop",
        }
        if raw_runs.empty or not required.issubset(raw_runs.columns):
            return pd.DataFrame()
        prepared = raw_runs.copy()
        for column in (
            "budget_ratio",
            "rmse_drop",
            "fit_time",
            "inference_time",
            "tree_count_total",
            "max_tree_depth",
        ):
            if column in prepared.columns:
                prepared[column] = pd.to_numeric(
                    prepared[column],
                    errors="coerce",
                )
        return prepared[
            prepared["scenario_family"].notna()
            & prepared["budget_ratio"].notna()
            & prepared["rmse_drop"].notna()
        ]

    def _build_dataset_model_plots(
        self,
        runs: pd.DataFrame,
        *,
        dataset: str,
        model: str,
        output_dir: Path,
    ) -> list[Path]:
        stem = f"{self._slug(dataset)}__{self._slug(model)}"
        paths = [
            self._plot_degradation_curves(
                runs,
                output_dir / f"degradation__{stem}.png",
                dataset,
                model,
            ),
            self._plot_runtime_quality(
                runs,
                output_dir / f"runtime_quality__{stem}.png",
                dataset,
                model,
            ),
        ]
        complexity_path = self._plot_complexity(
            runs,
            output_dir / f"complexity__{stem}.png",
            dataset,
            model,
        )
        if complexity_path is not None:
            paths.append(complexity_path)
        return paths

    def _plot_degradation_curves(
        self,
        runs: pd.DataFrame,
        path: Path,
        dataset: str,
        model: str,
    ) -> Path:
        figure, axis = plt.subplots(figsize=(8.2, 5.0))
        for label, group in self._scenario_groups(runs):
            axis.plot(
                group["budget_ratio"] * 100.0,
                group["rmse_drop"] * 100.0,
                marker="o",
                linewidth=1.8,
                markersize=4.5,
                label=label,
            )
        for guide in self.degradation_guides:
            axis.axhline(
                guide * 100.0,
                color="#7b8794" if guide else "#263746",
                linestyle="--" if guide else "-",
                linewidth=0.8,
                alpha=0.65,
            )
        axis.set(
            xlabel="Training budget, %",
            ylabel="RMSE degradation vs reference, %",
            title=f"{dataset}: degradation curves ({model})",
        )
        self._finish_axis(axis)
        self._save(figure, path)
        return path

    def _plot_runtime_quality(
        self,
        runs: pd.DataFrame,
        path: Path,
        dataset: str,
        model: str,
    ) -> Path:
        figure, axis = plt.subplots(figsize=(8.2, 5.0))
        for label, group in self._scenario_groups(runs):
            valid = group[
                group.get("fit_time", pd.Series(index=group.index)).gt(0)
            ]
            if valid.empty:
                continue
            axis.scatter(
                valid["fit_time"],
                valid["rmse_drop"] * 100.0,
                s=35.0 + 0.8 * valid["budget_ratio"] * 100.0,
                alpha=0.8,
                label=label,
            )
        axis.axhline(0.0, color="#263746", linewidth=0.8)
        if (runs.get("fit_time", pd.Series(dtype=float)) > 0).any():
            axis.set_xscale("log")
        axis.set(
            xlabel="Fit time, seconds (log scale)",
            ylabel="RMSE degradation vs reference, %",
            title=f"{dataset}: runtime-quality trade-off ({model})",
        )
        self._finish_axis(axis)
        self._save(figure, path)
        return path

    def _plot_complexity(
        self,
        runs: pd.DataFrame,
        path: Path,
        dataset: str,
        model: str,
    ) -> Path | None:
        if "tree_count_total" not in runs.columns:
            return None
        selected = runs.dropna(subset=["tree_count_total"])
        if selected.empty:
            return None

        figure, axis = plt.subplots(figsize=(8.2, 5.0))
        for label, group in self._scenario_groups(selected):
            axis.plot(
                group["budget_ratio"] * 100.0,
                group["tree_count_total"],
                marker="o",
                linewidth=1.8,
                markersize=4.5,
                label=label,
            )
        axis.set(
            xlabel="Training budget, %",
            ylabel="Total trees across active models",
            title=f"{dataset}: ensemble complexity ({model})",
        )
        self._finish_axis(axis)
        self._save(figure, path)
        return path

    @staticmethod
    def _scenario_groups(
        runs: pd.DataFrame,
    ) -> list[tuple[str, pd.DataFrame]]:
        grouped: list[tuple[str, pd.DataFrame]] = []
        for scenario, frame in runs.groupby("scenario_family", dropna=False):
            aggregated = (
                frame.groupby("budget_ratio", as_index=False)
                .mean(numeric_only=True)
                .sort_values("budget_ratio")
            )
            label = str(scenario).split("__", maxsplit=1)[-1]
            grouped.append((label, aggregated))
        return grouped

    @staticmethod
    def _finish_axis(axis) -> None:
        axis.grid(axis="y", color="#d9e0e6", linewidth=0.7, alpha=0.8)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(
                handles,
                labels,
                frameon=False,
                fontsize=8,
                loc="best",
            )

    def _save(self, figure, path: Path) -> None:
        figure.tight_layout()
        figure.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(figure)

    @staticmethod
    def _slug(value: str) -> str:
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
        return slug or "unknown"
