from benchmark_runner import _with_enriched_dimensions

import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

def build_summary_tables(run_records: Iterable[Mapping[str, Any]], output_dir: Path) -> pd.DataFrame:
    df = pd.json_normalize(list(run_records), sep=".")
    df = _with_enriched_dimensions(df)

    df.to_csv(output_dir / "benchmark_runs.csv", index=False)
    (output_dir / "benchmark_runs.json").write_text(df.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")

    summary = (
        df.groupby(["dataset", "model", "strategy_base", "budget_percent"], as_index=False)
        .agg(
            roc_auc=("model_metrics.roc_auc", "mean"),
            accuracy=("model_metrics.accuracy", "mean"),
            f1_macro=("model_metrics.f1_macro", "mean"),
            f1_weighted=("model_metrics.f1_weighted", "mean"),
            fit_sec=("timings_sec.fit", "mean"),
            sample_sec=("timings_sec.sample", "mean"),
            infer_sec=("timings_sec.inference", "mean"),
            sample_size=("sample_stats.sample_size", "mean"),
        )
        .sort_values(["dataset", "model", "budget_percent", "f1_macro"], ascending=[True, True, True, False])
    )
    summary.to_csv(output_dir / "summary_by_strategy_budget.csv", index=False)

    final_summary = (
        summary.sort_values(["dataset", "model", "strategy_base", "budget_percent"])
        .groupby(["dataset", "model", "strategy_base"], as_index=False)
        .tail(1)
        .sort_values(["dataset", "model", "f1_macro"], ascending=[True, True, False])
    )
    final_summary.to_csv(output_dir / "summary_final_table.csv", index=False)
    return df


def plot_metrics_vs_budget(df: pd.DataFrame, output_dir: Path) -> None:
    metrics = ["model_metrics.f1_macro", "model_metrics.roc_auc", "timings_sec.fit", "sample_stats.sample_size"]
    labels = {
        "model_metrics.f1_macro": "F1 macro",
        "model_metrics.roc_auc": "ROC-AUC",
        "timings_sec.fit": "Training time (sec)",
        "sample_stats.sample_size": "Sample size",
    }

    filtered = df[df["budget_percent"].notna()].copy()
    filtered = filtered[filtered["strategy_base"] != "full_dataset"]
    if filtered.empty:
        return

    for dataset_name in sorted(filtered["dataset"].unique()):
        subset = filtered[filtered["dataset"] == dataset_name]
        for metric in metrics:
            grouped = (
                subset.groupby(["strategy_base", "budget_percent"], as_index=False)[metric]
                .mean()
                .sort_values("budget_percent")
            )
            fig, ax = plt.subplots(figsize=(9, 5))
            for strategy, strategy_data in grouped.groupby("strategy_base"):
                ax.plot(
                    strategy_data["budget_percent"],
                    strategy_data[metric],
                    marker="o",
                    linewidth=2,
                    label=strategy,
                )

            baseline = df[(df["dataset"] == dataset_name) & (df["strategy_base"] == "full_dataset")]
            if not baseline.empty and metric in baseline:
                baseline_value = float(pd.to_numeric(baseline[metric], errors="coerce").mean())
                if np.isfinite(baseline_value):
                    ax.axhline(baseline_value, linestyle="--", color="black", alpha=0.7, label="full_dataset")

            ax.set_title(f"{dataset_name}: {labels[metric]} vs data budget")
            ax.set_xlabel("Data budget (%)")
            ax.set_ylabel(labels[metric])
            ax.grid(alpha=0.25)
            ax.legend(loc="best", fontsize=8)
            fig.tight_layout()
            target = output_dir / f"metric_budget__{dataset_name}__{metric.replace('.', '_')}.png"
            fig.savefig(target, dpi=150)
            plt.close(fig)


def _parse_index_collection(value: Any) -> set[int]:
    if isinstance(value, list):
        return {int(v) for v in value}
    return set()


def plot_informative_overlap(df: pd.DataFrame, output_dir: Path) -> None:
    records = []
    for _, row in df.iterrows():
        records.append(
            {
                "dataset": row.get("dataset"),
                "strategy_base": row.get("strategy_base"),
                "informative_indices": _parse_index_collection(row.get("extra.informative_indices")),
            }
        )

    overlap_df = pd.DataFrame(records)
    overlap_df = overlap_df[overlap_df["strategy_base"] != "full_dataset"]
    if overlap_df.empty:
        return

    for dataset_name in sorted(overlap_df["dataset"].dropna().unique()):
        ds = overlap_df[overlap_df["dataset"] == dataset_name]
        by_strategy = ds.groupby("strategy_base")["informative_indices"].first()
        strategies = list(by_strategy.index)
        if len(strategies) < 2:
            continue

        matrix = np.zeros((len(strategies), len(strategies)), dtype=float)
        for i, left in enumerate(strategies):
            for j, right in enumerate(strategies):
                left_set = by_strategy[left]
                right_set = by_strategy[right]
                denom = len(left_set | right_set)
                matrix[i, j] = len(left_set & right_set) / denom if denom else 1.0

        fig, ax = plt.subplots(figsize=(7, 6))
        image = ax.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0)
        ax.set_xticks(np.arange(len(strategies)), labels=strategies, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(strategies)), labels=strategies)
        ax.set_title(f"{dataset_name}: informative sample overlap (Jaccard)")
        fig.colorbar(image, ax=ax, label="Jaccard similarity")
        fig.tight_layout()
        fig.savefig(output_dir / f"informative_overlap__{dataset_name}.png", dpi=150)
        plt.close(fig)


def plot_final_summary_table(df: pd.DataFrame, output_dir: Path) -> None:
    summary = (
        df.sort_values(["dataset", "model", "strategy_base", "budget_percent"]).groupby(["dataset", "model", "strategy_base"], as_index=False).tail(1)
    )
    if summary.empty:
        return

    columns = [
        "dataset",
        "model",
        "strategy_base",
        "budget_percent",
        "model_metrics.f1_macro",
        "model_metrics.roc_auc",
        "timings_sec.fit",
        "timings_sec.sample",
    ]
    table = summary[columns].copy().sort_values(["dataset", "model", "model_metrics.f1_macro"], ascending=[True, True, False])
    table.columns = ["dataset", "model", "strategy", "budget_%", "f1_macro", "roc_auc", "fit_sec", "sample_sec"]
    table = table.round({"budget_%": 2, "f1_macro": 4, "roc_auc": 4, "fit_sec": 3, "sample_sec": 3})

    fig, ax = plt.subplots(figsize=(14, max(4, 0.35 * len(table) + 1)))
    ax.axis("off")
    rendered = ax.table(cellText=table.values, colLabels=table.columns, loc="center")
    rendered.auto_set_font_size(False)
    rendered.set_fontsize(8)
    rendered.scale(1.0, 1.25)
    ax.set_title("Final benchmark summary")
    fig.tight_layout()
    fig.savefig(output_dir / "final_summary_table.png", dpi=170)
    plt.close(fig)