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
    metrics = ["model_metrics.f1_macro", "model_metrics.roc_auc", "timings_sec.fit"]
    labels = {
        "model_metrics.f1_macro": "F1 macro",
        "model_metrics.roc_auc": "ROC-AUC",
        "timings_sec.fit": "Training time (sec)",
            }

    filtered = df[df["budget_percent"].notna()].copy()
    full_baseline_mask = filtered["strategy_base"].fillna("").str.startswith("full_dataset")
    strategy_rows = filtered[~full_baseline_mask].copy()
    baseline_rows = filtered[full_baseline_mask].copy()

    strategy_rows["budget_percent"] = pd.to_numeric(strategy_rows["budget_percent"], errors="coerce")
    strategy_rows["sample_stats.sample_size"] = pd.to_numeric(strategy_rows.get("sample_stats.sample_size"), errors="coerce")
    strategy_rows["strategy_params.budget_size"] = pd.to_numeric(strategy_rows.get("strategy_params.budget_size"), errors="coerce")
    strategy_rows["strategy_params.informative_size"] = pd.to_numeric(strategy_rows.get("strategy_params.informative_size"), errors="coerce")

    strategy_rows["budget_size_effective"] = strategy_rows["strategy_params.budget_size"].fillna(strategy_rows["sample_stats.sample_size"])
    strategy_rows["effective_to_budget_percent"] = (
        100.0 * strategy_rows["sample_stats.sample_size"] / strategy_rows["budget_size_effective"]
    ).replace([np.inf, -np.inf], np.nan)
    strategy_rows["informative_to_budget_percent"] = (
        100.0 * strategy_rows["strategy_params.informative_size"] / strategy_rows["budget_size_effective"]
    ).replace([np.inf, -np.inf], np.nan)

    if strategy_rows.empty:
        return

    for dataset_name in sorted(strategy_rows["dataset"].dropna().unique()):
        subset = strategy_rows[strategy_rows["dataset"] == dataset_name]
        dataset_baseline = baseline_rows[baseline_rows["dataset"] == dataset_name]

        diagnostics = (
            subset.groupby(["strategy_base", "budget_percent"], as_index=False)
            .agg(
                effective_size=("sample_stats.sample_size", "mean"),
                budget_size=("budget_size_effective", "mean"),
                informative_size=("strategy_params.informative_size", "mean"),
                effective_to_budget_pct=("effective_to_budget_percent", "mean"),
                informative_to_budget_pct=("informative_to_budget_percent", "mean"),
            )
            .sort_values(["budget_percent", "strategy_base"])
        )

        for metric in metrics:
            grouped = (
                subset.groupby(["strategy_base", "budget_percent"], as_index=False)[metric]
                .mean()
                .sort_values("budget_percent")
            )

            baseline_metric_value = float("nan")
            if not dataset_baseline.empty and metric in dataset_baseline:
                baseline_metric_value = float(pd.to_numeric(dataset_baseline[metric], errors="coerce").mean())

            fig, (ax, ax_info) = plt.subplots(
                nrows=2,
                figsize=(11, 7),
                gridspec_kw={"height_ratios": [4.4, 1.8]},
                sharex=False,
            )

            for strategy, strategy_data in grouped.groupby("strategy_base"):
                ax.plot(strategy_data["budget_percent"], strategy_data[metric], marker="o", linewidth=2, label=strategy)

            if np.isfinite(baseline_metric_value):
                ax.axhline(
                    baseline_metric_value,
                    linestyle="--",
                    color="black",
                    alpha=0.7,
                    label="full_dataset baseline",
                )

            ax.set_title(f"{dataset_name}: {labels[metric]} vs data budget")
            ax.set_xlabel("Data budget (%)")
            ax.set_ylabel(labels[metric])
            ax.grid(alpha=0.25)
            ax.legend(loc="best", fontsize=8)

            diag_metric = diagnostics.copy()
            diag_metric["budget_percent"] = diag_metric["budget_percent"].map(lambda x: f"{x:.0f}%" if np.isfinite(x) else "NA")
            diag_metric["eff_vs_budget"] = diag_metric.apply(
                lambda r: f"{r['effective_size']:.0f}/{r['budget_size']:.0f} ({r['effective_to_budget_pct']:.0f}%)"
                if np.isfinite(r["effective_to_budget_pct"]) and np.isfinite(r["budget_size"]) else "NA",
                axis=1,
            )
            diag_metric["inf_vs_budget"] = diag_metric.apply(
                lambda r: f"{r['informative_size']:.0f}/{r['budget_size']:.0f} ({r['informative_to_budget_pct']:.0f}%)"
                if np.isfinite(r["informative_to_budget_pct"]) and np.isfinite(r["budget_size"]) else "NA",
                axis=1,
            )
            diag_metric = diag_metric[["strategy_base", "budget_percent", "eff_vs_budget", "inf_vs_budget"]]
            diag_metric.columns = ["Strategy", "Budget", "Effective set / budget", "Informative candidates / budget"]

            ax_info.axis("off")
            info_table = ax_info.table(
                cellText=diag_metric.values,
                colLabels=diag_metric.columns,
                loc="center",
            )
            info_table.auto_set_font_size(False)
            info_table.set_fontsize(8)
            info_table.scale(1.0, 1.2)
            ax_info.set_title("Selection diagnostics (ratios are relative to budget size)", fontsize=9, pad=8)

            fig.tight_layout()
            fig.savefig(output_dir / f"metric_budget__{dataset_name}__{metric.replace('.', '_')}.png", dpi=150)
            plt.close(fig)

            if np.isfinite(baseline_metric_value) and abs(baseline_metric_value) > 1e-12:
                relative = grouped.copy()
                relative["relative_change_percent"] = 100.0 * (relative[metric] - baseline_metric_value) / abs(baseline_metric_value)

                budgets = sorted(relative["budget_percent"].dropna().unique())
                strategies = sorted(relative["strategy_base"].dropna().unique())
                x = np.arange(len(budgets))
                width = 0.8 / max(1, len(strategies))

                fig_rel, ax_rel = plt.subplots(figsize=(10, 5))
                for idx, strategy in enumerate(strategies):
                    strategy_data = relative[relative["strategy_base"] == strategy]
                    values = []
                    for budget in budgets:
                        point = strategy_data[strategy_data["budget_percent"] == budget]
                        values.append(float(point["relative_change_percent"].iloc[0]) if not point.empty else np.nan)
                    offsets = x - 0.4 + width / 2 + idx * width
                    ax_rel.bar(offsets, values, width=width, label=strategy, alpha=0.9)

                ax_rel.axhline(0.0, linestyle="--", color="black", alpha=0.7)
                ax_rel.set_xticks(x)
                ax_rel.set_xticklabels([f"{budget:.0f}%" for budget in budgets])
                ax_rel.set_title(f"{dataset_name}: relative {labels[metric]} change vs full dataset")
                ax_rel.set_xlabel("Data budget (%)")
                ax_rel.set_ylabel("Relative change (%)")
                ax_rel.grid(alpha=0.25, axis="y")
                ax_rel.legend(loc="best", fontsize=8)
                fig_rel.tight_layout()
                rel_target = output_dir / f"metric_budget_relative__{dataset_name}__{metric.replace('.', '_')}.png"
                fig_rel.savefig(rel_target, dpi=150)
                plt.close(fig_rel)


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
        colorbar = fig.colorbar(image, ax=ax, label="Jaccard similarity")
        colorbar.ax.text(
            0.5,
            -0.08,
            "low ≈ weak overlap, high ≈ strong overlap",
            transform=colorbar.ax.transAxes,
            ha="center",
            va="top",
            fontsize=8,
        )
        fig.tight_layout()
        fig.savefig(output_dir / f"informative_overlap__{dataset_name}.png", dpi=150)
        plt.close(fig)


def plot_optimal_strategy_overview(df: pd.DataFrame, output_dir: Path) -> None:
    metrics = ["model_metrics.f1_macro", "model_metrics.roc_auc", "timings_sec.fit"]
    metric_labels = {
        "model_metrics.f1_macro": "F1 macro",
        "model_metrics.roc_auc": "ROC-AUC",
        "timings_sec.fit": "Training time (sec)",
            }
    higher_is_better = {
        "model_metrics.f1_macro": True,
        "model_metrics.roc_auc": True,
        "timings_sec.fit": False,
        "sample_stats.sample_size": False,
    }

    filtered = df[df["budget_percent"].notna()].copy()
    filtered["budget_percent"] = pd.to_numeric(filtered["budget_percent"], errors="coerce")
    filtered = filtered[filtered["dataset"].notna()]
    if filtered.empty:
        return

    datasets = sorted(filtered["dataset"].unique())
    delta_matrix = np.full((len(metrics), len(datasets)), np.nan)
    annotation_matrix = [["" for _ in datasets] for _ in metrics]

    for col, dataset_name in enumerate(datasets):
        ds = filtered[filtered["dataset"] == dataset_name]
        baseline = ds[ds["strategy_base"].fillna("").str.startswith("full_dataset")]
        candidates = ds[~ds["strategy_base"].fillna("").str.startswith("full_dataset")]
        if candidates.empty:
            continue

        for row, metric in enumerate(metrics):
            if metric not in ds:
                continue

            baseline_value = float(pd.to_numeric(baseline[metric], errors="coerce").mean()) if not baseline.empty else np.nan
            grouped = (
                candidates.groupby(["strategy_base", "budget_percent"], as_index=False)[metric]
                .mean()
                .dropna(subset=[metric])
            )
            if grouped.empty:
                continue

            idx = grouped[metric].idxmax() if higher_is_better[metric] else grouped[metric].idxmin()
            best = grouped.loc[idx]
            best_value = float(best[metric])

            if np.isfinite(baseline_value) and abs(baseline_value) > 1e-12:
                if higher_is_better[metric]:
                    delta = 100.0 * (best_value - baseline_value) / abs(baseline_value)
                else:
                    delta = 100.0 * (baseline_value - best_value) / abs(baseline_value)
                delta_matrix[row, col] = delta

            annotation_matrix[row][col] = f"{best['strategy_base']}@{best['budget_percent']:.0f}%"

    fig, ax = plt.subplots(figsize=(max(10, len(datasets) * 1.7), 6.5))
    heatmap = ax.imshow(delta_matrix, cmap="RdYlGn", aspect="auto")

    ax.set_xticks(np.arange(len(datasets)), labels=datasets, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(metrics)), labels=[metric_labels[m] for m in metrics])
    ax.set_title("Best strategy by dataset/metric vs full-dataset baseline (improvement %)\n(positive is better)")

    for i in range(len(metrics)):
        for j in range(len(datasets)):
            delta = delta_matrix[i, j]
            label = annotation_matrix[i][j]
            if not label:
                continue
            text = f"{label}"
            if np.isfinite(delta):
                text += f"\n{delta:+.1f}%"
            ax.text(j, i, text, ha="center", va="center", fontsize=7, color="black")

    fig.colorbar(heatmap, ax=ax, label="Improvement vs baseline (%)")
    fig.tight_layout()
    fig.savefig(output_dir / "optimal_strategy_overview.png", dpi=160)
    plt.close(fig)


def plot_time_reinvestment_scenario(df: pd.DataFrame, output_dir: Path) -> None:
    quality_metric = "model_metrics.f1_macro"
    filtered = df[df["budget_percent"].notna()].copy()
    filtered[quality_metric] = pd.to_numeric(filtered.get(quality_metric), errors="coerce")
    filtered["timings_sec.fit"] = pd.to_numeric(filtered.get("timings_sec.fit"), errors="coerce")
    filtered = filtered[filtered["dataset"].notna()]
    if filtered.empty:
        return

    rows: list[dict[str, Any]] = []
    for dataset_name in sorted(filtered["dataset"].unique()):
        ds = filtered[filtered["dataset"] == dataset_name]
        baseline = ds[ds["strategy_base"].fillna("").str.startswith("full_dataset")]
        candidates = ds[~ds["strategy_base"].fillna("").str.startswith("full_dataset")]
        if baseline.empty or candidates.empty:
            continue

        baseline_quality = float(pd.to_numeric(baseline[quality_metric], errors="coerce").mean())
        baseline_fit = float(pd.to_numeric(baseline["timings_sec.fit"], errors="coerce").mean())
        if not np.isfinite(baseline_quality) or not np.isfinite(baseline_fit) or baseline_fit <= 0:
            continue

        grouped = (
            candidates.groupby(["strategy_base", "budget_percent"], as_index=False)
            .agg(
                quality=(quality_metric, "mean"),
                fit_sec=("timings_sec.fit", "mean"),
            )
            .dropna(subset=["quality", "fit_sec"])
        )
        for _, row in grouped.iterrows():
            fit_sec = float(row["fit_sec"])
            quality = float(row["quality"])
            if fit_sec <= 0:
                continue

            time_gain = max(0.0, baseline_fit - fit_sec)
            reinvest_multiplier = time_gain / fit_sec
            quality_gap = baseline_quality - quality
            recovered_share = 1.0 - np.exp(-reinvest_multiplier) if quality_gap > 0 else 0.0
            quality_after = quality + max(0.0, quality_gap) * recovered_share
            quality_after = min(quality_after, baseline_quality)

            rows.append(
                {
                    "dataset": dataset_name,
                    "strategy": row["strategy_base"],
                    "budget_percent": float(row["budget_percent"]),
                    "baseline_quality": baseline_quality,
                    "sampled_quality": quality,
                    "quality_after_reinvest": quality_after,
                    "quality_gap": quality_gap,
                    "recovered_share": recovered_share,
                    "baseline_fit_sec": baseline_fit,
                    "sampled_fit_sec": fit_sec,
                    "time_gain_sec": time_gain,
                    "additional_fit_equivalents": reinvest_multiplier,
                }
            )

    scenario_df = pd.DataFrame(rows)
    if scenario_df.empty:
        return

    scenario_df.to_csv(output_dir / "time_reinvestment_scenario.csv", index=False)
    summary = (
        scenario_df.groupby("dataset", as_index=False)
        .agg(
            sampled_quality=("sampled_quality", "max"),
            quality_after_reinvest=("quality_after_reinvest", "max"),
            baseline_quality=("baseline_quality", "mean"),
        )
        .sort_values("dataset")
    )

    x = np.arange(len(summary))
    width = 0.26
    fig, ax = plt.subplots(figsize=(max(11, len(summary) * 0.55), 5.2))
    ax.bar(x - width, summary["sampled_quality"], width=width, label="best sampled", alpha=0.9)
    ax.bar(x, summary["quality_after_reinvest"], width=width, label="hypothetical after reinvest", alpha=0.9)
    ax.bar(x + width, summary["baseline_quality"], width=width, label="full-dataset baseline", alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(summary["dataset"], rotation=35, ha="right")
    ax.set_title("Time-reinvestment scenario: can tuning time compensate quality drop?")
    ax.set_ylabel("F1 macro")
    ax.grid(alpha=0.25, axis="y")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "time_reinvestment_scenario.png", dpi=160)
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
