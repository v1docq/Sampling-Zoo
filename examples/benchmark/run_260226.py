from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from benchmark_sampling_strategies import make_strategies
from bechmark_models import make_model_pool

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from benchmark_datasets import DatasetBundle, load_dataset
from benchmark_logging import BenchmarkLogger
from benchmark_runner import SpecialStrategyBenchmarkRunner
from benchmark_viz import plot_metrics_vs_budget,plot_informative_overlap,plot_final_summary_table
DEFAULT_BUDGET_RATIOS: tuple[float, ...] = (0.01, 0.05, 0.10, 0.20)

AMLB_CATEGORY_PROFILES: dict[str, tuple[str, ...]] = {
    "small_samples_many_classes": ("amlb_optdigits", "amlb_vehicle"),
    "large_samples_binary": ("amlb_adult", "amlb_covertype"),
    "balanced_multiclass": ("amlb_mfeat_factors", "amlb_segment"),
}


@dataclass(frozen=True)
class BenchmarkRunConfig:
    full_benchmark: bool = True
    include_amlb: bool = False
    amlb_categories: Sequence[str] | None = ("small_samples_many_classes",)
    budget_ratios: Sequence[float] = DEFAULT_BUDGET_RATIOS
    seed: int = 42
    enable_diagnostic_plots: bool = False
    show_progress: bool = True


class BudgetPolicy:
    def __init__(self, seed: int) -> None:
        self.seed = seed

    @staticmethod
    def _normalize_scores(
        raw_scores: Sequence[float] | None,
        informative_indices: np.ndarray,
        train_size: int,
    ) -> np.ndarray | None:
        if raw_scores is None:
            return None

        scores = np.asarray(raw_scores, dtype=float).reshape(-1)
        if scores.shape[0] == informative_indices.shape[0]:
            return scores

        if scores.shape[0] == train_size:
            return scores[informative_indices]
        return None

    @staticmethod
    def _select_top_k_by_importance(
        informative_indices: np.ndarray,
        informative_scores: np.ndarray | None,
        budget_size: int,
    ) -> np.ndarray:
        if informative_indices.size <= budget_size:
            return informative_indices
        if informative_scores is None:
            return informative_indices[:budget_size]
        ranked_positions = np.argsort(-informative_scores, kind="mergesort")
        return informative_indices[ranked_positions[:budget_size]]

    def apply(
        self,
        informative_indices: Sequence[int],
        informative_scores: Sequence[float] | None,
        train_size: int,
        budget_ratio: float,
    ) -> dict[str, Any]:
        informative = np.unique(np.asarray(informative_indices, dtype=int))
        informative = informative[(informative >= 0) & (informative < train_size)]
        if informative.size == 0:
            raise ValueError("Sampling strategy returned no informative indices.")

        budget_size = max(1, int(round(train_size * budget_ratio)))
        budget_size = min(budget_size, train_size)

        normalized_scores = self._normalize_scores(informative_scores, informative, train_size)
        selected = self._select_top_k_by_importance(informative, normalized_scores, budget_size)
        action = "truncate_to_top_k" if informative.size > budget_size else "keep_informative"

        if selected.size < budget_size:
            rng = np.random.default_rng(self.seed)
            remaining = np.setdiff1d(np.arange(train_size, dtype=int), selected, assume_unique=False)
            needed = min(budget_size - selected.size, remaining.size)
            if needed > 0:
                filled = rng.choice(remaining, size=needed, replace=False)
                selected = np.concatenate([selected, filled])
                action = "top_up_with_raw_samples"

        selected = np.unique(selected)
        if selected.size > budget_size:
            selected = selected[:budget_size]

        return {
            "sample_indices": selected,
            "budget_size": int(budget_size),
            "budget_ratio": float(budget_ratio),
            "informative_size": int(informative.size),
            "policy_action": action,
        }


class StrategySuiteFactory:
    def __init__(self, seed: int, budget_ratios: Sequence[float]) -> None:
        self.seed = seed
        self.budget_ratios = budget_ratios
        self.policy = BudgetPolicy(seed=seed)

    @staticmethod
    def strategy_base_name(strategy_name: str) -> str:
        budget_suffix = "__budget_"
        if budget_suffix in strategy_name:
            return strategy_name.split(budget_suffix, 1)[0]
        return strategy_name

    def _with_budget_variants(
        self,
        strategy_name: str,
        strategy_fn: Callable[[DatasetBundle], Mapping[str, Any]],
    ) -> dict[str, Callable[[DatasetBundle], dict[str, Any]]]:
        def _runner(bundle: DatasetBundle, budget_ratio: float) -> dict[str, Any]:
            result = dict(strategy_fn(bundle))
            raw_indices = np.asarray(result.get("sample_indices", []), dtype=int)
            raw_scores = result.get("sample_scores")

            budgeted = self.policy.apply(
                informative_indices=raw_indices,
                informative_scores=raw_scores,
                train_size=bundle.y_train.shape[0],
                budget_ratio=budget_ratio,
            )

            result["sample_indices"] = budgeted["sample_indices"]
            result["strategy_params"] = {
                **(result.get("strategy_params") or {}),
                "budget_ratio": budgeted["budget_ratio"],
                "budget_size": budgeted["budget_size"],
                "informative_size": budgeted["informative_size"],
                "budget_policy_action": budgeted["policy_action"],
            }
            result["extra"] = {
                **(result.get("extra") or {}),
                "informative_indices": np.unique(raw_indices).tolist(),
                "budget_ratio": budgeted["budget_ratio"],
                "budget_size": budgeted["budget_size"],
            }
            return result

        wrapped: dict[str, Callable[[DatasetBundle], dict[str, Any]]] = {}
        for ratio in self.budget_ratios:
            ratio_tag = f"{int(round(ratio * 100)):02d}"

            def _factory(bundle: DatasetBundle, current_ratio: float = ratio):
                return _runner(bundle, current_ratio)

            wrapped[f"{strategy_name}__budget_{ratio_tag}"] = _factory
        return wrapped

    @staticmethod
    def _full_dataset_strategy(bundle: DatasetBundle) -> dict[str, Any]:
        full_indices = np.arange(bundle.y_train.shape[0], dtype=int)
        return {
            "sample_indices": full_indices,
            "strategy_params": {
                "budget_ratio": 1.0,
                "budget_size": int(full_indices.size),
                "informative_size": int(full_indices.size),
                "budget_policy_action": "full_dataset_baseline",
            },
            "extra": {"informative_indices": full_indices.tolist(), "budget_ratio": 1.0},
        }

    def build(self) -> dict[str, Callable[[DatasetBundle], dict[str, Any]]]:
        strategy_pool = make_strategies(seed=self.seed)
        wrapped: dict[str, Callable[[DatasetBundle], dict[str, Any]]] = {}
        for strategy_name, strategy_fn in strategy_pool.items():
            wrapped.update(self._with_budget_variants(strategy_name, strategy_fn))
        wrapped["full_dataset"] = self._full_dataset_strategy
        return wrapped


class DatasetCatalog:
    def __init__(self, category_profiles: Mapping[str, Sequence[str]]) -> None:
        self.category_profiles = category_profiles

    def resolve_names(self, full_benchmark: bool, include_amlb: bool, amlb_categories: Sequence[str] | None) -> list[str]:
        dataset_names = ["mixed_hard"]
        if full_benchmark:
            dataset_names = ["high_cardinality_categorical", "large_numeric", "mixed_hard"]
        if include_amlb:
            dataset_names.extend(["amlb_adult", "amlb_covertype"])

        for category in amlb_categories or []:
            datasets_for_category = self.category_profiles.get(category)
            if not datasets_for_category:
                raise ValueError(f"Unknown AMLB category: {category}. Available: {sorted(self.category_profiles)}")
            dataset_names.extend(datasets_for_category)

        return list(dict.fromkeys(dataset_names))

    def load(self, dataset_names: Sequence[str], seed: int, show_progress: bool) -> list[DatasetBundle]:
        datasets: list[DatasetBundle] = []
        for dataset_name in tqdm(dataset_names, desc="Load datasets", disable=not show_progress):
            try:
                datasets.append(load_dataset(dataset_name, seed=seed))
            except Exception as err:
                print(f"[WARN] dataset {dataset_name} is skipped: {err}")
        return datasets


class BenchmarkReportBuilder:
    @staticmethod
    def _with_enriched_dimensions(df: pd.DataFrame) -> pd.DataFrame:
        enriched = df.copy()
        enriched["strategy_base"] = enriched["strategy"].map(StrategySuiteFactory.strategy_base_name)
        enriched["budget_ratio"] = pd.to_numeric(enriched.get("strategy_params.budget_ratio"), errors="coerce")
        enriched["budget_ratio"] = enriched["budget_ratio"].fillna(pd.to_numeric(enriched.get("extra.budget_ratio"), errors="coerce"))
        enriched["budget_percent"] = (enriched["budget_ratio"] * 100.0).round(2)
        enriched["model"] = enriched["strategy"].str.rsplit("__", n=1).str[-1]
        return enriched

    def build_tables(self, run_records: Iterable[Mapping[str, Any]], output_dir: Path) -> pd.DataFrame:
        df = pd.json_normalize(list(run_records), sep=".")
        df = self._with_enriched_dimensions(df)

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

    def plot_metrics_vs_budget(self, df: pd.DataFrame, output_dir: Path) -> None:
        metrics = ["model_metrics.f1_macro", "model_metrics.roc_auc", "timings_sec.fit", "sample_stats.sample_size"]
        labels = {
            "model_metrics.f1_macro": "F1 macro",
            "model_metrics.roc_auc": "ROC-AUC",
            "timings_sec.fit": "Training time (sec)",
            "sample_stats.sample_size": "Sample size",
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
                    fig_rel.savefig(output_dir / f"metric_budget_relative__{dataset_name}__{metric.replace('.', '_')}.png", dpi=150)
                    plt.close(fig_rel)

    @staticmethod
    def _parse_index_collection(value: Any) -> set[int]:
        if isinstance(value, list):
            return {int(v) for v in value}
        return set()

    def plot_informative_overlap(self, df: pd.DataFrame, output_dir: Path) -> None:
        plot_informative_overlap(df,output_dir)

    def plot_optimal_strategy_overview(self, df: pd.DataFrame, output_dir: Path) -> None:
        metrics = ["model_metrics.f1_macro", "model_metrics.roc_auc", "timings_sec.fit", "sample_stats.sample_size"]
        metric_labels = {
            "model_metrics.f1_macro": "F1 macro",
            "model_metrics.roc_auc": "ROC-AUC",
            "timings_sec.fit": "Training time (sec)",
            "sample_stats.sample_size": "Sample size",
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

    def plot_final_summary_table(self, df: pd.DataFrame, output_dir: Path) -> None:
        plot_final_summary_table(df,output_dir)



class BenchmarkOrchestrator:
    def __init__(self, config: BenchmarkRunConfig) -> None:
        self.config = config
        self.dataset_catalog = DatasetCatalog(AMLB_CATEGORY_PROFILES)
        self.strategy_factory = StrategySuiteFactory(seed=config.seed, budget_ratios=config.budget_ratios)
        self.report_builder = BenchmarkReportBuilder()

    def run(self) -> Path:
        base_dir = Path(__file__).resolve().parent
        run_id = f"run_260226_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger = BenchmarkLogger(run_id=run_id, artifacts_root=base_dir / "results")

        dataset_names = self.dataset_catalog.resolve_names(
            full_benchmark=self.config.full_benchmark,
            include_amlb=self.config.include_amlb,
            amlb_categories=self.config.amlb_categories,
        )
        datasets = self.dataset_catalog.load(dataset_names, seed=self.config.seed, show_progress=self.config.show_progress)
        if not datasets:
            raise RuntimeError("No datasets available for benchmark run.")

        strategy_pool = self.strategy_factory.build()
        model_pool = make_model_pool(seed=self.config.seed)

        run_records: list[dict[str, Any]] = []
        model_iter = tqdm(model_pool.items(), total=len(model_pool), desc="Models", disable=not self.config.show_progress)
        for model_name, model in model_iter:
            runner = SpecialStrategyBenchmarkRunner(
                logger=logger,
                enable_diagnostic_plots=self.config.enable_diagnostic_plots,
                show_progress=self.config.show_progress,
            )
            model_tagged_strategies = {f"{name}__{model_name}": fn for name, fn in strategy_pool.items()}
            run_records.extend(runner.run(datasets=datasets, strategies=model_tagged_strategies, base_model=model))

        enriched_df = self.report_builder.build_tables(run_records, logger.paths.root)
        self.report_builder.plot_metrics_vs_budget(enriched_df, logger.paths.plots)
        self.report_builder.plot_informative_overlap(enriched_df, logger.paths.plots)
        self.report_builder.plot_optimal_strategy_overview(enriched_df, logger.paths.plots)
        self.report_builder.plot_final_summary_table(enriched_df, logger.paths.plots)

        run_meta = {
            "run_id": logger.run_id,
            "output_dir": str(logger.paths.root),
            "datasets": [dataset.name for dataset in datasets],
            "strategies": sorted({StrategySuiteFactory.strategy_base_name(name) for name in strategy_pool}),
            "models": list(model_pool.keys()),
            "budget_ratios": [float(ratio) for ratio in self.config.budget_ratios],
            "amlb_categories": list(self.config.amlb_categories or []),
            "records": len(run_records),
            "enable_diagnostic_plots": self.config.enable_diagnostic_plots,
        }
        (logger.paths.root / "run_meta.json").write_text(json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8")

        print(f"Benchmark completed. Artifacts: {logger.paths.root}")
        return logger.paths.root


# Backward-compatible wrappers used in tests and by external code.
def _apply_budget_policy(
    informative_indices: Sequence[int],
    informative_scores: Sequence[float] | None,
    train_size: int,
    budget_ratio: float,
    seed: int,
) -> dict[str, Any]:
    return BudgetPolicy(seed=seed).apply(
        informative_indices=informative_indices,
        informative_scores=informative_scores,
        train_size=train_size,
        budget_ratio=budget_ratio,
    )


def resolve_datasets(full_benchmark: bool, include_amlb: bool, amlb_categories: Sequence[str] | None) -> list[str]:
    return DatasetCatalog(AMLB_CATEGORY_PROFILES).resolve_names(
        full_benchmark=full_benchmark,
        include_amlb=include_amlb,
        amlb_categories=amlb_categories,
    )


def run_bench_pipeline(
    full_benchmark: bool = True,
    include_amlb: bool = False,
    amlb_categories: Sequence[str] | None = ("small_samples_many_classes",),
    budget_ratios: Sequence[float] = DEFAULT_BUDGET_RATIOS,
    enable_diagnostic_plots: bool = False,
    show_progress: bool = True,
) -> None:
    config = BenchmarkRunConfig(
        full_benchmark=full_benchmark,
        include_amlb=include_amlb,
        amlb_categories=amlb_categories,
        budget_ratios=budget_ratios,
        enable_diagnostic_plots=enable_diagnostic_plots,
        show_progress=show_progress,
    )
    BenchmarkOrchestrator(config).run()


if __name__ == "__main__":
    run_bench_pipeline()
