from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from benchmark_sampling_strategies import make_strategies
from bechmark_models import make_model_pool

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from benchmark_datasets import DatasetBundle, load_dataset
from benchmark_logging import BenchmarkLogger
from benchmark_runner import SpecialStrategyBenchmarkRunner


DEFAULT_BUDGET_RATIOS: tuple[float, ...] = (0.01, 0.05, 0.10, 0.20)

AMLB_CATEGORY_PROFILES: dict[str, tuple[str, ...]] = {
    "small_samples_many_classes": ("amlb_optdigits", "amlb_vehicle"),
    "large_samples_binary": ("amlb_adult", "amlb_covertype"),
    "balanced_multiclass": ("amlb_mfeat_factors", "amlb_segment"),
}


def _strategy_base_name(strategy_name: str) -> str:
    budget_suffix = "__budget_"
    if budget_suffix in strategy_name:
        return strategy_name.split(budget_suffix, 1)[0]
    return strategy_name


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


def _apply_budget_policy(
    informative_indices: Sequence[int],
    informative_scores: Sequence[float] | None,
    train_size: int,
    budget_ratio: float,
    seed: int,
) -> dict[str, Any]:
    informative = np.unique(np.asarray(informative_indices, dtype=int))
    informative = informative[(informative >= 0) & (informative < train_size)]
    if informative.size == 0:
        raise ValueError("Sampling strategy returned no informative indices.")

    budget_size = max(1, int(round(train_size * budget_ratio)))
    budget_size = min(budget_size, train_size)

    normalized_scores = _normalize_scores(informative_scores, informative, train_size)

    selected = _select_top_k_by_importance(informative, normalized_scores, budget_size)
    action = "truncate_to_top_k" if informative.size > budget_size else "keep_informative"

    if selected.size < budget_size:
        rng = np.random.default_rng(seed)
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


def with_budget_variants(
    strategy_name: str,
    strategy_fn,
    budget_ratios: Sequence[float],
    seed: int,
):
    def _runner(bundle: DatasetBundle, budget_ratio: float) -> dict[str, Any]:
        result = dict(strategy_fn(bundle))
        raw_indices = np.asarray(result.get("sample_indices", []), dtype=int)
        raw_scores = result.get("sample_scores")

        budgeted = _apply_budget_policy(
            informative_indices=raw_indices,
            informative_scores=raw_scores,
            train_size=bundle.y_train.shape[0],
            budget_ratio=budget_ratio,
            seed=seed,
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

    wrapped = {}
    for ratio in budget_ratios:
        ratio_tag = f"{int(round(ratio * 100)):02d}"

        def _factory(bundle: DatasetBundle, current_ratio: float = ratio):
            return _runner(bundle, current_ratio)

        wrapped[f"{strategy_name}__budget_{ratio_tag}"] = _factory
    return wrapped


def make_benchmark_strategies(seed: int, budget_ratios: Sequence[float]) -> dict[str, Any]:
    strategy_pool = make_strategies(seed=seed)
    wrapped: dict[str, Any] = {}
    for strategy_name, strategy_fn in strategy_pool.items():
        wrapped.update(with_budget_variants(strategy_name, strategy_fn, budget_ratios, seed))

    def full_dataset_strategy(bundle: DatasetBundle) -> dict[str, Any]:
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

    wrapped["full_dataset"] = full_dataset_strategy
    return wrapped


def _with_enriched_dimensions(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    enriched["strategy_base"] = enriched["strategy"].map(_strategy_base_name)
    enriched["budget_ratio"] = pd.to_numeric(enriched.get("strategy_params.budget_ratio"), errors="coerce")
    enriched["budget_ratio"] = enriched["budget_ratio"].fillna(pd.to_numeric(enriched.get("extra.budget_ratio"), errors="coerce"))
    enriched["budget_percent"] = (enriched["budget_ratio"] * 100.0).round(2)
    enriched["model"] = enriched["strategy"].str.rsplit("__", n=1).str[-1]
    return enriched


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


def resolve_datasets(full_benchmark: bool, include_amlb: bool, amlb_categories: Sequence[str] | None) -> list[str]:
    dataset_names = ["mixed_hard"]
    if full_benchmark:
        dataset_names = ["high_cardinality_categorical", "large_numeric", "mixed_hard"]

    if include_amlb:
        dataset_names.extend(["amlb_adult", "amlb_covertype"])

    for category in amlb_categories or []:
        datasets_for_category = AMLB_CATEGORY_PROFILES.get(category)
        if not datasets_for_category:
            raise ValueError(f"Unknown AMLB category: {category}. Available: {sorted(AMLB_CATEGORY_PROFILES)}")
        dataset_names.extend(datasets_for_category)

    unique_names = []
    seen = set()
    for name in dataset_names:
        if name not in seen:
            seen.add(name)
            unique_names.append(name)
    return unique_names


def main(
    full_benchmark: bool = True,
    include_amlb: bool = False,
    amlb_categories: Sequence[str] | None = ("small_samples_many_classes",),
    budget_ratios: Sequence[float] = DEFAULT_BUDGET_RATIOS,
) -> None:
    base_dir = Path(__file__).resolve().parent
    run_id = f"run_260226_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger = BenchmarkLogger(run_id=run_id, artifacts_root=base_dir / "results")

    dataset_names = resolve_datasets(full_benchmark, include_amlb, amlb_categories)

    datasets: list[DatasetBundle] = []
    for dataset_name in dataset_names:
        try:
            datasets.append(load_dataset(dataset_name, seed=42))
        except Exception as err:
            print(f"[WARN] dataset {dataset_name} is skipped: {err}")

    if not datasets:
        raise RuntimeError("No datasets available for benchmark run.")

    strategy_pool = make_benchmark_strategies(seed=42, budget_ratios=budget_ratios)
    model_pool = make_model_pool(seed=42)

    run_records: list[dict[str, Any]] = []
    for model_name, model in model_pool.items():
        runner = SpecialStrategyBenchmarkRunner(logger=logger)
        model_tagged_strategies = {f"{name}__{model_name}": fn for name, fn in strategy_pool.items()}
        run_records.extend(runner.run(datasets=datasets, strategies=model_tagged_strategies, base_model=model))

    enriched_df = build_summary_tables(run_records, logger.paths.root)
    plot_metrics_vs_budget(enriched_df, logger.paths.plots)
    plot_informative_overlap(enriched_df, logger.paths.plots)
    plot_final_summary_table(enriched_df, logger.paths.plots)

    run_meta = {
        "run_id": logger.run_id,
        "output_dir": str(logger.paths.root),
        "datasets": [dataset.name for dataset in datasets],
        "strategies": sorted({_strategy_base_name(name) for name in strategy_pool}),
        "models": list(model_pool.keys()),
        "budget_ratios": [float(ratio) for ratio in budget_ratios],
        "amlb_categories": list(amlb_categories or []),
        "records": len(run_records),
    }
    (logger.paths.root / "run_meta.json").write_text(json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Benchmark completed. Artifacts: {logger.paths.root}")


if __name__ == "__main__":
    main()
