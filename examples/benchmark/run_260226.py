from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from benchmark_datasets import DatasetBundle, load_dataset, resolve_dataset_names
from benchmark_logging import BenchmarkLogger
from benchmark_runner import SpecialStrategyBenchmarkRunner
from benchmark_sampling_strategies import make_strategies
from bechmark_models import make_model_pool

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def _extract_base_strategy(strategy_name: str) -> str:
    name = strategy_name
    if "__model_" in name:
        name = name.split("__model_", 1)[0]
    return name.split("__pct_", 1)[0]


def _extract_data_percent(strategy_name: str, default_percent: float) -> float:
    match = re.search(r"__pct_(\d+)", strategy_name)
    if match:
        return float(match.group(1))
    if "full_dataset" in strategy_name:
        return 100.0
    return float(default_percent)


def _apply_data_budget(
    informative_indices: Sequence[int],
    train_size: int,
    target_percent: int,
    seed: int,
    sample_scores: Sequence[float] | None = None,
) -> tuple[np.ndarray, Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    target_size = max(1, int(round(train_size * (target_percent / 100.0))))

    informative = np.asarray(informative_indices, dtype=int)
    informative = informative[(informative >= 0) & (informative < train_size)]
    informative = np.unique(informative)

    details: Dict[str, Any] = {
        "target_percent": int(target_percent),
        "target_size": int(target_size),
        "informative_size": int(informative.size),
        "budget_action": "exact",
    }

    if informative.size == target_size:
        return informative, details

    if informative.size > target_size:
        details["budget_action"] = "trim_top_k"
        if sample_scores is not None:
            raw_scores = np.asarray(sample_scores, dtype=float)
            if raw_scores.shape[0] == informative.size:
                order = np.argsort(-raw_scores)
                informative = informative[order]
        return np.sort(informative[:target_size]), details

    details["budget_action"] = "pad_with_random"
    need = target_size - informative.size
    remainder = np.setdiff1d(np.arange(train_size, dtype=int), informative, assume_unique=False)
    if remainder.size > 0:
        add = rng.choice(remainder, size=min(need, remainder.size), replace=False)
        informative = np.concatenate([informative, np.asarray(add, dtype=int)])
    return np.sort(np.unique(informative)), details


def _build_budgeted_strategy(
    strategy_name: str,
    strategy_fn,
    data_percent: int,
    seed: int,
):
    def _wrapped(bundle: DatasetBundle) -> Dict[str, Any]:
        raw_output = dict(strategy_fn(bundle))
        informative_indices = np.asarray(raw_output.get("sample_indices", []), dtype=int)
        if informative_indices.size == 0:
            raise ValueError(f"{strategy_name}: strategy returned empty informative indices")

        adjusted_indices, details = _apply_data_budget(
            informative_indices=informative_indices,
            train_size=bundle.metadata.n_train,
            target_percent=data_percent,
            seed=seed,
            sample_scores=raw_output.get("sample_scores"),
        )

        strategy_params = dict(raw_output.get("strategy_params", {}))
        strategy_params.update({
            "data_percent": data_percent,
            "target_sample_size": details["target_size"],
            "informative_size": details["informative_size"],
        })

        extra = dict(raw_output.get("extra", {}))
        extra.update({
            "budget_action": details["budget_action"],
            "data_percent": data_percent,
            "informative_indices": informative_indices.astype(int),
            "informative_size": details["informative_size"],
        })

        raw_output["sample_indices"] = adjusted_indices
        raw_output["strategy_params"] = strategy_params
        raw_output["extra"] = extra
        return raw_output

    return _wrapped


def _make_full_dataset_strategy():
    def _full(bundle: DatasetBundle) -> Dict[str, Any]:
        full_idx = np.arange(bundle.metadata.n_train, dtype=int)
        return {
            "sample_indices": full_idx,
            "strategy_params": {"data_percent": 100, "target_sample_size": int(bundle.metadata.n_train)},
            "extra": {
                "data_percent": 100,
                "budget_action": "full_dataset",
                "informative_indices": full_idx,
                "informative_size": int(bundle.metadata.n_train),
            },
        }

    return _full


def build_summary_tables(run_records: Iterable[Mapping[str, Any]], output_dir: Path) -> pd.DataFrame:
    df = pd.json_normalize(list(run_records), sep=".")
    df["base_strategy"] = df["strategy"].apply(_extract_base_strategy)
    df["data_percent"] = df["strategy"].apply(lambda name: _extract_data_percent(str(name), default_percent=20))

    df.to_csv(output_dir / "benchmark_runs.csv", index=False)
    (output_dir / "benchmark_runs.json").write_text(
        df.to_json(orient="records", force_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = (
        df.groupby(["dataset", "base_strategy", "data_percent"], as_index=False)
        .agg(
            roc_auc=("model_metrics.roc_auc", "mean"),
            f1_macro=("model_metrics.f1_macro", "mean"),
            f1_weighted=("model_metrics.f1_weighted", "mean"),
            fit_sec=("timings_sec.fit", "mean"),
            sample_sec=("timings_sec.sample", "mean"),
            infer_sec=("timings_sec.inference", "mean"),
            sample_size=("sample_stats.sample_size", "mean"),
        )
        .sort_values(["dataset", "base_strategy", "data_percent"])
    )
    summary.to_csv(output_dir / "summary_by_strategy.csv", index=False)
    return df


def _load_indices(path_str: str | None) -> set[int]:
    if not path_str:
        return set()
    path = Path(path_str)
    if not path.exists():
        return set()
    return set(np.load(path).astype(int).tolist())


def create_visualizations(df: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    # 2.1 Dynamics by data percentage
    metrics = [
        ("model_metrics.f1_macro", "F1 macro"),
        ("model_metrics.roc_auc", "ROC-AUC"),
        ("timings_sec.fit", "Fit time (sec)"),
        ("timings_sec.sample", "Sampling time (sec)"),
    ]

    for dataset_name, ds_df in df.groupby("dataset"):
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        for ax, (metric_col, title) in zip(axes.flatten(), metrics):
            grouped = (
                ds_df.groupby(["base_strategy", "data_percent"], as_index=False)[metric_col]
                .mean()
                .sort_values("data_percent")
            )
            for strategy_name, strategy_df in grouped.groupby("base_strategy"):
                ax.plot(strategy_df["data_percent"], strategy_df[metric_col], marker="o", label=strategy_name)
            ax.set_title(title)
            ax.set_xlabel("Training data percent")
            ax.grid(alpha=0.25)
        axes[0, 0].legend(loc="best", fontsize=8)
        fig.suptitle(f"Metric dynamics by data budget: {dataset_name}")
        fig.tight_layout()
        fig.savefig(output_dir / f"metrics_dynamics__{dataset_name}.png", dpi=150)
        plt.close(fig)

    # 2.2 Informative sample overlap (Jaccard)
    for dataset_name, ds_df in df.groupby("dataset"):
        for data_percent, pct_df in ds_df.groupby("data_percent"):
            per_strategy: dict[str, set[int]] = {}
            for _, row in pct_df.iterrows():
                strategy = row["base_strategy"]
                if strategy in per_strategy:
                    continue
                indices = _load_indices(row.get("extra.informative_indices_path"))
                if indices:
                    per_strategy[strategy] = indices

            if len(per_strategy) < 2:
                continue

            names = sorted(per_strategy)
            matrix = np.zeros((len(names), len(names)), dtype=float)
            for i, left in enumerate(names):
                for j, right in enumerate(names):
                    inter = len(per_strategy[left] & per_strategy[right])
                    union = len(per_strategy[left] | per_strategy[right])
                    matrix[i, j] = inter / union if union else 1.0

            fig, ax = plt.subplots(figsize=(7, 6))
            im = ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis")
            ax.set_xticks(range(len(names)))
            ax.set_yticks(range(len(names)))
            ax.set_xticklabels(names, rotation=45, ha="right")
            ax.set_yticklabels(names)
            ax.set_title(f"Informative sample overlap (Jaccard)\n{dataset_name}, {int(data_percent)}%")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            fig.savefig(output_dir / f"informative_overlap__{dataset_name}__pct_{int(data_percent)}.png", dpi=150)
            plt.close(fig)

    # 2.3 Final summary table as image
    summary = (
        df.groupby(["dataset", "base_strategy", "data_percent"], as_index=False)
        .agg(
            f1_macro=("model_metrics.f1_macro", "mean"),
            roc_auc=("model_metrics.roc_auc", "mean"),
            fit_sec=("timings_sec.fit", "mean"),
            sample_sec=("timings_sec.sample", "mean"),
            sample_size=("sample_stats.sample_size", "mean"),
        )
        .sort_values(["dataset", "f1_macro"], ascending=[True, False])
        .head(40)
    )

    fig, ax = plt.subplots(figsize=(16, max(6, 0.35 * len(summary))))
    ax.axis("off")
    table = ax.table(
        cellText=summary.round(4).astype(str).values,
        colLabels=list(summary.columns),
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.2)
    fig.tight_layout()
    fig.savefig(output_dir / "final_summary_table.png", dpi=180)
    plt.close(fig)


def main(
    FULL_BENCHMARK: bool = True,
    INCLUDE_AMLB: bool = False,
    data_percents: Sequence[int] = (1, 5, 10, 20),
    amlb_categories: Sequence[str] = ("small_samples_many_classes",),
) -> None:
    base_dir = Path(__file__).resolve().parent
    run_id = f"run_260226_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger = BenchmarkLogger(run_id=run_id, artifacts_root=base_dir / "results")

    dataset_names = ["mixed_hard"]
    if FULL_BENCHMARK:
        dataset_names = ["high_cardinality_categorical", "large_numeric", "mixed_hard"]
    dataset_names = resolve_dataset_names(
        dataset_names=dataset_names,
        include_amlb=INCLUDE_AMLB,
        amlb_categories=amlb_categories,
    )

    datasets: list[DatasetBundle] = []
    for dataset_name in dataset_names:
        try:
            datasets.append(load_dataset(dataset_name, seed=42))
        except Exception as err:
            print(f"[WARN] dataset {dataset_name} is skipped: {err}")

    if not datasets:
        raise RuntimeError("No datasets available for benchmark run.")

    strategy_pool = make_strategies(seed=42)
    model_pool = make_model_pool(seed=42)

    run_records: list[dict[str, Any]] = []
    for model_name, model in model_pool.items():
        runner = SpecialStrategyBenchmarkRunner(logger=logger)
        model_tagged_strategies: dict[str, Any] = {
            f"full_dataset__model_{model_name}": _make_full_dataset_strategy(),
        }
        for strategy_name, strategy_fn in strategy_pool.items():
            for pct in data_percents:
                tagged_name = f"{strategy_name}__pct_{int(pct)}__model_{model_name}"
                model_tagged_strategies[tagged_name] = _build_budgeted_strategy(
                    strategy_name=strategy_name,
                    strategy_fn=strategy_fn,
                    data_percent=int(pct),
                    seed=42 + int(pct),
                )

        run_records.extend(
            runner.run(datasets=datasets, strategies=model_tagged_strategies, base_model=model)
        )

    run_df = build_summary_tables(run_records, logger.paths.root)
    create_visualizations(run_df, logger.paths.root)

    run_meta = {
        "run_id": logger.run_id,
        "output_dir": str(logger.paths.root),
        "datasets": [dataset.name for dataset in datasets],
        "strategies": list(strategy_pool.keys()),
        "data_percents": list(map(int, data_percents)),
        "amlb_categories": list(amlb_categories),
        "models": list(model_pool.keys()),
        "records": len(run_records),
    }
    (logger.paths.root / "run_meta.json").write_text(
        json.dumps(run_meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Benchmark completed. Artifacts: {logger.paths.root}")


if __name__ == "__main__":
    main()
