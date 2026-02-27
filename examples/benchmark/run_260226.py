from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping
import sys
import pandas as pd

from benchmark_sampling_strategies import make_strategies
from bechmark_models import make_model_pool

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from benchmark_datasets import DatasetBundle, load_dataset
from benchmark_logging import BenchmarkLogger
from benchmark_runner import SpecialStrategyBenchmarkRunner

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover - optional
    LGBMClassifier = None

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception:  # pragma: no cover - optional
    torch = None


def build_summary_tables(run_records: Iterable[Mapping[str, Any]], output_dir: Path) -> None:
    df = pd.json_normalize(list(run_records), sep=".")
    df.to_csv(output_dir / "benchmark_runs.csv", index=False)
    (output_dir / "benchmark_runs.json").write_text(df.to_json(orient="records", force_ascii=False, indent=2),
                                                    encoding="utf-8")

    summary = (
        df.groupby(["dataset", "strategy"], as_index=False)
        .agg(
            roc_auc=("model_metrics.roc_auc", "mean"),
            f1_macro=("model_metrics.f1_macro", "mean"),
            f1_weighted=("model_metrics.f1_weighted", "mean"),
            fit_sec=("timings_sec.fit", "mean"),
            sample_sec=("timings_sec.sample", "mean"),
            infer_sec=("timings_sec.inference", "mean"),
            sample_size=("sample_stats.sample_size", "mean"),
        )
        .sort_values(["dataset", "f1_macro"], ascending=[True, False])
    )
    summary.to_csv(output_dir / "summary_by_strategy.csv", index=False)


def main(FULL_BENCHMARK: bool = True, INCLUDE_AMLB: bool = False) -> None:
    base_dir = Path(__file__).resolve().parent
    run_id = f"run_260226_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger = BenchmarkLogger(run_id=run_id, artifacts_root=base_dir / "results")

    dataset_names = ["mixed_hard"]
    if FULL_BENCHMARK:
        dataset_names = ["high_cardinality_categorical", "large_numeric", "mixed_hard"]
    if INCLUDE_AMLB:
        dataset_names.append("amlb_adult")
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
        model_tagged_strategies = {f"{name}__{model_name}": fn for name, fn in strategy_pool.items()}
        run_records.extend(runner.run(datasets=datasets, strategies=model_tagged_strategies, base_model=model))

    build_summary_tables(run_records, logger.paths.root)

    run_meta = {
        "run_id": logger.run_id,
        "output_dir": str(logger.paths.root),
        "datasets": [dataset.name for dataset in datasets],
        "strategies": list(strategy_pool.keys()),
        "models": list(model_pool.keys()),
        "records": len(run_records),
    }
    (logger.paths.root / "run_meta.json").write_text(json.dumps(run_meta, ensure_ascii=False, indent=2),
                                                     encoding="utf-8")

    print(f"Benchmark completed. Artifacts: {logger.paths.root}")


if __name__ == "__main__":
    main()
