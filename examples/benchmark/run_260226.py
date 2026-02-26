from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import sys

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.sampling_strategies.delaunay_sempler import DelaunaySampler
from core.sampling_strategies.hdbscan_sampler import HDBScanSampler
from core.sampling_strategies.random_sampler import RandomSplitSampler
from core.sampling_strategies.spectral.spectral_leverage import SpectralLeverageSampler
from core.sampling_strategies.spectral.tensor_energy import TensorEnergySampler
from core.sampling_strategies.voronoi_sampler import VoronoiSampler
from benchmark_datasets import DatasetBundle, load_dataset
from benchmark_logging import BenchmarkLogger
from benchmark_runner import SpecialStrategyBenchmarkRunner


def _to_dense(matrix: np.ndarray | sparse.spmatrix) -> np.ndarray:
    return matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)


def _sample_from_partitions(partitions: Mapping[str, np.ndarray], n_train: int, sample_ratio: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    picks: list[int] = []
    for _, indices in partitions.items():
        idx = np.asarray(indices, dtype=int)
        if idx.size == 0:
            continue
        take = min(idx.size, max(1, int(round(idx.size * sample_ratio))))
        picks.extend(rng.choice(idx, size=take, replace=False).tolist())

    if not picks:
        all_idx = np.arange(n_train)
        fallback_take = max(10, int(n_train * sample_ratio))
        picks = rng.choice(all_idx, size=min(fallback_take, n_train), replace=False).tolist()

    return np.unique(np.asarray(picks, dtype=int))


def make_strategies(seed: int = 42) -> Dict[str, Any]:
    def spectral_leverage_strategy(bundle: DatasetBundle) -> Dict[str, Any]:
        x = _to_dense(bundle.X_train_processed)
        sample_size = max(500, int(0.2 * x.shape[0]))
        sampler = SpectralLeverageSampler(sample_size=sample_size, approx_rank=32, random_state=seed, return_weights=True)
        sampler.fit(x)
        sampled = sampler.sample_indices(replace=False)
        sample_indices, weights = sampled if isinstance(sampled, tuple) else (sampled, None)
        return {
            "sample_indices": np.asarray(sample_indices, dtype=int),
            "sample_scores": np.asarray(weights, dtype=float) if weights is not None else None,
            "strategy_params": {"sample_size": sample_size, "approx_rank": 32},
        }

    def tensor_energy_strategy(bundle: DatasetBundle) -> Dict[str, Any]:
        x = _to_dense(bundle.X_train_processed)
        # Convert matrix to pseudo-tensor: [samples, chunks, chunk_width]
        chunk_width = 8
        n_samples, n_features = x.shape
        pad = (-n_features) % chunk_width
        if pad:
            x = np.pad(x, ((0, 0), (0, pad)), mode="constant")
            n_features = x.shape[1]
        x_tensor = x.reshape(n_samples, n_features // chunk_width, chunk_width)

        sampler = TensorEnergySampler(
            sample_size=max(500, int(0.2 * n_samples)),
            modes=[0, 1],
            approx_rank=[32, 8],
            random_state=seed,
            return_weights=True,
        )
        sampler.fit(x_tensor)
        sampled = sampler.sample_indices(replace=False)
        sampled_pairs, weights = sampled if isinstance(sampled, tuple) else (sampled, None)
        # Tensor sampler returns tuples -> map to row indices (mode 0)
        sample_indices = np.unique(np.asarray([pair[0] for pair in sampled_pairs], dtype=int))
        return {
            "sample_indices": sample_indices,
            "sample_scores": np.asarray(weights, dtype=float) if weights is not None else None,
            "strategy_params": {"modes": [0, 1], "chunk_width": chunk_width},
        }

    def voronoi_strategy(bundle: DatasetBundle) -> Dict[str, Any]:
        x = _to_dense(bundle.X_train_processed)
        y = bundle.y_train.to_numpy()
        sampler = VoronoiSampler(n_partitions=12, random_state=seed, emptiness_threshold=0.01)
        sampler.fit(x, y)
        sampled_indices = _sample_from_partitions(sampler.partitions, len(y), sample_ratio=0.2, seed=seed)
        labels = sampler.predict_partitions(x)
        return {
            "sample_indices": sampled_indices,
            "cluster_labels": np.asarray(labels, dtype=int),
            "strategy_params": {"n_partitions": 12},
        }

    def hdbscan_strategy(bundle: DatasetBundle) -> Dict[str, Any]:
        x = _to_dense(bundle.X_train_processed)
        y = bundle.y_train.to_numpy()
        sampler = HDBScanSampler(min_cluster_size=50, one_cluster=True, all_points=True, random_state=seed)
        sampler.fit(x, y)
        sampled_indices = _sample_from_partitions(sampler.partitions, len(y), sample_ratio=0.2, seed=seed)
        labels = sampler.predict_partitions(x)
        return {
            "sample_indices": sampled_indices,
            "cluster_labels": np.asarray(labels, dtype=int),
            "strategy_params": {"min_cluster_size": 50, "fallback_dbscan": bool(getattr(sampler, "_dbscan_fallback", False))},
        }

    def delaunay_strategy(bundle: DatasetBundle) -> Dict[str, Any]:
        x = _to_dense(bundle.X_train_processed)
        y = bundle.y_train.to_numpy()
        sampler = DelaunaySampler(
            n_partitions=12,
            n_clusters=30,
            random_state=seed,
            emptiness_threshold=0.01,
            dim_reduction_method="pca",
            dim_reduction_target=2,
        )
        sampler.fit(x, y)
        sampled_indices = _sample_from_partitions(sampler.partitions, len(y), sample_ratio=0.2, seed=seed)
        simplex_ids = sampler.predict_partitions(x)
        return {
            "sample_indices": sampled_indices,
            "simplex_ids": np.asarray(simplex_ids, dtype=int),
            "strategy_params": {"n_partitions": 12, "n_clusters": 30, "dim_reduction_method": "pca"},
        }

    def random_strategy(bundle: DatasetBundle) -> Dict[str, Any]:
        x = bundle.X_train
        y = bundle.y_train.to_numpy()
        sampler = RandomSplitSampler(n_partitions=12, random_state=seed)
        sampler.fit(x, y)
        sampled_indices = _sample_from_partitions(sampler.partitions, len(y), sample_ratio=0.2, seed=seed)
        return {
            "sample_indices": sampled_indices,
            "strategy_params": {"n_partitions": 12},
        }

    return {
        "spectral_leverage": spectral_leverage_strategy,
        "tensor_energy": tensor_energy_strategy,
        "voronoi": voronoi_strategy,
        #"hdbscan": hdbscan_strategy,
        "delaunay": delaunay_strategy,
        "random": random_strategy,
    }


def build_summary_tables(run_records: Iterable[Mapping[str, Any]], output_dir: Path) -> None:
    df = pd.json_normalize(list(run_records), sep=".")
    df.to_csv(output_dir / "benchmark_runs.csv", index=False)
    (output_dir / "benchmark_runs.json").write_text(df.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")

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


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    run_id = f"run_260226_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger = BenchmarkLogger(run_id=run_id, artifacts_root=base_dir / "results")

    datasets = [
        load_dataset("high_cardinality_categorical", seed=42),
        load_dataset("large_numeric", seed=42),
        load_dataset("mixed_hard", seed=42),
    ]
    strategies = make_strategies(seed=42)

    model = RandomForestClassifier(
        n_estimators=80,
        max_depth=10,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42,
    )

    runner = SpecialStrategyBenchmarkRunner(logger=logger)
    run_records = runner.run(datasets=datasets, strategies=strategies, base_model=model)

    build_summary_tables(run_records, logger.paths.root)

    run_meta = {
        "run_id": logger.run_id,
        "output_dir": str(logger.paths.root),
        "datasets": [dataset.name for dataset in datasets],
        "strategies": list(strategies.keys()),
        "records": len(run_records),
    }
    (logger.paths.root / "run_meta.json").write_text(json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Benchmark completed. Artifacts: {logger.paths.root}")


if __name__ == "__main__":
    main()
