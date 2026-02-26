from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence

import os

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neural_network import MLPClassifier

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.sampling_strategies.delaunay_sempler import DelaunaySampler
from core.sampling_strategies.hdbscan_sampler import HDBScanSampler
from core.sampling_strategies.random_sampler import RandomSplitSampler
from core.sampling_strategies.spectral.spectral_leverage import SpectralLeverageSampler
from core.sampling_strategies.spectral.tensor_energy import TensorEnergySampler
from core.sampling_strategies.voronoi_sampler import VoronoiSampler
from examples.benchmark.benchmark_datasets import DatasetBundle, load_dataset
from examples.benchmark.benchmark_logging import BenchmarkLogger
from examples.benchmark.benchmark_runner import SpecialStrategyBenchmarkRunner

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


@dataclass
class SearchResult:
    best_params: Dict[str, Any]
    best_score: float


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
        fallback_take = max(10, int(n_train * sample_ratio))
        picks = rng.choice(np.arange(n_train), size=min(fallback_take, n_train), replace=False).tolist()

    return np.unique(np.asarray(picks, dtype=int))


def _build_inner_split(y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    train_idx, valid_idx = next(splitter.split(np.zeros_like(y), y))
    return train_idx, valid_idx


def _score_sample_indices(sample_indices: Sequence[int], y_train: np.ndarray, val_idx: np.ndarray) -> float:
    sample_idx = np.asarray(sample_indices, dtype=int)
    if sample_idx.size == 0:
        return -1.0
    sample_set = set(sample_idx.tolist())
    overlap = np.array([idx for idx in val_idx if idx in sample_set], dtype=int)
    coverage = len(sample_set) / max(len(y_train), 1)
    class_balance = len(np.unique(y_train[sample_idx])) / max(len(np.unique(y_train)), 1)
    overlap_penalty = 0.25 if overlap.size > 0 else 0.0
    return 0.6 * class_balance + 0.4 * coverage - overlap_penalty


def _search_params(
    candidates: Iterable[Dict[str, Any]],
    sampler_factory: Callable[[Dict[str, Any]], tuple[Sequence[int], Dict[str, Any]]],
    y_train: np.ndarray,
    seed: int,
) -> SearchResult:
    _, valid_idx = _build_inner_split(y_train, seed)
    best_score = -np.inf
    best_params: Dict[str, Any] = {}
    for params in candidates:
        indices, extra = sampler_factory(params)
        score = _score_sample_indices(indices, y_train, valid_idx)
        if score > best_score:
            best_score = score
            best_params = {**params, **extra}
    return SearchResult(best_params=best_params, best_score=float(best_score))


def _make_pytorch_classifier(seed: int) -> ClassifierMixin:
    if torch is None:
        return MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=100, random_state=seed)

    class TorchMLPClassifier(ClassifierMixin):
        def __init__(self, input_dim: Optional[int] = None, hidden_dim: int = 128, epochs: int = 12, lr: float = 1e-3):
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.epochs = epochs
            self.lr = lr

        def _build_model(self, in_dim: int):
            return nn.Sequential(
                nn.Linear(in_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(self.hidden_dim // 2, 2),
            )

        def fit(self, X, y):
            X = np.asarray(X, dtype=np.float32)
            y = np.asarray(y, dtype=np.int64)
            torch.manual_seed(seed)
            self.model_ = self._build_model(X.shape[1])
            optimizer = optim.Adam(self.model_.parameters(), lr=self.lr)
            criterion = nn.CrossEntropyLoss()
            self.model_.train()
            x_t = torch.from_numpy(X)
            y_t = torch.from_numpy(y)
            for _ in range(self.epochs):
                optimizer.zero_grad()
                logits = self.model_(x_t)
                loss = criterion(logits, y_t)
                loss.backward()
                optimizer.step()
            return self

        def predict_proba(self, X):
            X = np.asarray(X, dtype=np.float32)
            self.model_.eval()
            with torch.no_grad():
                logits = self.model_(torch.from_numpy(X))
                probs = torch.softmax(logits, dim=1).cpu().numpy()
            return probs

        def predict(self, X):
            return np.argmax(self.predict_proba(X), axis=1)

    return TorchMLPClassifier()


def make_model_pool(seed: int = 42) -> Dict[str, ClassifierMixin]:
    models: Dict[str, ClassifierMixin] = {
        "random_forest": RandomForestClassifier(
            n_estimators=80,
            max_depth=10,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=seed,
        ),
        "pytorch_mlp": _make_pytorch_classifier(seed),
    }

    if LGBMClassifier is not None:
        models["lightgbm"] = LGBMClassifier(
            n_estimators=220,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            verbosity=-1,
        )
    else:
        models["hist_gradient_boosting"] = HistGradientBoostingClassifier(
            max_depth=8,
            learning_rate=0.06,
            max_iter=250,
            random_state=seed,
        )

    return models


def make_strategies(seed: int = 42) -> Dict[str, Any]:
    def spectral_leverage_strategy(bundle: DatasetBundle) -> Dict[str, Any]:
        x = _to_dense(bundle.X_train_processed)
        y = bundle.y_train.to_numpy()

        grid = [{"approx_rank": r} for r in (16, 32)]

        def _factory(params: Dict[str, Any]):
            sample_size = max(500, int(0.2 * x.shape[0]))
            sampler = SpectralLeverageSampler(sample_size=sample_size, approx_rank=params["approx_rank"], random_state=seed, return_weights=True)
            sampler.fit(x)
            sampled = sampler.sample_indices(replace=False)
            sample_indices, weights = sampled if isinstance(sampled, tuple) else (sampled, None)
            return sample_indices, {"sample_size": sample_size, "weights_available": weights is not None}

        search = _search_params(grid, _factory, y, seed)
        sample_indices, extra = _factory(search.best_params)
        sampler = SpectralLeverageSampler(sample_size=search.best_params["sample_size"], approx_rank=search.best_params["approx_rank"], random_state=seed, return_weights=True)
        sampler.fit(x)
        sampled = sampler.sample_indices(replace=False)
        _, weights = sampled if isinstance(sampled, tuple) else (sampled, None)

        return {
            "sample_indices": np.asarray(sample_indices, dtype=int),
            "sample_scores": np.asarray(weights, dtype=float) if weights is not None else None,
            "strategy_params": {**search.best_params, **extra},
            "extra": {"param_search_score": search.best_score, "param_grid_size": len(grid)},
        }

    def tensor_energy_strategy(bundle: DatasetBundle) -> Dict[str, Any]:
        x = _to_dense(bundle.X_train_processed)
        y = bundle.y_train.to_numpy()
        chunk_width = 8
        n_samples, n_features = x.shape
        pad = (-n_features) % chunk_width
        if pad:
            x = np.pad(x, ((0, 0), (0, pad)), mode="constant")
            n_features = x.shape[1]
        x_tensor = x.reshape(n_samples, n_features // chunk_width, chunk_width)

        grid = [{"rank_a": 16, "rank_b": 6}, {"rank_a": 32, "rank_b": 10}]

        def _factory(params: Dict[str, Any]):
            sampler = TensorEnergySampler(
                sample_size=max(500, int(0.2 * n_samples)),
                modes=[0, 1],
                approx_rank=[params["rank_a"], params["rank_b"]],
                random_state=seed,
                return_weights=True,
            )
            sampler.fit(x_tensor)
            sampled = sampler.sample_indices(replace=False)
            sampled_pairs, _ = sampled if isinstance(sampled, tuple) else (sampled, None)
            sample_indices = np.unique(np.asarray([pair[0] for pair in sampled_pairs], dtype=int))
            return sample_indices, {"chunk_width": chunk_width}

        search = _search_params(grid, _factory, y, seed)
        sample_indices, extra = _factory(search.best_params)

        sampler = TensorEnergySampler(
            sample_size=max(500, int(0.2 * n_samples)),
            modes=[0, 1],
            approx_rank=[search.best_params["rank_a"], search.best_params["rank_b"]],
            random_state=seed,
            return_weights=True,
        )
        sampler.fit(x_tensor)
        sampled = sampler.sample_indices(replace=False)
        _, weights = sampled if isinstance(sampled, tuple) else (sampled, None)

        return {
            "sample_indices": sample_indices,
            "sample_scores": np.asarray(weights, dtype=float) if weights is not None else None,
            "strategy_params": {**search.best_params, **extra, "modes": [0, 1]},
            "extra": {"param_search_score": search.best_score, "param_grid_size": len(grid)},
        }

    def voronoi_strategy(bundle: DatasetBundle) -> Dict[str, Any]:
        x = _to_dense(bundle.X_train_processed)
        y = bundle.y_train.to_numpy()
        grid = [{"n_partitions": n} for n in (8, 12)]

        def _factory(params: Dict[str, Any]):
            sampler = VoronoiSampler(n_partitions=params["n_partitions"], random_state=seed, emptiness_threshold=0.01)
            sampler.fit(x, y)
            sampled_indices = _sample_from_partitions(sampler.partitions, len(y), sample_ratio=0.2, seed=seed)
            return sampled_indices, {"cluster_labels": sampler.predict_partitions(x)}

        search = _search_params(grid, _factory, y, seed)
        sample_indices, extra = _factory(search.best_params)
        return {
            "sample_indices": sample_indices,
            "cluster_labels": np.asarray(extra["cluster_labels"], dtype=int),
            "strategy_params": search.best_params,
            "extra": {"param_search_score": search.best_score, "param_grid_size": len(grid)},
        }

    def hdbscan_strategy(bundle: DatasetBundle) -> Dict[str, Any]:
        x = _to_dense(bundle.X_train_processed)
        y = bundle.y_train.to_numpy()
        grid = [{"min_cluster_size": m} for m in (25, 60)]

        def _factory(params: Dict[str, Any]):
            sampler = HDBScanSampler(min_cluster_size=params["min_cluster_size"], one_cluster=True, all_points=True, random_state=seed)
            sampler.fit(x, y)
            sampled_indices = _sample_from_partitions(sampler.partitions, len(y), sample_ratio=0.2, seed=seed)
            return sampled_indices, {"cluster_labels": sampler.predict_partitions(x), "fallback_dbscan": bool(getattr(sampler, "_dbscan_fallback", False))}

        search = _search_params(grid, _factory, y, seed)
        sample_indices, extra = _factory(search.best_params)
        return {
            "sample_indices": sample_indices,
            "cluster_labels": np.asarray(extra["cluster_labels"], dtype=int),
            "strategy_params": {**search.best_params, "fallback_dbscan": extra["fallback_dbscan"]},
            "extra": {"param_search_score": search.best_score, "param_grid_size": len(grid)},
        }

    def delaunay_strategy(bundle: DatasetBundle) -> Dict[str, Any]:
        x = _to_dense(bundle.X_train_processed)
        y = bundle.y_train.to_numpy()
        grid = [{"n_clusters": n} for n in (18, 30)]

        def _factory(params: Dict[str, Any]):
            sampler = DelaunaySampler(
                n_partitions=12,
                n_clusters=params["n_clusters"],
                random_state=seed,
                emptiness_threshold=0.01,
                dim_reduction_method="pca",
                dim_reduction_target=2,
            )
            sampler.fit(x, y)
            sampled_indices = _sample_from_partitions(sampler.partitions, len(y), sample_ratio=0.2, seed=seed)
            return sampled_indices, {"simplex_ids": sampler.predict_partitions(x)}

        search = _search_params(grid, _factory, y, seed)
        sample_indices, extra = _factory(search.best_params)
        return {
            "sample_indices": sample_indices,
            "simplex_ids": np.asarray(extra["simplex_ids"], dtype=int),
            "strategy_params": {**search.best_params, "n_partitions": 12, "dim_reduction_method": "pca"},
            "extra": {"param_search_score": search.best_score, "param_grid_size": len(grid)},
        }

    def random_strategy(bundle: DatasetBundle) -> Dict[str, Any]:
        y = bundle.y_train.to_numpy()
        grid = [{"n_partitions": n} for n in (8, 12)]

        def _factory(params: Dict[str, Any]):
            sampler = RandomSplitSampler(n_partitions=params["n_partitions"], random_state=seed)
            sampler.fit(bundle.X_train, y)
            sampled_indices = _sample_from_partitions(sampler.partitions, len(y), sample_ratio=0.2, seed=seed)
            return sampled_indices, {}

        search = _search_params(grid, _factory, y, seed)
        sample_indices, _ = _factory(search.best_params)
        return {
            "sample_indices": sample_indices,
            "strategy_params": search.best_params,
            "extra": {"param_search_score": search.best_score, "param_grid_size": len(grid)},
        }

    return {
        "spectral_leverage": spectral_leverage_strategy,
        "tensor_energy": tensor_energy_strategy,
        "voronoi": voronoi_strategy,
        "hdbscan": hdbscan_strategy,
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

    dataset_names = ["mixed_hard"]
    if os.getenv("FULL_BENCHMARK", "0") == "1":
        dataset_names = ["high_cardinality_categorical", "large_numeric", "mixed_hard"]
    if os.getenv("INCLUDE_AMLB", "0") == "1":
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
    (logger.paths.root / "run_meta.json").write_text(json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Benchmark completed. Artifacts: {logger.paths.root}")


if __name__ == "__main__":
    main()
