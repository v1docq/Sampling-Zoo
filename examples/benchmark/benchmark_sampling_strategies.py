from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence
import sys
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
BENCHMARK_DIR = Path(__file__).resolve().parent
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

from benchmark_models import _search_params, _sample_from_partitions, _to_dense

from sampling_zoo.core.sampling_strategies.delaunay_sempler import DelaunaySampler
from sampling_zoo.core.sampling_strategies.hdbscan_sampler import HDBScanSampler
from sampling_zoo.core.sampling_strategies.random_sampler import RandomSplitSampler
from sampling_zoo.core.sampling_strategies.spectral.spectral_leverage import SpectralLeverageSampler
from sampling_zoo.core.sampling_strategies.spectral.tensor_energy import TensorEnergySampler
from sampling_zoo.core.sampling_strategies.voronoi_sampler import VoronoiSampler
from benchmark_datasets import DatasetBundle


try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover - optional
    LGBMClassifier = None

def make_strategies(seed: int = 42) -> Dict[str, Any]:
    def _bounded_sample_size(n_rows: int, target_ratio: float = 0.2, min_target: int = 500) -> int:
        target = max(min_target, int(target_ratio * n_rows))
        return max(1, min(n_rows, target))
    def spectral_leverage_strategy(bundle: DatasetBundle) -> Dict[str, Any]:
        x = _to_dense(bundle.X_train_processed)
        y = bundle.y_train.to_numpy()

        grid = [{"approx_rank": r} for r in (16, 32)]

        def _factory(params: Dict[str, Any]):
            sample_size = _bounded_sample_size(x.shape[0])
            sampler = SpectralLeverageSampler(sample_size=sample_size, approx_rank=params["approx_rank"], random_state=seed, return_weights=True)
            sampler.fit(x)
            sampled = sampler.sample_indices(replace=False)
            sample_indices, weights = sampled if isinstance(sampled, tuple) else (sampled, None)
            return sample_indices, {"sample_size": sample_size, "weights_available": weights is not None}

        search = _search_params(grid, _factory, y, seed)
        sample_indices, extra = _factory(search.best_params)
        sampler = SpectralLeverageSampler(sample_size=search.best_params["sample_size"],
                                          approx_rank=search.best_params["approx_rank"], random_state=seed,
                                          return_weights=True)
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
                sample_size=_bounded_sample_size(x.shape[0]),
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
            sample_size=_bounded_sample_size(x.shape[0]),
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
        #"hdbscan": hdbscan_strategy, #too long
        "delaunay": delaunay_strategy,
        "random": random_strategy,
    }


def make_chunking_strategy_configs(
    problem_type: str,
    strategy_names: Sequence[str],
    n_partitions: int = 3,
    seed: int = 42,
    ensemble_method: str = "voting",
    chunk_fraction: float | None = None,
    budget_ratio: float | None = None,
    force_chunking: bool = False,
    extra_strategy_params: Dict[str, Any] | None = None,
) -> Dict[str, Dict[str, Any]]:
    configs: Dict[str, Dict[str, Any]] = {}
    extra_strategy_params = extra_strategy_params or {}
    for strategy_name in strategy_names:
        normalized_name = strategy_name.strip().lower()
        if normalized_name not in {"difficulty", "random", "feature_clustering", "rmt_contraction"}:
            raise ValueError(f"Unsupported chunking strategy: {strategy_name}")

        strategy_config: Dict[str, Any] = {
            "strategy": normalized_name,
            "n_partitions": n_partitions,
            "random_state": seed,
            "ensemble_method": ensemble_method,
        }
        if budget_ratio is not None:
            strategy_config["budget_ratio"] = float(budget_ratio)
        if chunk_fraction is not None:
            strategy_config["experiment_chunk_fraction"] = float(chunk_fraction)
        if force_chunking:
            strategy_config["force_chunking"] = True

        if normalized_name == "difficulty":
            strategy_config["problem"] = problem_type
            strategy_config["chunks_percent"] = 100
        elif normalized_name == "feature_clustering":
            strategy_config["method"] = "kmeans"
        elif normalized_name == "rmt_contraction":
            strategy_config.update({
                "n_views": "auto",
                "n_views_policy": "auto",
                "min_views": 4,
                "max_views": 32,
                "target_feature_coverage": 0.95,
                "spectrum_stability_tolerance": 0.05,
                "view_strategy": "gaussian",
                "embedding_mode": "sv_scaled",
                "partition_selection_method": "auto",
                "cluster_algorithms": ["kmeans", "bisecting_kmeans", "gmm", "hdbscan"],
                "cluster_selection_metric": "balanced_silhouette",
                "cluster_ensemble_method": "coassociation",
                "min_partitions": 2,
                "max_partitions": max(8, n_partitions),
                "min_auto_partition_size": 256,
                "partition_selection_sample_size": 5000,
                "max_cluster_imbalance_ratio": 5.0,
                "min_cluster_fraction": 0.05,
                "imbalance_penalty_weight": 0.15,
                "tiny_cluster_penalty_weight": 0.30,
                "target_contrast_weight": 0.0,
                "cluster_target_type": problem_type,
                "missing_class_penalty_weight": 0.25,
                "single_class_penalty_weight": 0.50,
                "class_distribution_drift_weight": 0.25,
                "cluster_vote_temperature": 0.05,
                "projection_dim": 8,
                "initial_rank_fraction": 0.25,
                "rank_selection_method": "explained_variance",
                "explained_variance_threshold": 0.95,
                "min_rank": 1,
                "null_diagnostic_enabled": False,
                "null_model_policies": [
                    "feature_permutation",
                    "moment_matched_gaussian",
                    "view_resampling",
                ],
                "null_resamples": 16,
                "null_quantile": 0.95,
                "null_min_selection_frequency": 0.80,
                "null_primary_policy": "feature_permutation",
                "subspace_diagnostic_enabled": False,
                "subspace_resamples": 16,
                "subspace_quantile": 0.90,
                "subspace_max_principal_angle_degrees": 15.0,
                "subspace_max_normalized_projection_distance": 0.25,
                "subspace_max_rank": 64,
                "selection_method": "hybrid",
                "routing_temperature": 1.0,
                "routing_shrinkage": 0.05,
                "backend": "auto",
                "max_encoded_features": 4096,
            })
            if chunk_fraction is not None:
                strategy_config["chunk_fraction"] = float(chunk_fraction)

        strategy_config.update(extra_strategy_params.get(normalized_name, {}))

        configs[normalized_name] = strategy_config

    return configs
