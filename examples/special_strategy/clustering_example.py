"""FeatureClusteringSampler в унифицированном формате."""

import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from core.api.api_main import SamplingStrategyFactory
from core.utils.synt_data import create_noisy_dataset

DATASET_SAMPLES = 10_000
STRATEGY_TYPE = "feature_clustering"
CLUSTERING_MODELS = ["kmeans"]


def run_feature_clustering() -> None:
    data = create_noisy_dataset(DATASET_SAMPLES)

    for model in CLUSTERING_MODELS:
        strategy_params = {"n_clusters": 3, "method": model}
        factory = SamplingStrategyFactory()
        strategy = factory.create_strategy(strategy_type=STRATEGY_TYPE, **strategy_params)
        strategy.fit(data[["feature_1", "feature_2"]], target=data["target"])

        partitions = strategy.get_partitions(data[["feature_1", "feature_2"]], target=data["target"])
        indices = {name: part["feature"].index.to_numpy() for name, part in partitions.items()}
        strategy.print_fold_summary(f"FeatureClustering ({model})", indices, data["target"])


if __name__ == "__main__":
    run_feature_clustering()
