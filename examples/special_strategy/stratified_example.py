"""Базовый пример StratifiedSplitSampler в унифицированном виде."""

import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from core.api.api_main import SamplingStrategyFactory
from core.sampling_strategies.base_sampler import HierarchicalStratifiedMixin
from core.utils.synt_data import create_distrib_dataset

DATASET_SAMPLES = 10_000
STRATEGY_TYPE = "stratified"
STRATEGY_PARAMS = {"n_partitions": 4}


def run_stratified_sampler() -> None:
    data = create_distrib_dataset(DATASET_SAMPLES)
    factory = SamplingStrategyFactory()
    strategy = factory.create_strategy(strategy_type=STRATEGY_TYPE, **STRATEGY_PARAMS)
    strategy.fit(data, target=["feature_1", "feature_2", "target"])

    partitions = strategy.get_partitions(data[["feature_1", "feature_2"]], target=data["target"])
    indices = {name: part["feature"].index.to_numpy() for name, part in partitions.items()}
    HierarchicalStratifiedMixin.print_fold_summary("StratifiedSplitSampler", indices, data["target"])


if __name__ == "__main__":
    run_stratified_sampler()
