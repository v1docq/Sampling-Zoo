"""Балансировка классов с помощью BalanceSampler."""

import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from core.api.api_main import SamplingStrategyFactory
from core.sampling_strategies.base_sampler import HierarchicalStratifiedMixin
from core.utils.synt_data import create_noisy_dataset

DATASET_SAMPLES = 10_000
STRATEGY_TYPE = "balance"
STRATEGY_PARAMS = {"balance_method": "smote", "n_partitions": 4}


def run_balance_sampler() -> None:
    data = create_noisy_dataset(DATASET_SAMPLES)
    factory = SamplingStrategyFactory()
    strategy = factory.create_and_fit(
        strategy_type=STRATEGY_TYPE,
        data=data[["feature_1", "feature_2"]],
        target=data["target"],
        strategy_kwargs=STRATEGY_PARAMS,
    )
    partitions = strategy.get_partitions()
    indices = {name: part["feature"].index.to_numpy() for name, part in partitions.items()}
    HierarchicalStratifiedMixin.print_fold_summary("BalanceSampler", indices, pd.concat(
        [part["target"] for part in partitions.values()]
    ))


if __name__ == "__main__":
    run_balance_sampler()
