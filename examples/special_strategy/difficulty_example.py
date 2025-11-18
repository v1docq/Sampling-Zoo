"""Пример DifficultySampler с единым оформлением."""

import pathlib
import sys

import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from core.api.api_main import SamplingStrategyFactory
from core.sampling_strategies.base_sampler import HierarchicalStratifiedMixin
from core.utils.synt_data import create_sklearn_dataset

TASK_TYPE = "classification"
STRATEGY_TYPE = "difficulty"
STRATEGY_PARAMS = {"n_partitions": 3, "random_state": 42}


def run_difficulty_sampler() -> None:
    data = create_sklearn_dataset(TASK_TYPE)
    features = data[["feature_1", "feature_2"]]
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(features, data["target"])

    strategy_params = STRATEGY_PARAMS | {"model": model}
    factory = SamplingStrategyFactory()
    strategy = factory.create_and_fit(
        strategy_type=STRATEGY_TYPE,
        data=features,
        target=data["target"],
        strategy_kwargs=strategy_params,
    )
    partitions = strategy.get_partitions(features, target=data["target"])
    indices = {name: part["feature"].index.to_numpy() for name, part in partitions.items()}
    HierarchicalStratifiedMixin.print_fold_summary("DifficultySampler", indices, data["target"])


if __name__ == "__main__":
    run_difficulty_sampler()
