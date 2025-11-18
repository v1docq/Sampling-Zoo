"""Демонстрация RandomSampler в едином формате примеров.

Сценарий показывает:
* создание семплирования через ``SamplingStrategyFactory``;
* вывод распределения классов в полученных фолдах;
* минимальный рабочий пример с синтетическими данными.
"""

import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from core.api.api_main import SamplingStrategyFactory
from core.sampling_strategies.base_sampler import HierarchicalStratifiedMixin
from core.utils.synt_data import create_noisy_dataset


def _summarize_partitions(partitions: dict[str, dict], target: pd.Series) -> None:
    indices = {name: part["feature"].index.to_numpy() for name, part in partitions.items()}
    HierarchicalStratifiedMixin.print_fold_summary("RandomSampler", indices, target)


def run_random_sampler(samples: int = 10_000) -> None:
    data = create_noisy_dataset(samples)
    factory = SamplingStrategyFactory()
    partitions = factory.fit_transform(
        "random",
        data[["feature_1", "feature_2"]],
        target=data["target"],
        strategy_kwargs={"n_partitions": 3},
    )

    _summarize_partitions(partitions, data["target"])


if __name__ == "__main__":
    run_random_sampler()
