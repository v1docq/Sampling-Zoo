"""Пример использования многоуровневого стратифицированного разбиения.

Сценарии демонстрируют:
* работу AdvancedStratifiedSampler с редкими классами;
* повторное использование базового метода в StratifiedSplitSampler;
* проверку, что в каждом фолде присутствуют все классы.
"""

import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from core.sampling_strategies.stratified_sampler import AdvancedStratifiedSampler, StratifiedSplitSampler


def run_advanced_sampler():
    """Разбиение с явным контролем редких классов."""
    rng = np.random.default_rng(0)
    # Четыре класса, два из них редкие
    classes = np.array([0] * 45 + [1] * 40 + [2] * 3 + [3] * 2)
    rng.shuffle(classes)
    features = pd.DataFrame({"feature": rng.normal(size=len(classes))})

    sampler = AdvancedStratifiedSampler(n_splits=5, random_state=0)
    sampler.fit(features, classes, min_samples_per_class=1)
    sampler.print_fold_summary("AdvancedStratifiedSampler", sampler.get_partitions(), pd.Series(classes))


def run_factory_sampler():
    """Использование StratifiedSplitSampler, которое теперь использует базовый метод."""
    rng = np.random.default_rng(1)
    classes = np.array(["apple"] * 12 + ["banana"] * 8 + ["cherry"] * 2)
    rng.shuffle(classes)
    df = pd.DataFrame({"feature_1": rng.normal(size=len(classes)), "target": classes})

    sampler = StratifiedSplitSampler(n_partitions=3, random_state=1)
    sampler.fit(df, target=["feature_1", "target"], data_target=df["target"])
    sampler.print_fold_summary("StratifiedSplitSampler", sampler.partitions, df["target"])

if __name__ == "__main__":
    run_advanced_sampler()
    run_factory_sampler()
