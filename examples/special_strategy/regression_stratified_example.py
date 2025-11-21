"""Демонстрация RegressionStratifiedSampler на регрессионных данных."""

import pathlib
import sys
from core.api.api_main import SamplingStrategyFactory
from core.utils.synt_data import create_synt_tabular_data
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

DATASET_SAMPLES = 2_000
strategy_regression_params = dict(n_bins=4, encode="ordinal", strategy="quantile", n_partitions=3)
strategy_regression_params_one_hot = dict(n_bins=8,
                                          encode="onehot-dense",
                                          strategy="uniform",
                                          n_partitions=3)


def summarize_partitions(strategy, label: str) -> None:
    partitions = strategy.get_partitions()
    print(f"\n{label}: bin edges = {strategy.bin_edges_}")
    for name, indices in partitions.items():
        bin_counts = pd.Series(strategy.binned_target_[indices]).value_counts().sort_index()
        print(f"{name}: size={len(indices)}, bin distribution={bin_counts.to_dict()}")


def run_regression_stratified_example() -> None:
    features, target = create_synt_tabular_data(DATASET_SAMPLES, n_features=6, problem_type="regression")
    factory = SamplingStrategyFactory()

    quantile_sampler = factory.create_strategy(strategy_type="regression_stratified", **strategy_regression_params)
    quantile_sampler.fit(features, data_target=target)
    summarize_partitions(quantile_sampler, "Quantile binning (4 bins)")

    uniform_sampler = factory.create_strategy(strategy_type="regression_stratified",
                                              **strategy_regression_params_one_hot)
    uniform_sampler.fit(features, data_target=target)
    summarize_partitions(uniform_sampler, "Uniform binning (8 bins)")
    _ = 1


if __name__ == "__main__":
    run_regression_stratified_example()
