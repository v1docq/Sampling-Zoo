import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from core.utils.synt_data import create_synt_time_series_data, create_synt_tabular_data

from core.api.api_main import SamplingStrategyFactory

TEMPORAL_STRATEGY_LIST = ['seasonal', 'sliding_window', 'sequential']


def split_with_api(data, strategy: str):
    """Демонстрация использования API"""
    print("\n=== Factory Pattern Demo ===")
    splitter = SamplingStrategyFactory()
    trained_strategy = splitter.create_and_fit(
        'temporal_split',
        data,
        strategy_kwargs={'n_splits': 3, 'method': strategy},
        fit_kwargs={'time_column': 'timestamp', 'series_id_column': 'series_id'},
    )
    return trained_strategy.get_partitions(data)


if __name__ == "__main__":
    data = create_synt_tabular_data()
    result_dict = {strategy: split_with_api(data, strategy) for strategy in TEMPORAL_STRATEGY_LIST}
