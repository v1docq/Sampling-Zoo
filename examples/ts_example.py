import sys
import os

from core.utils.synt_data import create_synt_time_series_data

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.api.api_main import SamplingStrategyFactory

TEMPORAL_STRATEGY_LIST = ['seasonal', 'sliding_window', 'sequential']



def split_with_api(data, strategy:str):
    """Демонстрация использования API"""
    print("\n=== Factory Pattern Demo ===")
    splitter = SamplingStrategyFactory()
    strategy = splitter.create_strategy('temporal_split', n_splits=3, method=strategy)
    strategy.fit(data, time_column='timestamp', series_id_column='series_id')
    partitions = strategy.get_partitions(data)
    return partitions


if __name__ == "__main__":
    data = create_synt_time_series_data()
    result_dict = {strategy:split_with_api(data, strategy) for strategy in TEMPORAL_STRATEGY_LIST}

