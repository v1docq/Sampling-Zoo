import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.utils.synt_data import create_synt_tabular_data



from core.api.api_main import SamplingStrategyFactory

TABULAR_STRATEGY_LIST = {
    'feature_clustering': dict(n_clusters=4, method='kmeans'),
    'difficulty': dict(difficulty_threshold=0.7),
    'uncertainty': dict(uncertainty_threshold=0.7)
}


def split_with_api(features, target, strategy: str, strategy_params: dict):
    """Демонстрация использования API"""
    print("\n=== Factory Pattern Demo ===")
    splitter = SamplingStrategyFactory()
    strategy = splitter.create_and_fit(strategy, data=features, target=target, strategy_kwargs=strategy_params)
    return strategy.get_partitions(features, target)


if __name__ == "__main__":
    features, target = create_synt_tabular_data()
    result_dict = {strategy: split_with_api(features, target, strategy, strategy_params)
                   for strategy, strategy_params in TABULAR_STRATEGY_LIST.items()}
