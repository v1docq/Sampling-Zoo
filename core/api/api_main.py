from typing import Dict, Any, Union, List
import numpy as np
import pandas as pd
from core.sampling_strategies.base_sampler import BaseSampler
from core.sampling_strategies.temporal_sampler import TemporalSplitSampler
from core.sampling_strategies.feature_sampler import FeatureBasedClusteringSampler, TSNEClusteringSampler
from core.sampling_strategies.diff_sampler import DifficultyBasedSampler, UncertaintySampler
from core.sampling_strategies.random_sampler import RandomSplitSampler
from core.sampling_strategies.stratified_sampler import StratifiedSplitSampler

class SamplingStrategyFactory:
    """
    Фабрика для создания стратегий семплирования
    """
    def __init__(self):
        self.strategy_map = {

            # Random split
            'random_split': RandomSplitSampler,

            #Stratified Sampling
            'stratified_split': StratifiedSplitSampler,

            # Temporal strategies
            'temporal_split': TemporalSplitSampler,

            # Feature-based strategies
            'feature_clustering': FeatureBasedClusteringSampler,
            'tsne_clustering': TSNEClusteringSampler,

            # Difficulty-based strategies
            'difficulty': DifficultyBasedSampler,
            'uncertainty': UncertaintySampler,
        }

    def create_strategy(self, strategy_type: str, **kwargs) -> BaseSampler:
        """
        Создает стратегию семплирования по названию

        Args:
            strategy_type: Тип стратегии
            **kwargs: Параметры для стратегии

        Returns:
            Объект стратегии семплирования
        """


        if strategy_type not in self.strategy_map:
            raise ValueError(f"Unknown strategy type: {strategy_type}. "
                             f"Available: {list(self.strategy_map.keys())}")

        return self.strategy_map[strategy_type](**kwargs)

    @staticmethod
    def get_available_strategies() -> List[str]:
        """Возвращает список доступных стратегий"""
        return ['temporal_split', 'feature_clustering',
                'tsne_clustering', 'difficulty', 'uncertainty']

class AdaptiveSampler:
    """
    Адаптивный семплер, который автоматически выбирает стратегию
    """

    def __init__(self):
        self.strategy = None
        self.data_type = None

    def auto_select_strategy(self, data: Union[np.ndarray, pd.DataFrame],
                             target: np.ndarray = None) -> BaseSampler:
        """
        Автоматически выбирает стратегию на основе характеристик данных
        """
        # Простая эвристика для выбора стратегии
        if isinstance(data, pd.DataFrame):
            if 'timestamp' in data.columns:
                self.data_type = 'time_series'
                self.strategy = TemporalSplitSampler()
            else:
                self.data_type = 'tabular'
                if target is not None:
                    self.strategy = FeatureBasedClusteringSampler()
                else:
                    self.strategy = FeatureBasedClusteringSampler(method='dbscan')
        else:
            self.data_type = 'array'
            self.strategy = FeatureBasedClusteringSampler()

        return self.strategy