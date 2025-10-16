from typing import Dict, Any, Union, List
import numpy as np
import pandas as pd
from .base_sampler import BaseSampler
from .temporal_samplers import TemporalSplitSampler, SeasonalSampler
from .feature_based_samplers import FeatureBasedClusteringSampler, TSNEClusteringSampler
from .difficulty_samplers import DifficultyBasedSampler, UncertaintySampler


class SamplingStrategyFactory:
    """
    Фабрика для создания стратегий семплирования
    """

    @staticmethod
    def create_strategy(strategy_type: str, **kwargs) -> BaseSampler:
        """
        Создает стратегию семплирования по названию

        Args:
            strategy_type: Тип стратегии
            **kwargs: Параметры для стратегии

        Returns:
            Объект стратегии семплирования
        """
        strategy_map = {
            # Temporal strategies
            'temporal_split': TemporalSplitSampler,
            'seasonal': SeasonalSampler,

            # Feature-based strategies
            'feature_clustering': FeatureBasedClusteringSampler,
            'tsne_clustering': TSNEClusteringSampler,

            # Difficulty-based strategies
            'difficulty': DifficultyBasedSampler,
            'uncertainty': UncertaintySampler,
        }

        if strategy_type not in strategy_map:
            raise ValueError(f"Unknown strategy type: {strategy_type}. "
                             f"Available: {list(strategy_map.keys())}")

        return strategy_map[strategy_type](**kwargs)

    @staticmethod
    def get_available_strategies() -> List[str]:
        """Возвращает список доступных стратегий"""
        return ['temporal_split', 'seasonal', 'feature_clustering',
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