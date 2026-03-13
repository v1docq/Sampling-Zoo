from typing import Dict, Any, Union, List
import numpy as np
import pandas as pd
from core.sampling_strategies.base_sampler import BaseSampler
from core.sampling_strategies.temporal_sampler import TemporalSplitSampler
from core.sampling_strategies.feature_sampler import FeatureBasedClusteringSampler, TSNEClusteringSampler
from core.sampling_strategies.diff_sampler import DifficultyBasedSampler, UncertaintySampler
from core.sampling_strategies.random_sampler import RandomSplitSampler
from core.sampling_strategies.stratified_sampler import (
    AdvancedStratifiedSampler,
    RegressionStratifiedSampler,
    StratifiedSplitSampler,
)
from core.sampling_strategies.balance_sampler import StratifiedBalancedSplitSampler
from core.sampling_strategies.spectral.spectral_leverage import SpectralLeverageSampler
from core.sampling_strategies.spectral.tensor_energy import TensorEnergySampler
from core.sampling_strategies.delaunay_sempler import DelaunaySampler
from core.sampling_strategies.hdbscan_sampler import HDBScanSampler
from core.sampling_strategies.voronoi_sampler import VoronoiSampler
from core.sampling_strategies.kernel_sampler import KernelSampler

class SamplingStrategyFactory:
    """
    Фабрика для создания стратегий семплирования
    """
    def __init__(self):
        # CHUNKING SAMPLERS
        self.chunking_strategies = {
            # Random split
            'random': RandomSplitSampler,

            # Stratified Sampling
            'stratified': StratifiedSplitSampler,
            'advanced_stratified': AdvancedStratifiedSampler,
            'regression_stratified': RegressionStratifiedSampler,

            # Temporal strategies
            'temporal_split': TemporalSplitSampler,
            'temporal': TemporalSplitSampler,

            # Difficulty-based strategies
            'difficulty': DifficultyBasedSampler,
            'uncertainty': UncertaintySampler,

            # Class balance strategies
            'balance': StratifiedBalancedSplitSampler,

            # Clustering-based strategies
            'feature_clustering': FeatureBasedClusteringSampler,
            'tsne_clustering': TSNEClusteringSampler,
            'delaunay': DelaunaySampler,
            'hdbscan': HDBScanSampler,
            'voronoi': VoronoiSampler,
        }

        # SUBSET SAMPLERS
        self.subset_strategies = {
            'spectral_leverage': SpectralLeverageSampler,
            'tensor_energy': TensorEnergySampler,
            'kernel': KernelSampler,
        }

        self.strategy_map = {**self.chunking_strategies, **self.subset_strategies}

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

    def create_and_fit(self, strategy_type: str, data: Union[np.ndarray, pd.DataFrame], target: Any = None,
                       strategy_kwargs: Dict[str, Any] | None = None,
                       fit_kwargs: Dict[str, Any] | None = None) -> BaseSampler:
        """Создает стратегию и сразу обучает её на переданных данных."""
        strategy_kwargs = strategy_kwargs or {}
        fit_kwargs = fit_kwargs or {}

        strategy = self.create_strategy(strategy_type, **strategy_kwargs)
        if target is None:
            strategy.fit(data, **fit_kwargs)
        else:
            strategy.fit(data, target=target, **fit_kwargs)

        return strategy

    def fit_transform(self, strategy_type: str, data: Union[np.ndarray, pd.DataFrame], target: Any = None,
                      strategy_kwargs: Dict[str, Any] | None = None,
                      fit_kwargs: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """Удобный вызов для создания стратегии и получения разбиений."""
        strategy = self.create_and_fit(strategy_type, data, target, strategy_kwargs, fit_kwargs)
        return strategy.get_partitions(data, target) if target is not None else strategy.get_partitions()

    def get_available_strategies(self) -> List[str]:
        """Возвращает список доступных стратегий"""
        return sorted(list(self.strategy_map.keys()))

    def get_chunking_strategies(self) -> List[str]:
        """Возвращает список чанковых стратегий"""
        return sorted(list(self.chunking_strategies.keys()))

    def get_subset_strategies(self) -> List[str]:
        """Возвращает список subset стратегий"""
        return sorted(list(self.subset_strategies.keys()))

    def is_chunking_strategy(self, strategy_type: str) -> bool:
        """Проверяет, является ли стратегия чанковой"""
        return strategy_type in self.chunking_strategies

    def is_subset_strategy(self, strategy_type: str) -> bool:
        """Проверяет, является ли стратегия subset"""
        return strategy_type in self.subset_strategies

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