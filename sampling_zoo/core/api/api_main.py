from typing import Dict, Any, Union, List
import numpy as np
import pandas as pd
from sampling_zoo.core.sampling_strategies.base_sampler import BaseSampler
from sampling_zoo.core.sampling_strategies.temporal_sampler import TemporalSplitSampler
from sampling_zoo.core.sampling_strategies.feature_sampler import FeatureBasedClusteringSampler, TSNEClusteringSampler
from sampling_zoo.core.sampling_strategies.diff_sampler import DifficultyBasedSampler, UncertaintySampler
from sampling_zoo.core.sampling_strategies.random_sampler import RandomSplitSampler
from sampling_zoo.core.sampling_strategies.stratified_sampler import (
    AdvancedStratifiedSampler,
    RegressionStratifiedSampler,
    StratifiedSplitSampler,
)
from sampling_zoo.core.sampling_strategies.balance_sampler import StratifiedBalancedSplitSampler
from sampling_zoo.core.sampling_strategies.spectral.spectral_leverage import SpectralLeverageSampler
from sampling_zoo.core.sampling_strategies.spectral.tensor_energy import TensorEnergySampler
from sampling_zoo.core.sampling_strategies.spectral.rmt_contraction_sampler import RMTContractionTensorSampler
from sampling_zoo.core.sampling_strategies.delaunay_sempler import DelaunaySampler
from sampling_zoo.core.sampling_strategies.hdbscan_sampler import HDBScanSampler
from sampling_zoo.core.sampling_strategies.voronoi_sampler import VoronoiSampler
try:
    from sampling_zoo.core.sampling_strategies.kernel_sampler import KernelSampler
except Exception:  # pragma: no cover - optional torch-backed sampler
    KernelSampler = None

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
            'rmt_contraction': RMTContractionTensorSampler,
        }

        # SUBSET SAMPLERS
        self.subset_strategies = {
            'spectral_leverage': SpectralLeverageSampler,
            'tensor_energy': TensorEnergySampler,
        }
        if KernelSampler is not None:
            self.subset_strategies['kernel'] = KernelSampler

        self.strategy_map = {**self.chunking_strategies, **self.subset_strategies}

    @staticmethod
    def _partition_budget_keys() -> set:
        return {'budget_ratio'}

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
                       strategy_kwargs: Dict = None,
                       fit_kwargs: Dict = None) -> BaseSampler:
        """Создает стратегию и сразу обучает её на переданных данных."""
        strategy_kwargs = strategy_kwargs or {}
        fit_kwargs = fit_kwargs or {}
        strategy_kwargs = {
            key: value
            for key, value in strategy_kwargs.items()
            if key not in self._partition_budget_keys()
        }

        strategy = self.create_strategy(strategy_type, **strategy_kwargs)
        if self.is_subset_strategy(strategy_type):
            strategy.fit(data, **fit_kwargs)
            return strategy
        elif self.is_chunking_strategy(strategy_type):
            if target is None or "target" in fit_kwargs:
                strategy.fit(data, **fit_kwargs)
            else:
                strategy.fit(data, target=target, **fit_kwargs)

        return strategy

    def fit_transform(self, strategy_type: str, data: Union[np.ndarray, pd.DataFrame], target: Any = None,
                      strategy_kwargs: Dict = None, fit_kwargs: Dict = None,
                      return_strategy: bool = False) -> Any:
        """Удобный вызов для создания стратегии и получения разбиений или индексов."""
        strategy_kwargs = strategy_kwargs or {}
        budget_kwargs = {
            key: strategy_kwargs[key]
            for key in self._partition_budget_keys()
            if key in strategy_kwargs
        }
        strategy = self.create_and_fit(strategy_type, data, target, strategy_kwargs, fit_kwargs)

        if self.is_subset_strategy(strategy_type):
            return strategy.sample_indices() if not return_strategy else (strategy, strategy.sample_indices())
        elif self.is_chunking_strategy(strategy_type):
            partitions = strategy.get_partitions(data, target) if target is not None else strategy.get_partitions()
            partitions, budget_policy = self._apply_partition_budget(
                partitions=partitions,
                total_rows=len(data),
                budget_ratio=budget_kwargs.get('budget_ratio'),
                random_state=strategy_kwargs.get('random_state'),
            )
            strategy.budget_policy_ = budget_policy
            return partitions if not return_strategy else (strategy, partitions)

    @staticmethod
    def _apply_partition_budget(partitions: Dict[Any, Any],
                                total_rows: int,
                                budget_ratio: Any = None,
                                random_state: Any = None) -> tuple[Dict[Any, Any], Dict[str, Any]]:
        if budget_ratio is None:
            return partitions, {'applied': False}

        budget_ratio = float(budget_ratio)
        if not 0 < budget_ratio <= 1:
            raise ValueError("budget_ratio must be in (0, 1]")

        sizes = {
            name: SamplingStrategyFactory._partition_size(partition_data)
            for name, partition_data in partitions.items()
        }
        sizes = {name: size for name, size in sizes.items() if size > 0}
        if not sizes:
            return partitions, {'applied': False, 'reason': 'empty_partitions'}

        budget_size = max(1, min(total_rows, int(round(total_rows * budget_ratio))))
        current_size = int(sum(sizes.values()))
        if current_size <= budget_size:
            return partitions, {
                'applied': False,
                'budget_ratio': budget_ratio,
                'budget_size': budget_size,
                'current_size': current_size,
            }

        ordered_names = sorted(sizes, key=lambda name: sizes[name], reverse=True)
        if budget_size < len(ordered_names):
            ordered_names = ordered_names[:budget_size]

        ordered_total = sum(sizes[name] for name in ordered_names)
        counts = {
            name: max(1, min(sizes[name], int(np.floor(budget_size * sizes[name] / ordered_total))))
            for name in ordered_names
        }

        while sum(counts.values()) > budget_size:
            candidates = [name for name, count in counts.items() if count > 1]
            if not candidates:
                break
            counts[max(candidates, key=lambda name: counts[name])] -= 1

        while sum(counts.values()) < budget_size:
            candidates = [name for name in ordered_names if counts[name] < sizes[name]]
            if not candidates:
                break
            counts[max(candidates, key=lambda name: sizes[name] - counts[name])] += 1

        rng = np.random.default_rng(random_state)
        budgeted = {}
        for name in ordered_names:
            local_indices = np.sort(rng.choice(np.arange(sizes[name]), size=counts[name], replace=False))
            budgeted[name] = SamplingStrategyFactory._take_partition_rows(partitions[name], local_indices)

        return budgeted, {
            'applied': True,
            'budget_ratio': budget_ratio,
            'budget_size': budget_size,
            'current_size': current_size,
            'selected_size': int(sum(counts.values())),
            'partition_sizes': {name: int(count) for name, count in counts.items()},
        }

    @staticmethod
    def _partition_size(partition_data: Any) -> int:
        if isinstance(partition_data, dict):
            return len(partition_data['feature'])
        return len(partition_data)

    @staticmethod
    def _take_partition_rows(partition_data: Any, local_indices: np.ndarray) -> Any:
        if isinstance(partition_data, dict):
            return {
                key: SamplingStrategyFactory._take_rows(value, local_indices)
                for key, value in partition_data.items()
            }
        return np.asarray(partition_data)[local_indices]

    @staticmethod
    def _take_rows(value: Any, local_indices: np.ndarray) -> Any:
        if isinstance(value, (pd.DataFrame, pd.Series)):
            return value.iloc[local_indices].reset_index(drop=True)
        return np.asarray(value)[local_indices]

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
