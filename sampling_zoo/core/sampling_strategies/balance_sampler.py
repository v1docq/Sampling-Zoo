import numpy as np
import pandas as pd
from typing import Dict, Union, Optional
from sklearn.model_selection import StratifiedKFold
from .base_sampler import BaseSampler, HierarchicalStratifiedMixin
from ..utils.utils import to_dataframe, to_series
from ..repository.model_repo import SamplingModels


class StratifiedBalancedSplitSampler(BaseSampler, HierarchicalStratifiedMixin):
    """
    Семплер, который выполняет стратифицированное разбиение,
    а затем применяет к каждому чанку метод балансировки
    """

    def __init__(self, n_partitions: int = 5, random_state: int = 42,
                 balance_method: str = None, balancer_kwargs: Dict = None):
        BaseSampler.__init__(self, random_state=random_state)
        HierarchicalStratifiedMixin.__init__(
            self,
            n_partitions=n_partitions,
            random_state=random_state,
            logger_name="StratifiedBalancedSplitSampler",
        )
        self.n_partitions = n_partitions
        self.BALANCERS = SamplingModels.balance_samplers.value
        if balance_method and balance_method.lower() not in self.BALANCERS:
            raise ValueError(f"Unknown method '{balance_method}'. Available methods: {list(self.BALANCERS.keys())}")

        self.balance_method_name = balance_method.lower() if balance_method else None
        self.balancer_kwargs = balancer_kwargs if balancer_kwargs is not None else {}
        self.partitions = {}

    def fit(self, data: Union[np.ndarray, pd.DataFrame], target: Union[pd.Series, np.ndarray]):
        """Выполняет разбиение и балансировку."""

        data_df = to_dataframe(data)
        target_series = to_series(target, index=data_df.index)

        # Определяем, является ли задача классификацией
        if target_series.nunique() < 0.1 * len(target_series) or target_series.dtype == 'object':
            print("Warning: Stratified balancing is intended for classification tasks.")

        skf = StratifiedKFold(n_splits=self.n_partitions, shuffle=True, random_state=self.random_state)

        for i, (_, part_idx) in enumerate(skf.split(data_df, target_series)):
            data_chunk = data_df.iloc[part_idx]
            target_chunk = target_series.iloc[part_idx]

            if self.balance_method_name:
                balancer_class = self.BALANCERS[self.balance_method_name]

                kwargs = self.balancer_kwargs.copy()

                # Для SMOTE: k_neighbors не может быть больше числа сэмплов - 1
                if self.balance_method_name == 'smote' and 'k_neighbors' not in kwargs:
                    min_samples = target_chunk.value_counts().min()
                    kwargs['k_neighbors'] = max(1, min_samples - 1)

                # Инициализируем балансировщик с нужными параметрами
                # Пытаемся передать random_state, если алгоритм его поддерживает
                try:
                    balancer = balancer_class(random_state=self.random_state, **kwargs)
                except TypeError:
                    balancer = balancer_class(**kwargs)

                data_balanced, target_balanced = balancer.fit_resample(data_chunk, target_chunk)
                self.partitions[f'partition_{i}'] = dict(feature=data_balanced, target=target_balanced)
            else:
                self.partitions[f'partition_{i}'] = dict(feature=data_chunk, target=target_chunk)

        return self

    def get_partitions(
        self,
        data: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        target: Optional[Union[np.ndarray, pd.Series]] = None,
    ) -> Dict[str, tuple[pd.DataFrame, pd.Series]]:
        """Возвращает словарь с чанками, к которым применялся метод балансировки."""
        return self._get_partitions_default(data=data, target=target)
