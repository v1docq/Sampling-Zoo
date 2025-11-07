import numpy as np
import pandas as pd
from typing import Dict, Any, Union
from sklearn.model_selection import StratifiedKFold
from .base_sampler import BaseSampler
from ..repository.model_repo import SamplingModels


class StratifiedBalancedSplitSampler(BaseSampler):
    """
    Семплер, который выполняет стратифицированное разбиение,
    а затем применяет к каждому чанку метод балансировки
    """

    def __init__(self, n_partitions: int = 5, random_state: int = 42,
                 balance_method: str = None, balancer_kwargs: Dict = None):
        self.n_partitions = n_partitions
        self.random_state = random_state
        self.BALANCERS = SamplingModels.balance_samplers.value
        if balance_method and balance_method.lower() not in self.BALANCERS:
            raise ValueError(f"Unknown method '{balance_method}'. Available methods: {list(self.BALANCERS.keys())}")

        self.balance_method_name = balance_method.lower() if balance_method else None
        self.balancer_kwargs = balancer_kwargs if balancer_kwargs is not None else {}
        self.partitions = {}

    def fit(self, data: pd.DataFrame, target: Union[pd.Series, np.ndarray]):
        """Выполняет разбиение и балансировку."""

        # Определяем, является ли задача классификацией
        if len(np.unique(target)) < 0.1 * len(target) or target.dtype == 'object':
            print("Warning: Stratified balancing is intended for classification tasks.")

        if isinstance(target, np.ndarray):
            target = pd.Series(target, index=data.index)

        skf = StratifiedKFold(n_splits=self.n_partitions, shuffle=True, random_state=self.random_state)

        for i, (_, part_idx) in enumerate(skf.split(data, target)):
            data_chunk = data.iloc[part_idx]
            target_chunk = target.iloc[part_idx]

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

    def get_partitions(self) -> Dict[str, tuple[pd.DataFrame, pd.Series]]:
        """Возвращает словарь с чанками, к которым применялся метод балансировки."""
        return self.partitions
