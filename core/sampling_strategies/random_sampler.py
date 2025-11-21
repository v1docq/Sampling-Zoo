import numpy as np
import pandas as pd
from typing import Dict, Any, Union


from .base_sampler import BaseSampler, HierarchicalStratifiedMixin


class RandomSplitSampler(BaseSampler, HierarchicalStratifiedMixin):
    """
    Семплирование с рандомным распределением по поднаборам    
    """
    def __init__(self, n_partitions: int = 5, random_state: int = 42):
        BaseSampler.__init__(self, random_state=random_state)
        HierarchicalStratifiedMixin.__init__(
            self,
            n_splits=n_partitions,
            random_state=random_state,
            logger_name="RandomSplitSampler",
        )
        self.n_partitions = n_partitions
        self.partitions = {}

    def fit(self, data: pd.DataFrame, target: np.ndarray = None):
        """
        Args:
            data: Матрица признаков или сырые данные
            target: Целевая переменная (опционально)
        """
        indices = np.arange(len(data))
        rng = np.random.default_rng(self.random_state)
        rng.shuffle(indices)
        partitions = np.array_split(indices, self.n_partitions)

        for i in range(self.n_partitions):
            self.partitions[f'partition_{i}'] = partitions[i]

        return self

    def get_partitions(self, data, target) -> Dict[Any, np.ndarray]:
        partition = {cluster: dict(feature=data.iloc[idx],
                                   target=target[idx]) for cluster, idx in self.partitions.items()}
        return partition
