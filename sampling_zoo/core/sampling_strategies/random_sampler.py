import math
import numpy as np
import pandas as pd
from typing import Dict, Any, Union, Optional


from .base_sampler import BaseSampler, HierarchicalStratifiedMixin
from ..utils.utils import to_dataframe


class RandomSplitSampler(BaseSampler, HierarchicalStratifiedMixin):
    """
    Семплирование с рандомным распределением по поднаборам    
    """
    def __init__(self, n_partitions: int = 5, random_state: int = 42, chunks_percent: int = 100):
        BaseSampler.__init__(self, random_state=random_state)
        HierarchicalStratifiedMixin.__init__(
            self,
            n_partitions=n_partitions,
            random_state=random_state,
            logger_name="RandomSplitSampler",
        )
        self.n_partitions = n_partitions
        self.chunks_percent = chunks_percent
        self.partitions = {}

    def fit(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        target: Optional[Union[pd.Series, np.ndarray]] = None,
    ):
        """
        Args:
            data: Матрица признаков или сырые данные
            target: Целевая переменная (опционально)
        """
        data_df = to_dataframe(data)
        indices = np.arange(len(data_df))
        rng = np.random.default_rng(self.random_state)
        rng.shuffle(indices)
        partitions = np.array_split(indices, self.n_partitions)

        if self.chunks_percent < 100:
            chunks_to_keep = max(1, int(math.ceil(self.n_partitions * self.chunks_percent / 100)))
            partitions = partitions[:chunks_to_keep]
            self.n_partitions = chunks_to_keep

        for i in range(len(partitions)):
            self.partitions[f'partition_{i}'] = partitions[i]

        return self

    def get_partitions(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        target: Union[np.ndarray, pd.Series],
    ) -> Dict[Any, np.ndarray]:
        return self._get_partitions_default(data=data, target=target)
