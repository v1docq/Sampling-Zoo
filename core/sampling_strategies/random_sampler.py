import numpy as np
import pandas as pd
from typing import Dict, Any, Union, Optional


from .base_sampler import BaseSampler, HierarchicalStratifiedMixin
from ..utils.utils import to_dataframe


class RandomSplitSampler(BaseSampler, HierarchicalStratifiedMixin):
    """
    Семплирование с рандомным распределением по поднаборам    
    """
    def __init__(self, n_partitions: int = 5, random_state: int = 42):
        BaseSampler.__init__(self, random_state=random_state)
        HierarchicalStratifiedMixin.__init__(
            self,
            n_partitions=n_partitions,
            random_state=random_state,
            logger_name="RandomSplitSampler",
        )
        self.n_partitions = n_partitions
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

        for i in range(self.n_partitions):
            self.partitions[f'partition_{i}'] = partitions[i]

        return self

    def get_partitions(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        target: Union[np.ndarray, pd.Series],
    ) -> Dict[Any, np.ndarray]:
        return self._get_partitions_default(data=data, target=target)
