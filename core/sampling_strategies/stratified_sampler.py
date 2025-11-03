import numpy as np
import pandas as pd
from typing import Dict, Any, Union
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


from .base_sampler import BaseSampler

class StratifiedSplitSampler(BaseSampler):
    """
    Семплирование с сохранением распределений в указанных классах
    """
    def __init__(self, n_partitions: int = 5, random_state: int = 42, uniqueness_threshold: int = 0.3):
        self.n_partitions = n_partitions
        self.random_state = random_state
        self.uniqueness_threshold = uniqueness_threshold
        self.partitions = {}
    
    def fit(self, data: pd.DataFrame, strat_target: list[str], data_target: list[str] = None):
        """
        Args:
            data: Матрица признаков или сырые данные
            strat_target: Переменные, для которых будет сохранено распределение
            data_target: Целевая переменная (опционально)
        """
        mskf = MultilabelStratifiedKFold(n_splits=self.n_partitions, shuffle=True, random_state=self.random_state)
        
        processed_data = pd.DataFrame(index=data.index)
        
        for name in strat_target:
            # Если число уникальных слагаемых в столбце больше порога, то делим его на квантили 
            if len(data[name].unique()) / len(data) > self.uniqueness_threshold:
                processed_data[name] = pd.qcut(
                    data[name], 
                    q=self.n_partitions, 
                    labels=False,
                    duplicates='drop' 
                )
            else: 
                processed_data[name] = data[name]

        multilabel = pd.get_dummies(processed_data)

        for i, (_, part_idx) in enumerate(mskf.split(data, multilabel)):
            self.partitions[f'partition_{i}'] = part_idx

        return self

    def get_partitions(self, data, target) -> Dict[Any, np.ndarray]:
        partition = {cluster: dict(feature=data.iloc[idx],
                                   target=target[idx]) for cluster, idx in self.partitions.items()}
        return partition
            