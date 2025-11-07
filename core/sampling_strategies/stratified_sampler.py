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
    
    def fit(self, data: pd.DataFrame, target: list[str], data_target: list[str] = None):
        """
        Args:
            data: Матрица признаков или сырые данные
            strat_target: Переменные, для которых будет сохранено распределение
            data_target: Целевая переменная (опционально)
        """
        mskf = MultilabelStratifiedKFold(n_splits=self.n_partitions, shuffle=True, random_state=self.random_state)
        
        processed_data = pd.DataFrame(index=data.index)
        
        for name in target:
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
            self.partitions[f'chunk_{i}'] = part_idx

        return self

    def get_partitions(self, data, target) -> Dict[Any, np.ndarray]:
        partition = {cluster: dict(feature=data.iloc[idx],
                                   target=target.iloc[idx]) for cluster, idx in self.partitions.items()}
        return partition

    def check_partitions(self, partitions, data):
        print("Partition statistics:")
        feature_names = data.columns.to_list()
        feature_names = [x for x in feature_names if not x.__contains__('target')]
        for name, part in partitions.items():
            for feat in feature_names:
                indices = part['feature'].index.to_numpy()
                partition_data = data.iloc[indices]
                partition_data = partition_data[feat]
                print(f"\n{name} ({len(indices)} samples):")
                print(f"{feat}:")
                print("  Mean: {:.3f}, Std: {:.3f}, Var: {:.3f}".format(
              partition_data.mean(),
                    partition_data.std(),
                    partition_data.var()))

