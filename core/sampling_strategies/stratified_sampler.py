from collections import Counter
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold, RepeatedMultilabelStratifiedKFold, IterativeStratification
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import check_random_state

from .base_sampler import BaseSampler, HierarchicalStratifiedMixin

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
        if len(data_target.shape) >= 2:
            mskf = MultilabelStratifiedKFold(n_splits=self.n_partitions, shuffle=True, random_state=self.random_state)
            # Кодируем целевой признак для классификации
            le = OneHotEncoder()
            data_target = le.fit_transform(data_target.values.reshape(-1, 1))
            data_target = data_target.toarray()
            target = [x for x in target if not x.__contains__('target')]
            processed_data = pd.DataFrame(index=data.index)
            splits = mskf.iterative_stratified_split(processed_data, data_target)
            for name in target:
                # Если число уникальных слагаемых в столбце больше порога, то делим его на квантили
                if len(data[name].unique()) / len(data) > self.uniqueness_threshold:
                    processed_data[name] = pd.qcut(data[name], q=self.n_partitions, labels=False, duplicates='drop')
                else:
                    processed_data[name] = data[name]

            multilabel = pd.get_dummies(processed_data)
            split_iteration = mskf.split(multilabel.values, data_target.values)
        else:
            stratified_sampler = AdvancedStratifiedSampler(n_splits=self.n_partitions, random_state=self.random_state)
            stratified_sampler.fit(data, data_target.values)
            split_iteration = stratified_sampler.get_partitions()

        for i, part_idx in enumerate(split_iteration.values() if isinstance(split_iteration, dict) else split_iteration):
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


class IterativeStratifiedSampler:
    """
    Итеративный стратифицированный семплер для очень несбалансированных данных
    """

    def __init__(self, n_splits: int = 100, random_state: int = 42):
        self.n_splits = n_splits
        self.random_state = random_state
        np.random.seed(random_state)

    def iterative_stratified_split(self, features: np.ndarray, target: np.ndarray):
        """
        Итеративный алгоритм стратифицированного разбиения
        Основан на алгоритме итеративного стратифицирования из sklearn
        """

        random_state, n_samples = check_random_state(self.random_state), len(target)

        # Инициализируем фолды
        folds = [np.array([], dtype=int) for _ in range(self.n_splits)]
        fold_sizes = np.zeros(self.n_splits, dtype=int)

        # Целевой размер фолда
        target_fold_size = round(n_samples // self.n_splits)

        # Группируем индексы по классам
        class_indices = {}
        for class_label in np.unique(target):
            class_indices[class_label] = np.where(target == class_label)[0]

        # Сортируем классы по возрастанию частоты (сначала редкие)
        sorted_classes = sorted(class_indices.keys(),
                                key=lambda x: len(class_indices[x]))

        # Распределяем samples итеративно
        for class_label in sorted_classes:
            indices = class_indices[class_label]
            n_class_samples = len(indices)

            # Перемешиваем индексы класса
            random_state.shuffle(indices)

            # Распределяем по фолдам
            samples_assigned = 0

            while samples_assigned < n_class_samples:
                # Находим фолд с наименьшим количеством samples
                min_fold_idx = np.argmin(fold_sizes)

                # Добавляем один sample в этот фолд
                if samples_assigned < n_class_samples:
                    sample_idx = indices[samples_assigned]
                    folds[min_fold_idx] = np.append(folds[min_fold_idx], sample_idx)
                    fold_sizes[min_fold_idx] += 1
                    samples_assigned += 1

        # Создаем train/test сплиты
        return self._create_splits(n_samples, folds)

    def _create_splits(self, n_samples, folds):

        # Создаем train/test сплиты
        splits = []
        all_indices = np.arange(n_samples)

        for test_indices in folds:
            train_indices = np.setdiff1d(all_indices, test_indices)
            splits.append((train_indices, test_indices))

        return splits

    def split(self):
        for train, test in super().split(X, y, groups):
            yield train, test


class AdvancedStratifiedSampler(BaseSampler, HierarchicalStratifiedMixin):
    """
    Продвинутый стратифицированный семплер для многоклассовых задач
    с гарантированным присутствием всех классов в каждом фолде
    """

    def __init__(self, n_splits: int = 100, random_state: int = 42):
        BaseSampler.__init__(self, random_state=random_state)
        HierarchicalStratifiedMixin.__init__(self, n_splits=n_splits, random_state=random_state)
        self.partitions = {}

    def fit(self, data: pd.DataFrame, target: Union[pd.Series, np.ndarray], min_samples_per_class: int = 2):
        folds = self.hierarchical_stratified_split(data, np.asarray(target), min_samples_per_class)
        self.partitions = {f'chunk_{i}': fold_indices for i, fold_indices in enumerate(folds)}
        return self

    def get_partitions(self) -> Dict[str, np.ndarray]:
        return self.partitions

    def get_fold_statistics(self, folds: List[np.ndarray], y: np.ndarray) -> pd.DataFrame:
        stats = []

        for i, fold_indices in enumerate(folds):
            fold_classes = Counter(y[fold_indices])
            stats.append({
                'fold': i,
                'test_size': len(fold_indices),
                'test_n_classes': len(fold_classes),
                'test_min_class': min(fold_classes.values()) if fold_classes else 0,
                'test_max_class': max(fold_classes.values()) if fold_classes else 0
            })

        return pd.DataFrame(stats)