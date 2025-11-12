import numpy as np
import pandas as pd
from typing import Dict, Any, Union
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold, RepeatedMultilabelStratifiedKFold, \
    IterativeStratification
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, StratifiedGroupKFold
from sklearn.preprocessing import OneHotEncoder
from .base_sampler import BaseSampler
from sklearn.utils import check_random_state
from collections import defaultdict, Counter
from sklearn.model_selection import StratifiedShuffleSplit
from typing import List, Dict, Tuple, Optional
import logging

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
            mskf = AdvancedStratifiedSampler(n_splits=self.n_partitions, random_state=self.random_state)
            split_iteration = mskf.hierarchical_stratified_split(data,data_target.values)

        for i, part_idx in enumerate(split_iteration):
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


class AdvancedStratifiedSampler:
    """
    Продвинутый стратифицированный семплер для многоклассовых задач
    с гарантированным присутствием всех классов в каждом фолде
    """

    def __init__(self, n_splits: int = 100, random_state: int = 42):
        self.n_splits = n_splits
        self.random_state = random_state
        self.logger = self._setup_logger()
        # Используем StratifiedShuffleSplit для частых классов
        self.stratification_model = StratifiedShuffleSplit(n_splits=self.n_splits,
                                                           test_size=1 / self.n_splits, random_state=self.random_state)

    def _setup_logger(self):
        logger = logging.getLogger('StratifiedSampler')
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def analyze_class_distribution(self, y: np.ndarray) -> Dict:
        """Анализирует распределение классов"""
        class_counts = Counter(y)
        total_samples = len(y)

        analysis = {
            'n_classes': len(class_counts),
            'total_samples': total_samples,
            'class_distribution': class_counts,
            'min_class_count': min(class_counts.values()),
            'max_class_count': max(class_counts.values()),
            'problematic_classes': []
        }

        # Определяем проблемные классы (слишком редкие для n_splits)
        min_samples_per_fold = round(total_samples / self.n_splits)
        for class_label, count in class_counts.items():
            if count < self.n_splits:  # Меньше 1 sample на фолд в среднем
                analysis['problematic_classes'].append((class_label, count))

        self.logger.info(f"Анализ распределения: {analysis['n_classes']} классов")
        self.logger.info(f"Минимальный класс: {analysis['min_class_count']} samples")
        self.logger.info(f"Проблемные классы: {len(analysis['problematic_classes'])}")

        return analysis

    def hierarchical_stratified_split(self, X: pd.DataFrame, y: np.ndarray, min_samples_per_class: int = 2)\
            -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Многоуровневый стратифицированный сплит с гарантией всех классов в каждом фолде

        Args:
            X: Признаки
            y: Целевые переменные
            min_samples_per_class: Минимальное количество samples каждого класса в фолде

        Returns:
            List of (train_indices, test_indices) for each fold
        """

        # Анализируем распределение
        analysis = self.analyze_class_distribution(y)

        # Шаг 1: Разделяем классы на группы по частоте
        frequent_classes, rare_classes = self._separate_classes_by_frequency(y, analysis, min_samples_per_class)

        self.logger.info(f"Частые классы: {len(frequent_classes)}, Редкие классы: {len(rare_classes)}")

        # Шаг 2: Создаем базовые фолды только для частых классов
        base_folds = self._create_base_folds(y, frequent_classes)

        # Шаг 3: Распределяем редкие классы по фолдам
        final_folds = self._distribute_rare_classes(X, y, base_folds, rare_classes,
                                                    min_samples_per_class)

        # Шаг 4: Валидация результатов
        self._validate_folds(final_folds, y, analysis['n_classes'])

        return final_folds

    def _separate_classes_by_frequency(self, y: np.ndarray, analysis: Dict,
                                       min_samples: int) -> Tuple[List, List]:
        """Разделяет классы на частые и редкие"""
        frequent_classes = []
        rare_classes = []

        for class_label, count in analysis['class_distribution'].items():
            # Класс считается редким если его меньше чем n_splits * min_samples
            if count < self.n_splits * min_samples:
                rare_classes.append(class_label)
            else:
                frequent_classes.append(class_label)

        return frequent_classes, rare_classes

    def _create_base_folds(self, y: np.ndarray, frequent_classes: List) -> List[np.ndarray]:
        """Создает базовые фолды для частых классов"""
        # Фильтруем данные только для частых классов
        frequent_mask = np.isin(y, frequent_classes)
        y_frequent = y[frequent_mask]
        frequent_indices = np.where(frequent_mask)[0]
        folds = [[] for _ in range(self.n_splits)]
        if len(frequent_classes) == 0:
            return [np.array([], dtype=int) for _ in range(self.n_splits)]

        for fold_idx, (_, test_idx) in enumerate(self.stratification_model.split(frequent_indices, y_frequent)):
            # Преобразуем индексы обратно к оригинальным
            original_indices = frequent_indices[test_idx]
            folds[fold_idx] = original_indices

        return folds

    def _distribute_rare_classes(self, X: pd.DataFrame, y: np.ndarray,
                                 base_folds: List[np.ndarray], rare_classes: List,
                                 min_samples_per_class: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Распределяет редкие классы по фолдам"""

        rare_indices_by_class = {class_label: np.where(y == class_label)[0] for class_label in rare_classes}
        rare_sample_idx = np.concatenate([class_indices for class_label, class_indices in rare_indices_by_class.items()])
        updated_fold = [np.append(base_folds[fold_idx], rare_sample_idx) for fold_idx in range(len(base_folds))]
        return updated_fold

    def _validate_folds(self, folds: List[Tuple[np.ndarray, np.ndarray]],
                        y: np.ndarray, expected_n_classes: int):
        """Валидирует что все фолды содержат все классы"""
        validation_passed = True

        for i, idx in enumerate(folds):
            train_classes = set(y[idx])

            if len(train_classes) != expected_n_classes:
                self.logger.warning(f"Фолд {i}: train содержит {len(train_classes)} из {expected_n_classes} классов")
                validation_passed = False

        if validation_passed:
            self.logger.info("Валидация пройдена: все классы присутствуют в каждом фолде")
        else:
            self.logger.error("Валидация не пройдена: некоторые классы отсутствуют в фолдах")

    def get_fold_statistics(self, folds: List[Tuple[np.ndarray, np.ndarray]], y: np.ndarray) -> pd.DataFrame:
        """Возвращает статистику по фолдам"""
        stats = []

        for i, (train_idx, test_idx) in enumerate(folds):
            train_classes = Counter(y[train_idx])
            test_classes = Counter(y[test_idx])

            stats.append({
                'fold': i,
                'train_size': len(train_idx),
                'test_size': len(test_idx),
                'train_n_classes': len(train_classes),
                'test_n_classes': len(test_classes),
                'train_min_class': min(train_classes.values()) if train_classes else 0,
                'test_min_class': min(test_classes.values()) if test_classes else 0,
                'train_max_class': max(train_classes.values()) if train_classes else 0,
                'test_max_class': max(test_classes.values()) if test_classes else 0
            })

        return pd.DataFrame(stats)