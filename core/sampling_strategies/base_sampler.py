from abc import ABC, abstractmethod
import logging
from collections import Counter
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


class BaseSampler(ABC):
    """
    Абстрактный базовый класс для всех стратегий семплирования
    """

    def __init__(self, random_state: int = 42, **kwargs):
        self.random_state = random_state
        self.partitions_ = None

    @abstractmethod
    def fit(self, data: Union[np.ndarray, pd.DataFrame], **kwargs) -> 'BaseSampler':
        """
        Обучение семплера на данных

        Args:
            data: Входные данные
            **kwargs: Дополнительные параметры

        Returns:
            self: Обученный семплер
        """
        pass

    @abstractmethod
    def get_partitions(self) -> Dict[Any, np.ndarray]:
        """
        Возвращает индексы разделов

        Returns:
            Dict с индексами для каждого раздела
        """
        pass

    def transform(self, data: Union[np.ndarray, pd.DataFrame]) -> Dict[Any, Union[np.ndarray, pd.DataFrame]]:
        """
        Преобразует данные в разделы

        Args:
            data: Исходные данные

        Returns:
            Dict с разделенными данными
        """
        partitions_indices = self.get_partitions()
        result = {}

        for partition_name, indices in partitions_indices.items():
            if isinstance(data, pd.DataFrame):
                result[partition_name] = data.iloc[indices]
            else:
                result[partition_name] = data[indices]

        return result

    def fit_transform(self, data: Union[np.ndarray, pd.DataFrame], **kwargs) -> Dict[
        Any, Union[np.ndarray, pd.DataFrame]]:
        """
        Обучение и преобразование за один шаг
        """
        self.fit(data, **kwargs)
        return self.transform(data)

    def check_partitions(self, partitions, data):
        pass


class HierarchicalStratifiedMixin:
    """Mixin с реализацией многоуровневого стратифицированного разбиения."""

    def __init__(self, n_partitions: int = 5, random_state: int = 42, logger_name: str = "StratifiedSampler"):
        self.n_partitions = n_partitions
        self.random_state = random_state
        self.logger = self._setup_logger(logger_name)
        self.stratification_model = StratifiedShuffleSplit(
            n_splits=self.n_partitions, test_size=1 / self.n_partitions, random_state=self.random_state
        )

    @staticmethod
    def print_fold_summary(name: str, folds: Dict[str, np.ndarray], targets: Union[pd.Series, np.ndarray]) -> None:
        """Кратко печатает размеры и распределения классов по фолдам."""

        series = pd.Series(targets)
        print(f"\n{name}")
        for fold_name, indices in folds.items():
            fold_classes = Counter(series.iloc[indices])
            print(f"{fold_name}: size={len(indices)}, classes={dict(fold_classes)}")

    def _setup_logger(self, name: str):
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def analyze_class_distribution(self, y: np.ndarray) -> Dict:
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

        for class_label, count in class_counts.items():
            if count < self.n_partitions:
                analysis['problematic_classes'].append((class_label, count))

        self.logger.info(f"Анализ распределения: {analysis['n_classes']} классов")
        self.logger.info(f"Минимальное число семплов в классе: {analysis['min_class_count']} samples")
        #self.logger.info(f"Проблемные классы: {analysis['problematic_classes']}")

        return analysis

    def hierarchical_stratified_split(self, X: pd.DataFrame, y: np.ndarray, min_samples_per_class: int = 2):
        analysis = self.analyze_class_distribution(y)
        frequent_classes, rare_classes = self._separate_classes_by_frequency(y, analysis, min_samples_per_class)
        self.logger.info(f"Распределение семплов по классам: (имя класса: число семплов)")
        self.logger.info(f"Частые классы: {frequent_classes}")
        self.logger.info(f"Редкие классы: {rare_classes}")

        base_folds = self._create_base_folds(y, frequent_classes)
        final_folds = self._distribute_rare_classes(base_folds, rare_classes, y)
        self._validate_folds(final_folds, y, analysis['n_classes'])

        return final_folds

    def _separate_classes_by_frequency(self, y: np.ndarray, analysis: Dict, min_samples: int):
        frequent_classes, rare_classes = [], []
        for class_label, count in analysis['class_distribution'].items():
            if count < self.n_partitions * min_samples:
                rare_classes.append(class_label)
            else:
                frequent_classes.append(class_label)
        return frequent_classes, rare_classes

    def _create_base_folds(self, y: np.ndarray, frequent_classes: List):
        frequent_mask = np.isin(y, frequent_classes)
        y_frequent = y[frequent_mask]
        frequent_indices = np.where(frequent_mask)[0]

        if len(frequent_classes) == 0:
            return [np.array([], dtype=int) for _ in range(self.n_partitions)]

        folds = [[] for _ in range(self.n_partitions)]
        for fold_idx, (_, test_idx) in enumerate(self.stratification_model.split(frequent_indices, y_frequent)):
            folds[fold_idx] = frequent_indices[test_idx]
        return folds

    def _distribute_rare_classes(self, base_folds: List[np.ndarray], rare_classes: List, y: np.ndarray):
        rare_indices_by_class = {class_label: np.where(y == class_label)[0] for class_label in rare_classes}
        rare_sample_idx = np.concatenate([class_indices for class_indices in rare_indices_by_class.values()])\
            if rare_indices_by_class else np.array([], dtype=int)
        return [np.append(base_folds[fold_idx], rare_sample_idx) for fold_idx in range(len(base_folds))]

    def _validate_folds(self, folds, y: np.ndarray, expected_n_classes: int):
        validation_passed = True
        for i, idx in enumerate(folds):
            train_classes = set(y[idx])
            if len(train_classes) != expected_n_classes:
                self.logger.warning(
                    f"Фолд {i}: train содержит {len(train_classes)} из {expected_n_classes} классов"
                )
                validation_passed = False

        if validation_passed:
            self.logger.info("Валидация пройдена: все классы присутствуют в каждом фолде")
        else:
            self.logger.error("Валидация не пройдена: некоторые классы отсутствуют в фолдах")