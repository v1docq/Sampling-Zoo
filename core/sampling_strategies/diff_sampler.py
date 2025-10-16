import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score, mean_squared_error
from typing import Dict, Any, Union, Callable
from .base_sampler import BaseSampler


class DifficultyBasedSampler(BaseSampler):
    """
    Семплирование на основе сложности примеров
    """

    def __init__(self, difficulty_threshold: float = 0.8,
                 base_model: str = 'random_forest',
                 difficulty_metric: str = 'auto', **kwargs):
        super().__init__(**kwargs)
        self.difficulty_threshold = difficulty_threshold
        self.base_model = base_model
        self.difficulty_metric = difficulty_metric
        self.model = None
        self.difficulty_scores_ = None

    def fit(self, data: Union[np.ndarray, pd.DataFrame],
            target: np.ndarray, **kwargs) -> 'DifficultyBasedSampler':
        """
        Args:
            data: Признаки
            target: Целевая переменная
        """
        # Выбор базовой модели
        if self.base_model == 'random_forest':
            if self._is_classification(target):
                self.model = RandomForestClassifier(random_state=self.random_state, n_estimators=50)
            else:
                self.model = RandomForestRegressor(random_state=self.random_state, n_estimators=50)

        # Получаем out-of-fold предсказания чтобы избежать переобучения
        predictions = cross_val_predict(self.model, data, target, cv=5)

        # Вычисляем сложность примеров
        self.difficulty_scores_ = self._compute_difficulty_scores(data, target, predictions)

        # Создаем разделы на основе сложности
        hard_indices = np.where(self.difficulty_scores_ > self.difficulty_threshold)[0]
        easy_indices = np.where(self.difficulty_scores_ <= self.difficulty_threshold)[0]

        self.partitions = {'hard': hard_indices, 'easy': easy_indices}

        return self

    def _is_classification(self, target: np.ndarray) -> bool:
        """Определяет тип задачи (классификация или регрессия)"""
        return len(np.unique(target)) < 0.1 * len(target) or target.dtype == 'object'

    def _compute_difficulty_scores(self, data, target: np.ndarray, predictions: np.ndarray) -> np.ndarray:
        """Вычисляет оценку сложности для каждого примера"""
        if self._is_classification(target):
            # Для классификации: 1 - вероятность правильного класса
            if hasattr(self.model, 'predict_proba'):
                # Используем кросс-валидационные вероятности если доступны
                proba_predictions = cross_val_predict(self.model, data, target, cv=5, method='predict_proba')
                true_class_probs = proba_predictions[np.arange(len(target)), target]
                return 1 - true_class_probs
            else:
                # Используем accuracy-based метрику
                correct_predictions = (predictions == target)
                return 1 - correct_predictions.astype(float)
        else:
            # Для регрессии: нормализованная абсолютная ошибка
            errors = np.abs(predictions - target)
            return errors / (np.max(errors) + 1e-8)

    def get_difficulty_scores(self) -> np.ndarray:
        """Возвращает вычисленные оценки сложности"""
        return self.difficulty_scores_

    def get_partitions(self, data, target) -> Dict[Any, np.ndarray]:
        partition = {cluster: dict(feature=data.iloc[idx],
                                   target=target[idx]) for cluster, idx in self.partitions.items()}
        return partition


class UncertaintySampler(DifficultyBasedSampler):
    """
    Семплирование на основе неопределенности модели
    """

    def __init__(self, uncertainty_threshold: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.uncertainty_threshold = uncertainty_threshold
        self.uncertainty_scores_ = None

    def fit(self, data: Union[np.ndarray, pd.DataFrame],
            target: np.ndarray, **kwargs) -> 'UncertaintySampler':
        model = RandomForestClassifier(random_state=self.random_state, n_estimators=50)

        # Получаем вероятности через кросс-валидацию
        proba_predictions = cross_val_predict(model, data, target, cv=5, method='predict_proba')

        # Вычисляем неопределенность как энтропию распределения
        epsilon = 1e-8
        entropy = -np.sum(proba_predictions * np.log(proba_predictions + epsilon), axis=1)
        self.uncertainty_scores_ = entropy / np.log(proba_predictions.shape[1])  # Нормализуем

        # Создаем разделы
        high_uncertainty = np.where(self.uncertainty_scores_ > self.uncertainty_threshold)[0]
        low_uncertainty = np.where(self.uncertainty_scores_ <= self.uncertainty_threshold)[0]

        self.partitions = {
            'high_uncertainty': high_uncertainty,
            'low_uncertainty': low_uncertainty
        }

        return self

    def get_uncertainty_scores(self) -> np.ndarray:
        return self.uncertainty_scores_
