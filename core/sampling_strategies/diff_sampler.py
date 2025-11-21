import numpy as np
import pandas as pd

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score, mean_squared_error
from typing import Dict, Any, Union, Callable
from .base_sampler import BaseSampler, HierarchicalStratifiedMixin
from ..repository.model_repo import SupportingModels


class DifficultyBasedSampler(BaseSampler, HierarchicalStratifiedMixin):
    """
    Семплирование на основе сложности примеров
    """

    def __init__(self, difficulty_threshold: float = None,
                 difficulty_metric: str = 'auto',
                 n_partitions: int = 2, model: Any = None,
                 random_state: int = 42,
                 **kwargs):
        BaseSampler.__init__(self, random_state=random_state)
        HierarchicalStratifiedMixin.__init__(
            self,
            n_splits=n_partitions,
            random_state=random_state,
            logger_name="DifficultyBasedSampler",
        )
        self.difficulty_threshold = difficulty_threshold
        self.difficulty_metric = difficulty_metric
        self.model = model
        self.difficulty_scores_ = None
        self.n_partitions = n_partitions

    def fit(self, data: Union[np.ndarray, pd.DataFrame],
            target: np.ndarray, **kwargs) -> 'DifficultyBasedSampler':
        """
        Args:
            data: Признаки
            target: Целевая переменная
        """
        # Выбор базовой модели
        if self.model is None:
            model_params = dict(random_state=self.random_state, n_estimators=50)
            if self._is_classification(target):
                self.model = SupportingModels.difficulty_learner.value['classification'](**model_params)
            else:
                self.model = SupportingModels.difficulty_learner.value['regression'](**model_params)

        # Получаем out-of-fold предсказания чтобы избежать переобучения
        predictions = cross_val_predict(self.model, data, target, cv=5)

        # Вычисляем сложность примеров
        self.difficulty_scores_ = self._compute_difficulty_scores(data, target, predictions)

        # Создаем разделы на основе сложности
        # Если классов 2 и получен difficulty_threshold, делим на 2 части по нему, иначе - на равные части
        if self.n_partitions == 2 and self.difficulty_threshold is not None:
            hard_indices = np.where(self.difficulty_scores_ > self.difficulty_threshold)[0]
            easy_indices = np.where(self.difficulty_scores_ <= self.difficulty_threshold)[0]

            self.partitions = {'hard': hard_indices, 'easy': easy_indices}
        else: 
            #сортируем по возрастанию сложности и разбиваем на n_partitions равных частей
            split = np.array_split(np.argsort(self.difficulty_scores_), self.n_partitions)
            self.partitions = {f'chunk_{i}': indices for i, indices in enumerate(split)}

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

    def __init__(self, uncertainty_threshold: float = None, n_partitions: int = 2, random_state: int = 42, **kwargs):
        super().__init__(n_partitions=n_partitions, random_state=random_state, **kwargs)
        self.uncertainty_threshold = uncertainty_threshold
        self.uncertainty_scores_ = None
        self.model = None
        self.n_partitions = n_partitions

    def fit(self, data: Union[np.ndarray, pd.DataFrame],
            target: np.ndarray, **kwargs) -> 'UncertaintySampler':
        
        # Выбор базовой модели
        if self.model is None:
            model_params = dict(random_state=self.random_state, n_estimators=50)
            if self._is_classification(target):
                self.model = SupportingModels.difficulty_learner.value['classification'](**model_params)

        # Получаем вероятности через кросс-валидацию
        proba_predictions = cross_val_predict(self.model, data, target, cv=5, method='predict_proba')

        # Вычисляем неопределенность как энтропию распределения
        epsilon = 1e-8
        entropy = -np.sum(proba_predictions * np.log(proba_predictions + epsilon), axis=1)
        self.uncertainty_scores_ = entropy / np.log(proba_predictions.shape[1])  # Нормализуем

        # Если классов 2 и получен difficulty_threshold, делим на 2 части по нему, иначе - на равные части
        if self.n_partitions == 2 and self.uncertainty_threshold is not None:
            high_uncertainty_indices = np.where(self.uncertainty_scores_ > self.uncertainty_threshold)[0]
            low_uncertainty_indices = np.where(self.uncertainty_scores_ <= self.uncertainty_threshold)[0]
            self.partitions = {
                'high_uncertainty': high_uncertainty_indices,
                'low_uncertainty': low_uncertainty_indices
            }
        else:
            #сортируем по возрастанию неопределенности и разбиваем на n_partitions равных частей
            split = np.array_split(np.argsort(self.uncertainty_scores_), self.n_partitions)
            self.partitions = {f'uncertainty_level_{i}': indices for i, indices in enumerate(split)}
            
        return self
    
    def get_partitions(self, data, target) -> Dict[Any, np.ndarray]:
        partition = {cluster: dict(feature=data.iloc[idx],
                                   target=target[idx]) for cluster, idx in self.partitions.items()}
        return partition
    
    def get_uncertainty_scores(self) -> np.ndarray:
        return self.uncertainty_scores_
