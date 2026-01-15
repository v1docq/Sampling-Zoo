import numpy as np
import pandas as pd
import math

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from typing import Dict, Any, Union, Callable
from .base_sampler import BaseSampler, HierarchicalStratifiedMixin
from ..repository.model_repo import SupportingModels
from ..utils.utils import safe_index


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
            n_partitions=n_partitions,
            random_state=random_state,
            logger_name="DifficultyBasedSampler",
        )
        self.difficulty_threshold = difficulty_threshold
        self.difficulty_metric = difficulty_metric
        self.model = model
        self.difficulty_scores_ = None
        self.n_partitions = n_partitions

    def fit(self, data: Union[np.ndarray, pd.DataFrame], target: np.ndarray,
            problem: str = None, model=None, chunks_percent: int = 100, **kwargs) -> 'DifficultyBasedSampler':
        """
        Args:
            data: Признаки
            target: Целевая переменная
        """
        if problem is None:
            problem = 'classification' if self._is_classification(target) else 'regression'
        # Выбор базовой модели
        if model is not None:
            self.model = model
        else:
            self._select_model(problem=problem)
        # кодируем категориальные признаки, чтобы избежать ошибки модели
        data = self._encode_categorical(data)

        # Вычисляем сложность примеров
        self.difficulty_scores_ = self._compute_difficulty_scores(data, target, problem)

        # Создаем разделы на основе сложности
        # Если классов 2 и получен difficulty_threshold, делим на 2 части по нему, иначе - на равные части
        if self.n_partitions == 2 and self.difficulty_threshold is not None:
            hard_indices = np.where(self.difficulty_scores_ > self.difficulty_threshold)[0]
            easy_indices = np.where(self.difficulty_scores_ <= self.difficulty_threshold)[0]

            self.partitions = {'hard': hard_indices, 'easy': easy_indices}
        else:
            sorted_idx = np.argsort(self.difficulty_scores_)

            if chunks_percent < 100:
                chunks_to_keep = math.ceil(self.n_partitions * chunks_percent / 100)

                rows_per_chunk = len(sorted_idx) // self.n_partitions
                total_rows_needed = chunks_to_keep * rows_per_chunk

                k = max(1, len(sorted_idx) // total_rows_needed)

                # берём каждую k-ю строку из сортировки по сложности
                sorted_idx = sorted_idx[::k]
                sorted_idx = sorted_idx[:total_rows_needed]

                self.n_partitions = chunks_to_keep

            split = np.array_split(sorted_idx, self.n_partitions)
            self.partitions = {f'chunk_{i}': indices for i, indices in enumerate(split)}

        return self

    def _select_model(self, problem: str):
        if self.model is None:
            model_params = dict(random_state=self.random_state, n_estimators=50)
            self.model = SupportingModels.difficulty_learner.value[problem](**model_params)

    @staticmethod
    def _encode_categorical(data: Union[np.ndarray, pd.DataFrame], encoding_type: str = "label"):
        if isinstance(data, np.ndarray):
            df = pd.DataFrame(data)
        else:
            df = data.copy()

        categorical_columns = df.select_dtypes(include=["object", "category"]).columns.tolist()

        if encoding_type == "label":
            for col in categorical_columns:
                df[col] = LabelEncoder().fit_transform(df[col].astype(str))

            return df

        elif encoding_type == "one-hot":
            ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
            encoded = ohe.fit_transform(df[categorical_columns])

            encoded_df = pd.DataFrame(
                encoded,
                columns=ohe.get_feature_names_out(categorical_columns),
                index=df.index
            )

            numeric_df = df.drop(columns=categorical_columns)
            final_df = pd.concat([numeric_df, encoded_df], axis=1)

            return final_df

        else:
            raise NotImplementedError("encoding_type must be 'label' or 'one-hot'")

    @staticmethod
    def _is_classification(target: np.ndarray) -> bool:
        """Определяет тип задачи (классификация или регрессия)"""
        return len(np.unique(target)) < 0.1 * len(target) or target.dtype == 'object'

    def _compute_difficulty_scores(self, data, target: np.ndarray, problem: str) -> np.ndarray:
        """Вычисляет оценку сложности для каждого примера"""
        if problem == 'classification':
            # Для классификации: 1 - вероятность правильного класса
            if hasattr(self.model, 'predict_proba'):
                # Используем кросс-валидационные вероятности если доступны
                proba_predictions = cross_val_predict(self.model, data, target, cv=5, method='predict_proba')
                true_class_probs = proba_predictions[np.arange(len(target)), target]
                return 1 - true_class_probs
            else:
                predictions = cross_val_predict(self.model, data, target, cv=5)
                # Используем accuracy-based метрику
                correct_predictions = (predictions == target)
                return 1 - correct_predictions.astype(float)
        elif problem == 'regression':
            predictions = cross_val_predict(self.model, data, target, cv=5)
            # Для регрессии: нормализованная абсолютная ошибка
            errors = np.abs(predictions - target)
            return errors / (np.max(errors) + 1e-8)
        else:
            raise NotImplementedError("problem must be 'classification' or 'regression'")

    def get_difficulty_scores(self) -> np.ndarray:
        """Возвращает вычисленные оценки сложности"""
        return self.difficulty_scores_

    def get_partitions(self, data, target) -> Dict[Any, np.ndarray]:
        partition = {
            cluster: dict(feature=safe_index(data, idx),
                          target=safe_index(target, idx))
            for cluster, idx in self.partitions.items()
        }
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
            target: np.ndarray, problem: str = None, model=None, **kwargs) -> 'UncertaintySampler':
        if problem is None:
            problem = 'classification' if self._is_classification(target) else 'regression'
        # Выбор базовой модели
        if model is not None:
            self.model = model
        else:
            self._select_model(problem=problem)

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
