import numpy as np
import pandas as pd
import math

from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from typing import Dict, Any, Union
from .base_sampler import BaseSampler, HierarchicalStratifiedMixin
from ..repository.model_repo import SupportingModels
from ..utils.utils import to_dataframe, to_numpy


class ModelBasedSampler:
    def __init__(
        self,
        problem: str = None,
        model: Any = None,
        chunks_percent: int = 100,
    ):
        self.problem = problem
        self.model = model
        self.chunks_percent = chunks_percent

    def _resolve_problem(self, target: np.ndarray) -> str:
        if self.problem is None:
            return 'classification' if self._is_classification(target) else 'regression'
        return self.problem

    def _ensure_model(self, problem: str) -> None:
        if self.model is None:
            self._select_model(problem=problem)

    def _select_model(self, problem: str) -> None:
        model_params = dict(random_state=self.random_state, n_estimators=50)
        self.model = SupportingModels.difficulty_learner.value[problem](**model_params)

    @staticmethod
    def _encode_categorical(data: Union[np.ndarray, pd.DataFrame], encoding_type: str = "label"):
        df = to_dataframe(data).copy()

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


class DifficultyBasedSampler(BaseSampler, ModelBasedSampler, HierarchicalStratifiedMixin):
    """
    Семплирование на основе сложности примеров
    """

    def __init__(self, difficulty_threshold: float = None,
                 difficulty_metric: str = 'auto',
                 n_partitions: int = 2,
                 problem: str = None,
                 model: Any = None,
                 chunks_percent: int = 100,
                 random_state: int = 42):
        BaseSampler.__init__(self, random_state=random_state)
        ModelBasedSampler.__init__(
            self,
            problem=problem,
            model=model,
            chunks_percent=chunks_percent,
        )
        HierarchicalStratifiedMixin.__init__(
            self,
            n_partitions=n_partitions,
            random_state=random_state,
            logger_name="DifficultyBasedSampler",
        )
        self.difficulty_threshold = difficulty_threshold
        self.difficulty_metric = difficulty_metric
        self.difficulty_scores_ = None
        self.n_partitions = n_partitions

    def fit(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        target: Union[pd.Series, np.ndarray],
    ):
        """
        Args:
            data: Признаки
            target: Целевая переменная
        """
        target = to_numpy(target)
        problem = self._resolve_problem(target)
        self._ensure_model(problem)
        # кодируем категориальные признаки, чтобы избежать ошибки модели
        data = self._encode_categorical(data)

        # Вычисляем сложность примеров
        self.difficulty_scores_ = self._compute_difficulty_scores(data, target, problem)

        sorted_idx = np.argsort(self.difficulty_scores_)

        if self.chunks_percent < 100:
            chunks_to_keep = math.ceil(self.n_partitions * self.chunks_percent / 100)

            rows_per_chunk = len(sorted_idx) // self.n_partitions
            total_rows_needed = chunks_to_keep * rows_per_chunk
            k = max(1, len(sorted_idx) // total_rows_needed)

            # берём каждую k-ю строку из сортировки по сложности
            sorted_idx = sorted_idx[::k]
            sorted_idx = sorted_idx[-total_rows_needed:]

            self.n_partitions = chunks_to_keep

        rows_per_chunk = len(sorted_idx) // self.n_partitions

        classes = np.unique(target)
        class_indices = {
            c: sorted_idx[target[sorted_idx] == c] for c in classes
        }
        class_counts = {
            c: len(class_indices[c]) for c in classes
        }
        min_per_class = {c: max(1, int(0.2 * rows_per_chunk * class_counts[c] / len(sorted_idx))) for c in classes}

        used_indices = set()
        ptr = 0
        partitions = {}

        for i in range(self.n_partitions):
            chunk_idx = []
            while len(chunk_idx) < rows_per_chunk and ptr < len(sorted_idx):
                idx = sorted_idx[ptr]
                ptr += 1
                if idx not in used_indices:
                    chunk_idx.append(idx)
            if len(chunk_idx) < rows_per_chunk // 2:
                break

            chunk_idx = np.array(chunk_idx)

            chunk_classes = target[chunk_idx]
            counts = dict(zip(*np.unique(chunk_classes, return_counts=True)))

            additions = []

            for c in classes:
                if counts.get(c, 0) >= min_per_class[c]:
                    continue

                need = min_per_class[c] - counts.get(c, 0)
                pool = class_indices[c]

                pool = pool[~np.isin(pool, list(used_indices))]

                if len(pool) == 0:
                    continue

                if i < self.n_partitions // 2:
                    pool = pool[:need]
                else:
                    pool = pool[-need:]

                additions.extend(pool.tolist())

            chunk_idx = np.concatenate([chunk_idx, np.array(additions, dtype=np.int64)])
            used_indices.update(chunk_idx.tolist())

            partitions[f'chunk_{i}'] = chunk_idx

        partitions = self.rebalance_partitions(partitions, target, min_per_class)
        self.partitions = partitions
        return self

    def rebalance_partitions(self, partitions, target, min_per_class):
        partitions_classes_count = {}

        for name, idx in partitions.items():
            classes, counts = np.unique(target[idx], return_counts=True)
            partitions_classes_count[name] = dict(zip(classes.tolist(), counts.tolist()))

        chunk_names = list(partitions.keys())
        n_chunks = len(chunk_names)

        class_to_indices = {}
        for c in min_per_class:
            idx = np.where(target == c)[0]
            class_to_indices[c] = idx[np.argsort(self.difficulty_scores_[idx])]

        for c, min_req in min_per_class.items():
            surplus_pools = {}

            for i, name in enumerate(chunk_names):
                count = partitions_classes_count[name].get(c, 0)
                if count > min_req:
                    idx = partitions[name]
                    cls_idx = idx[target[idx] == c]
                    surplus = count - min_req
                    surplus_pools[i] = cls_idx[-surplus:]

            for i in range(n_chunks - 1, -1, -1):
                name = chunk_names[i]
                cur_count = partitions_classes_count[name].get(c, 0)
                need = max(0, min_req - cur_count)

                if need == 0:
                    continue

                for j in range(i - 1, -1, -1):
                    if need == 0:
                        break
                    if j not in surplus_pools or need == 0:
                        continue

                    pool = surplus_pools[j]
                    take = min(len(pool), need)

                    moved = pool[-take:]
                    surplus_pools[j] = pool[:-take]
                    need -= take

                    partitions[name] = np.concatenate([partitions[name], moved])
                    partitions[f'chunk_{j}'] = np.setdiff1d(partitions[f'chunk_{j}'], moved, assume_unique=True)
        return partitions

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

    def get_partitions(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        target: Union[np.ndarray, pd.Series],
    ) -> Dict[Any, np.ndarray]:
        return self._get_partitions_default(data=data, target=target)

class UncertaintySampler(BaseSampler, ModelBasedSampler, HierarchicalStratifiedMixin):
    """
    Семплирование на основе неопределенности модели
    """

    def __init__(
        self,
        uncertainty_threshold: float = None,
        n_partitions: int = 2,
        problem: str = None,
        model: Any = None,
        chunks_percent: int = 100,
        random_state: int = 42,
    ):
        BaseSampler.__init__(self, random_state=random_state)
        ModelBasedSampler.__init__(
            self,
            problem=problem,
            model=model,
            chunks_percent=chunks_percent
        )
        HierarchicalStratifiedMixin.__init__(
            self,
            n_partitions=n_partitions,
            random_state=random_state,
            logger_name="UncertaintySampler",
        )
        self.uncertainty_threshold = uncertainty_threshold
        self.uncertainty_scores_ = None
        self.n_partitions = n_partitions

    def fit(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        target: Union[pd.Series, np.ndarray],
    ) -> 'UncertaintySampler':
        target = to_numpy(target)
        problem = self._resolve_problem(target)
        self._ensure_model(problem)
        data = self._encode_categorical(data)

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
    
    def get_partitions(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        target: Union[np.ndarray, pd.Series],
    ) -> Dict[Any, np.ndarray]:
        return self._get_partitions_default(data=data, target=target)
    
    def get_uncertainty_scores(self) -> np.ndarray:
        return self.uncertainty_scores_
