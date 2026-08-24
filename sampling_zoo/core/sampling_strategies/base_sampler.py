from abc import ABC, abstractmethod
import logging
from collections import Counter
from typing import Any, Dict, List, Union, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..utils.utils import safe_index

torch = None
_TORCH_IMPORT_ATTEMPTED = False


def _load_torch_backend() -> Optional[Any]:
    """Load torch lazily so base sampler imports stay lightweight."""

    global torch, _TORCH_IMPORT_ATTEMPTED
    if torch is not None:
        return torch
    if _TORCH_IMPORT_ATTEMPTED:
        return None
    _TORCH_IMPORT_ATTEMPTED = True
    try:
        import torch as torch_module
    except Exception:  # pragma: no cover - torch is optional
        torch = None
    else:
        torch = torch_module
    return torch


class BaseSampler(ABC):
    """
    Абстрактный базовый класс для всех стратегий семплирования
    """

    def __init__(self, random_state: int = 42, **kwargs):
        self.random_state = random_state
        self.partitions = None
        self.preprocessor_: Optional[ColumnTransformer] = None
        self.encoded_feature_subset_: Optional[np.ndarray] = None
        self.raw_encoded_feature_count_: Optional[int] = None
        self.encoded_feature_count_: Optional[int] = None

    @abstractmethod
    def fit(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        target: Optional[Union[np.ndarray, pd.Series]] = None,
        **kwargs,
    ) -> 'BaseSampler':
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
    def get_partitions(
        self,
        data: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        target: Optional[Union[np.ndarray, pd.Series]] = None,
    ) -> Dict[Any, Any]:
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
        partitions = self.get_partitions(data=data)
        if not partitions:
            return {}

        sample_value = next(iter(partitions.values()))
        if isinstance(sample_value, dict) and 'feature' in sample_value:
            return {name: chunk['feature'] for name, chunk in partitions.items()}

        return partitions

    def fit_transform(self, data: Union[np.ndarray, pd.DataFrame], **kwargs) -> Dict[
        Any, Union[np.ndarray, pd.DataFrame]]:
        """
        Обучение и преобразование за один шаг
        """
        self.fit(data, **kwargs)
        return self.transform(data)

    def check_partitions(self, partitions, data):
        pass

    @staticmethod
    def _partitions_contain_data(partitions: Dict[Any, Any]) -> bool:
        if not isinstance(partitions, dict) or not partitions:
            return False
        sample = next(iter(partitions.values()))
        return isinstance(sample, dict) and ('feature' in sample or 'target' in sample)

    def _build_feature_target_partitions(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        target: Optional[Union[np.ndarray, pd.Series]] = None,
    ) -> Dict[Any, Dict[str, Any]]:
        result = {}
        for partition_name, indices in self.partitions.items():
            chunk = {'feature': safe_index(data, indices)}
            if target is not None:
                chunk['target'] = safe_index(target, indices)
            result[partition_name] = chunk
        return result

    def _get_partitions_default(
        self,
        data: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        target: Optional[Union[np.ndarray, pd.Series]] = None,
    ) -> Dict[Any, Any]:
        if self.partitions is None:
            raise ValueError("Sampler not fitted. Call fit() first.")

        if self._partitions_contain_data(self.partitions):
            return self.partitions

        if data is None:
            return self.partitions

        return self._build_feature_target_partitions(data, target)

    @staticmethod
    def _validate_positive_int(name: str, value: int) -> int:
        if int(value) < 1:
            raise ValueError(f"{name} must be positive")
        return int(value)

    @staticmethod
    def _validate_fraction(name: str, value: float) -> float:
        value = float(value)
        if not (0 < value <= 1):
            raise ValueError(f"{name} must be in (0, 1]")
        return value

    @staticmethod
    def _validate_percent(name: str, value: float) -> float:
        value = float(value)
        if not (0 < value <= 100):
            raise ValueError(f"{name} must be in (0, 100]")
        return value

    @staticmethod
    def _validate_choice(name: str, value: str, choices: Sequence[str]) -> str:
        if value not in choices:
            allowed = ", ".join(choices)
            raise ValueError(f"{name} must be one of: {allowed}")
        return value

    def _configure_tabular_preprocessing(
        self,
        *,
        include_categorical: bool = True,
        max_one_hot_cardinality: int = 128,
        max_encoded_features: Optional[int] = None,
    ) -> None:
        self.include_categorical = bool(include_categorical)
        self.max_one_hot_cardinality = int(max_one_hot_cardinality)
        self.max_encoded_features = max_encoded_features

    def _fit_transform_features(self, data: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if isinstance(data, pd.DataFrame):
            X = self._fit_transform_dataframe(data)
        else:
            X = self._fit_transform_array(data)
        return self._cap_and_densify_encoded_features(X)

    def _fit_transform_dataframe(self, data: pd.DataFrame) -> Any:
        df = data.copy()
        numeric_cols = df.select_dtypes(include=[np.number, "bool"]).columns.tolist()
        categorical_cols = self._select_categorical_columns(df, numeric_cols)
        transformers = self._build_tabular_transformers(numeric_cols, categorical_cols)
        if not transformers:
            raise ValueError(f"No usable numeric/categorical columns for {type(self).__name__}")
        self.preprocessor_ = ColumnTransformer(transformers, remainder="drop", sparse_threshold=1.0)
        return self.preprocessor_.fit_transform(df)

    def _fit_transform_array(self, data: Union[np.ndarray, pd.DataFrame]) -> Any:
        arr = np.asarray(data)
        if arr.ndim != 2:
            raise ValueError("Input data must be a 2D matrix")
        self.preprocessor_ = ColumnTransformer([
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), list(range(arr.shape[1])))
        ])
        return self.preprocessor_.fit_transform(arr)

    def _select_categorical_columns(self, df: pd.DataFrame, numeric_cols: Sequence[str]) -> List[str]:
        categorical_cols = [c for c in df.columns if c not in numeric_cols]
        if not getattr(self, "include_categorical", True):
            return []
        return [
            c for c in categorical_cols
            if df[c].astype("string").nunique(dropna=True) <= getattr(self, "max_one_hot_cardinality", 128)
        ]

    @staticmethod
    def _build_tabular_transformers(
        numeric_cols: Sequence[str],
        categorical_cols: Sequence[str],
    ) -> List[Tuple[str, Pipeline, Sequence[str]]]:
        transformers: List[Tuple[str, Pipeline, Sequence[str]]] = []
        if numeric_cols:
            transformers.append((
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                numeric_cols,
            ))
        if categorical_cols:
            try:
                encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
            except TypeError:  # sklearn < 1.2
                encoder = OneHotEncoder(handle_unknown="ignore", sparse=True)
            transformers.append((
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", encoder),
                ]),
                categorical_cols,
            ))
        return transformers

    def _cap_and_densify_encoded_features(self, X: Any) -> np.ndarray:
        self.raw_encoded_feature_count_ = int(X.shape[1])
        if self.max_encoded_features is not None and X.shape[1] > self.max_encoded_features:
            rng = np.random.default_rng(self.random_state)
            keep = np.sort(rng.choice(X.shape[1], size=self.max_encoded_features, replace=False))
            self.encoded_feature_subset_ = keep
            X = X[:, keep]
        else:
            self.encoded_feature_subset_ = None
        self.encoded_feature_count_ = int(X.shape[1])
        return self._as_dense_float(X)

    def _transform_features(self, data: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if self.preprocessor_ is None:
            raise RuntimeError("Preprocessor is not fitted")
        X = self.preprocessor_.transform(data.copy() if isinstance(data, pd.DataFrame) else data)
        keep = getattr(self, "encoded_feature_subset_", None)
        if keep is not None:
            X = X[:, keep]
        return self._as_dense_float(X)

    @staticmethod
    def _as_dense_float(X: Any) -> np.ndarray:
        if sparse.issparse(X):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float64)
        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    def _resolve_backend(self) -> str:
        torch_module = _load_torch_backend()
        if self.backend == "numpy":
            return "numpy"
        if self.backend == "torch":
            if torch_module is None:
                raise ImportError("backend='torch' requires torch to be installed")
            self._resolve_torch_device()
            return "torch"
        if torch_module is None:
            return "numpy"
        self._resolve_torch_device()
        return "torch"

    def _resolve_torch_device(self) -> Any:
        torch_module = _load_torch_backend()
        if torch_module is None:
            return None
        device = torch_module.device(self.device)
        if device.type == "cuda" and not torch_module.cuda.is_available():
            if self.backend == "torch":
                raise ValueError(f"Requested torch device is not available: {self.device}")
            device = torch_module.device("cpu")
        return device

    def _torch_dtype(self) -> Any:
        torch_module = _load_torch_backend()
        if torch_module is None:
            return None
        return torch_module.float32 if self.dtype == "float32" else torch_module.float64

    def _to_torch_matrix(self, X: np.ndarray) -> Any:
        torch_module = _load_torch_backend()
        if torch_module is None:
            raise RuntimeError("Torch backend selected but torch is unavailable")
        return torch_module.as_tensor(X, dtype=self._torch_dtype(), device=self._resolve_torch_device())


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
