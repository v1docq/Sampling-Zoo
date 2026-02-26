from __future__ import annotations

from dataclasses import dataclass
from inspect import signature
from typing import Callable

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class DatasetMetadata:
    n_objects: int
    n_features: int
    n_train: int
    n_test: int
    n_categorical: int
    n_numeric: int
    categorical_cardinality: dict[str, int]


@dataclass(frozen=True)
class DatasetBundle:
    name: str
    seed: int
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    X_train_processed: sparse.spmatrix | np.ndarray
    X_test_processed: sparse.spmatrix | np.ndarray
    preprocessor: ColumnTransformer
    metadata: DatasetMetadata


def _make_one_hot_encoder() -> OneHotEncoder:
    if 'sparse_output' in signature(OneHotEncoder).parameters:
        return OneHotEncoder(handle_unknown='ignore', sparse_output=True)
    return OneHotEncoder(handle_unknown='ignore', sparse=True)


def _make_preprocessor(
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ('numeric', Pipeline([('scale', StandardScaler())]), numeric_columns),
            (
                'categorical',
                Pipeline([
                    (
                        'encode',
                        _make_one_hot_encoder(),
                    )
                ]),
                categorical_columns,
            ),
        ]
    )


def _build_bundle(
    name: str,
    seed: int,
    X: pd.DataFrame,
    y: pd.Series,
    numeric_columns: list[str],
    categorical_columns: list[str],
    test_size: float,
) -> DatasetBundle:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    preprocessor = _make_preprocessor(
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    metadata = DatasetMetadata(
        n_objects=len(X),
        n_features=X.shape[1],
        n_train=len(X_train),
        n_test=len(X_test),
        n_categorical=len(categorical_columns),
        n_numeric=len(numeric_columns),
        categorical_cardinality={
            column: int(X[column].nunique()) for column in categorical_columns
        },
    )

    return DatasetBundle(
        name=name,
        seed=seed,
        X_train=X_train.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
        X_train_processed=X_train_processed,
        X_test_processed=X_test_processed,
        preprocessor=preprocessor,
        metadata=metadata,
    )


def _load_high_cardinality_categorical(seed: int) -> DatasetBundle:
    rng = np.random.default_rng(seed)
    n_samples = 12000

    numeric_columns = [f'num_{idx}' for idx in range(6)]
    categorical_columns = [f'cat_{idx}' for idx in range(8)]

    X_numeric = rng.normal(0.0, 1.0, size=(n_samples, len(numeric_columns)))

    cardinalities = [40, 80, 120, 160, 220, 300, 400, 500]
    categorical_data: dict[str, pd.Series] = {}
    category_signal = np.zeros(n_samples)

    for column, cardinality in zip(categorical_columns, cardinalities):
        values = rng.integers(0, cardinality, size=n_samples)
        categorical_data[column] = pd.Series(
            [f'{column}_value_{val}' for val in values],
            dtype='string',
        )
        category_signal += (values % 7) / 6.0

    X = pd.DataFrame(X_numeric, columns=numeric_columns)
    for column in categorical_columns:
        X[column] = categorical_data[column]

    linear_signal = (
        0.9 * X['num_0']
        - 0.7 * X['num_1']
        + 0.6 * X['num_2']
        + 0.2 * category_signal
        + rng.normal(0.0, 0.35, n_samples)
    )
    y = pd.Series((linear_signal > np.quantile(linear_signal, 0.52)).astype(int), name='target')

    return _build_bundle(
        name='high_cardinality_categorical',
        seed=seed,
        X=X,
        y=y,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        test_size=0.2,
    )


def _load_large_numeric(seed: int) -> DatasetBundle:
    X_values, y_values = make_classification(
        n_samples=18000,
        n_features=120,
        n_informative=45,
        n_redundant=25,
        n_repeated=0,
        n_classes=2,
        n_clusters_per_class=2,
        flip_y=0.03,
        class_sep=1.1,
        random_state=seed,
    )

    numeric_columns = [f'num_{idx}' for idx in range(X_values.shape[1])]
    X = pd.DataFrame(X_values, columns=numeric_columns)
    y = pd.Series(y_values, name='target')

    return _build_bundle(
        name='large_numeric',
        seed=seed,
        X=X,
        y=y,
        numeric_columns=numeric_columns,
        categorical_columns=[],
        test_size=0.25,
    )


def _load_mixed_hard(seed: int) -> DatasetBundle:
    rng = np.random.default_rng(seed)

    X_numeric, y_values = make_classification(
        n_samples=18000,
        n_features=18,
        n_informative=10,
        n_redundant=4,
        n_repeated=0,
        n_classes=2,
        weights=[0.9, 0.1],
        class_sep=0.65,
        flip_y=0.09,
        random_state=seed,
    )

    numeric_columns = [f'num_{idx}' for idx in range(X_numeric.shape[1])]
    categorical_columns = ['cat_small', 'cat_medium', 'cat_rare', 'cat_noise']

    X = pd.DataFrame(X_numeric, columns=numeric_columns)
    y = pd.Series(y_values, name='target')

    X['cat_small'] = pd.Series(
        [f's_{value}' for value in rng.integers(0, 5, size=len(X))],
        dtype='string',
    )
    X['cat_medium'] = pd.Series(
        [f'm_{value}' for value in rng.integers(0, 30, size=len(X))],
        dtype='string',
    )

    rare_probability = np.where(y.values == 1, 0.22, 0.05)
    X['cat_rare'] = pd.Series(
        np.where(
            rng.uniform(size=len(X)) < rare_probability,
            'rare',
            'common',
        ),
        dtype='string',
    )

    X['cat_noise'] = pd.Series(
        [f'n_{value}' for value in rng.integers(0, 200, size=len(X))],
        dtype='string',
    )

    return _build_bundle(
        name='mixed_hard',
        seed=seed,
        X=X,
        y=y,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        test_size=0.2,
    )


def load_dataset(name: str, seed: int) -> DatasetBundle:
    loaders: dict[str, Callable[[int], DatasetBundle]] = {
        'high_cardinality_categorical': _load_high_cardinality_categorical,
        'large_numeric': _load_large_numeric,
        'mixed_hard': _load_mixed_hard,
    }

    normalized_name = name.strip().lower()
    if normalized_name not in loaders:
        available = ', '.join(sorted(loaders))
        raise ValueError(
            f'Unknown dataset profile: {name}. Available profiles: {available}'
        )

    return loaders[normalized_name](seed)
