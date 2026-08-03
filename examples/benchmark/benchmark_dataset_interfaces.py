from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression

try:
    from .benchmark_datasets import (
        OpenMLRawDatasetBundle,
        RawDatasetBundle,
        RawDatasetMetadata,
    )
except ImportError:  # pragma: no cover - direct script execution
    from benchmark_datasets import (
        OpenMLRawDatasetBundle,
        RawDatasetBundle,
        RawDatasetMetadata,
    )


@dataclass(frozen=True)
class CappedOpenMLRawDatasetBundle(OpenMLRawDatasetBundle):
    max_train_rows: int | None = None
    seed: int = 42

    def load_split_data(self, show_progress: bool = True):
        X_train, y_train, X_test, y_test, feature_columns, categorical_columns, numeric_columns = super().load_split_data(
            show_progress=show_progress,
        )
        if self.max_train_rows is not None and len(y_train) > self.max_train_rows:
            keep = (
                stratified_cap_indices(
                    y_train,
                    int(self.max_train_rows),
                    self.seed,
                )
                if self.problem_type == "classification"
                else _random_cap_indices(
                    len(y_train),
                    int(self.max_train_rows),
                    self.seed,
                )
            )
            X_train = X_train.iloc[keep].reset_index(drop=True)
            y_train = y_train.iloc[keep].reset_index(drop=True)
        return X_train, y_train, X_test, y_test, feature_columns, categorical_columns, numeric_columns


def cap_openml_dataset(
    dataset: OpenMLRawDatasetBundle,
    max_train_rows: int | None,
    seed: int,
) -> CappedOpenMLRawDatasetBundle:
    return CappedOpenMLRawDatasetBundle(
        name=dataset.name,
        problem_type=dataset.problem_type,
        target_name=dataset.target_name,
        source_path=dataset.source_path,
        X=dataset.X,
        y=dataset.y,
        metadata=dataset.metadata,
        feature_columns=dataset.feature_columns,
        categorical_columns=dataset.categorical_columns,
        numeric_columns=dataset.numeric_columns,
        task_id=dataset.task_id,
        task_name=dataset.task_name,
        suite_id=dataset.suite_id,
        dataset_id=dataset.dataset_id,
        max_train_rows=max_train_rows,
        seed=seed,
    )


def make_synthetic_regression_smoke_dataset(seed: int) -> RawDatasetBundle:
    X_values, y_values = make_regression(
        n_samples=900,
        n_features=14,
        n_informative=8,
        noise=15.0,
        random_state=seed,
    )
    X = pd.DataFrame(X_values, columns=[f"x_{idx}" for idx in range(X_values.shape[1])])
    y = pd.Series(y_values, name="target")
    numeric_columns = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_columns = [col for col in X.columns if col not in numeric_columns]
    return RawDatasetBundle(
        name="synthetic_rmt_regression_smoke",
        problem_type="regression",
        target_name="target",
        source_path="synthetic://rmt_regression_smoke",
        X=X,
        y=y,
        metadata=RawDatasetMetadata(
            n_objects=int(X.shape[0]),
            n_features=int(X.shape[1]),
            n_train_candidates=int(X.shape[0]),
            n_categorical=len(categorical_columns),
            n_numeric=len(numeric_columns),
        ),
        feature_columns=X.columns.tolist(),
        categorical_columns=categorical_columns,
        numeric_columns=numeric_columns,
    )


def make_synthetic_classification_smoke_dataset(
    seed: int,
    *,
    n_classes: int,
) -> RawDatasetBundle:
    if n_classes < 2:
        raise ValueError("n_classes must be at least two")
    weights = (
        [0.78, 0.22]
        if n_classes == 2
        else [0.55, 0.30] + [0.15 / (n_classes - 2)] * (n_classes - 2)
    )
    X_values, y_values = make_classification(
        n_samples=1200,
        n_features=16,
        n_informative=10,
        n_redundant=2,
        n_classes=n_classes,
        n_clusters_per_class=1,
        weights=weights,
        class_sep=1.2,
        random_state=seed,
    )
    X = pd.DataFrame(
        X_values,
        columns=[f"x_{index}" for index in range(X_values.shape[1])],
    )
    y = pd.Series(y_values, name="target")
    task_name = "binary" if n_classes == 2 else f"multiclass_{n_classes}"
    return RawDatasetBundle(
        name=f"synthetic_rmt_classification_{task_name}_smoke",
        problem_type="classification",
        target_name="target",
        source_path=f"synthetic://rmt_classification_{task_name}_smoke",
        X=X,
        y=y,
        metadata=RawDatasetMetadata(
            n_objects=int(X.shape[0]),
            n_features=int(X.shape[1]),
            n_train_candidates=int(X.shape[0]),
            n_categorical=0,
            n_numeric=int(X.shape[1]),
        ),
        feature_columns=X.columns.tolist(),
        categorical_columns=[],
        numeric_columns=X.columns.tolist(),
    )


def stratified_cap_indices(
    target: pd.Series | np.ndarray,
    cap: int,
    seed: int,
) -> np.ndarray:
    """Return an exact deterministic cap that preserves every class when feasible."""

    values = np.asarray(target).reshape(-1)
    cap = max(0, min(int(cap), values.size))
    if cap == values.size:
        return np.arange(values.size, dtype=int)
    if cap == 0:
        return np.asarray([], dtype=int)
    classes, encoded = np.unique(values, return_inverse=True)
    if cap < classes.size:
        return _random_cap_indices(values.size, cap, seed)

    counts = np.bincount(encoded, minlength=classes.size)
    allocation = np.ones(classes.size, dtype=int)
    remaining = cap - classes.size
    residual_capacity = counts - allocation
    if remaining > 0:
        ideal = remaining * residual_capacity / max(residual_capacity.sum(), 1)
        additions = np.minimum(np.floor(ideal).astype(int), residual_capacity)
        allocation += additions
        remaining -= int(additions.sum())
        remainders = ideal - np.floor(ideal)
        while remaining > 0:
            candidates = np.where(allocation < counts)[0]
            if candidates.size == 0:
                break
            best = int(
                candidates[
                    np.argmax(remainders[candidates])
                ]
            )
            allocation[best] += 1
            remainders[best] = -1.0
            remaining -= 1

    rng = np.random.default_rng(seed)
    selected = []
    for class_index, class_size in enumerate(allocation):
        candidates = np.where(encoded == class_index)[0]
        selected.extend(
            rng.choice(candidates, size=int(class_size), replace=False).tolist()
        )
    return np.sort(np.asarray(selected, dtype=int))


def _random_cap_indices(n_rows: int, cap: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.sort(
        rng.choice(np.arange(n_rows), size=int(cap), replace=False)
    )
