from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression

from benchmark_datasets import OpenMLRawDatasetBundle, RawDatasetBundle, RawDatasetMetadata


@dataclass(frozen=True)
class CappedOpenMLRawDatasetBundle(OpenMLRawDatasetBundle):
    max_train_rows: int | None = None
    seed: int = 42

    def load_split_data(self, show_progress: bool = True):
        X_train, y_train, X_test, y_test, feature_columns, categorical_columns, numeric_columns = super().load_split_data(
            show_progress=show_progress,
        )
        if self.max_train_rows is not None and len(y_train) > self.max_train_rows:
            if self.problem_type == "classification":
                keep = _stratified_cap_indices(y_train, int(self.max_train_rows), self.seed)
            else:
                rng = np.random.default_rng(self.seed)
                keep = np.sort(rng.choice(np.arange(len(y_train)), size=int(self.max_train_rows), replace=False))
            X_train = X_train.iloc[keep].reset_index(drop=True)
            y_train = y_train.iloc[keep].reset_index(drop=True)
        return X_train, y_train, X_test, y_test, feature_columns, categorical_columns, numeric_columns


def _stratified_cap_indices(y: pd.Series, max_rows: int, seed: int) -> np.ndarray:
    y_series = pd.Series(y).reset_index(drop=True)
    n_rows = len(y_series)
    if max_rows >= n_rows:
        return np.arange(n_rows, dtype=int)
    classes = y_series.dropna().unique()
    if max_rows < len(classes):
        max_rows = len(classes)

    rng = np.random.default_rng(seed)
    class_indices = {cls: np.flatnonzero((y_series == cls).to_numpy()) for cls in classes}
    selected: list[int] = []
    for cls in classes:
        indices = class_indices[cls]
        if indices.size > 0:
            selected.append(int(rng.choice(indices)))

    remaining_budget = max(0, max_rows - len(selected))
    selected_set = set(selected)
    remaining_indices = np.asarray([idx for idx in range(n_rows) if idx not in selected_set], dtype=int)
    if remaining_budget > 0 and remaining_indices.size > 0:
        chosen = rng.choice(remaining_indices, size=min(remaining_budget, remaining_indices.size), replace=False)
        selected.extend(int(idx) for idx in chosen)

    return np.sort(np.asarray(selected[:max_rows], dtype=int))


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


def make_synthetic_classification_smoke_dataset(seed: int, n_classes: int = 3) -> RawDatasetBundle:
    X_values, y_values = make_classification(
        n_samples=900,
        n_features=14,
        n_informative=8,
        n_redundant=2,
        n_classes=n_classes,
        n_clusters_per_class=1,
        random_state=seed,
    )
    X = pd.DataFrame(X_values, columns=[f"x_{idx}" for idx in range(X_values.shape[1])])
    y = pd.Series(y_values, name="target")
    numeric_columns = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_columns = [col for col in X.columns if col not in numeric_columns]
    return RawDatasetBundle(
        name=f"synthetic_rmt_classification_smoke_{n_classes}c",
        problem_type="classification",
        target_name="target",
        source_path="synthetic://rmt_classification_smoke",
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
