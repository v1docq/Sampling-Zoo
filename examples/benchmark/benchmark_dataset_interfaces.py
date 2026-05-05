from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.datasets import make_regression

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
            rng = np.random.default_rng(self.seed)
            keep = np.sort(rng.choice(np.arange(len(y_train)), size=int(self.max_train_rows), replace=False))
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
