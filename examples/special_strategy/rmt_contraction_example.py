from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

from sampling_zoo.core.utils.sampling_ensemble import SamplingEnsemble


def main() -> None:
    X, y = make_regression(n_samples=5000, n_features=40, noise=20.0, random_state=42)
    X = pd.DataFrame(X, columns=[f"x_{i}" for i in range(X.shape[1])])
    y = pd.Series(y, name="target")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=43)

    ensemble = SamplingEnsemble(
        problem="regression",
        partitioner_config={
            "strategy": "rmt_contraction",
            "n_partitions": 4,
            "n_views": 12,
            "projection_dim": 8,
            "initial_rank_fraction": 0.25,
            "rank_selection_method": "explained_variance",
            "explained_variance_threshold": 0.95,
            "selection_method": "hybrid",
            "chunk_fraction": 0.75,
            "routing_temperature": 1.0,
            "routing_shrinkage": 0.05,
        },
        model_factory=lambda: RandomForestRegressor(n_estimators=80, random_state=42, n_jobs=-1),
        ensemble_method="routed_weighted",
    )

    ensemble.train_partition_models(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        class_samples=None,
        cv_fold=1,
        validation_metric="rmse",
        train_all_chunks=True,
        save_models_to_disk=False,
    )

    preds = ensemble.ensemble_predict_batch(X_test, batch_size=1000)
    rmse = float(np.sqrt(np.mean((np.asarray(y_test) - preds) ** 2)))
    print({"rmse": rmse, "n_models": len(ensemble.models)})
    print(getattr(ensemble.partitioner, "diagnostics_", {}))


if __name__ == "__main__":
    main()
