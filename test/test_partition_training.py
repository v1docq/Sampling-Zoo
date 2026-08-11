from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sampling_zoo.core.experiment.contracts import PartitionModelMode
from sampling_zoo.core.utils.partition_training import (
    build_model_training_partitions,
    normalize_partition_model_mode,
)
from sampling_zoo.core.utils.sampling_ensemble import SamplingEnsemble


class _MeanRegressor:
    def fit(self, X, y):
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_, dtype=float)


class _WeightedMeanRegressor:
    def fit(self, X, y, sample_weight=None):
        self.sample_weight_ = np.asarray(sample_weight, dtype=float)
        self.mean_ = float(np.average(y, weights=self.sample_weight_))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_, dtype=float)


def test_concatenated_training_partition_preserves_rows_and_order() -> None:
    partitions = {
        "chunk_0": {
            "feature": pd.DataFrame({"x": [1.0, 2.0]}),
            "target": pd.Series([10.0, 20.0]),
        },
        "chunk_1": {
            "feature": pd.DataFrame({"x": [3.0]}),
            "target": pd.Series([30.0]),
        },
    }

    result = build_model_training_partitions(
        partitions,
        PartitionModelMode.CONCATENATED,
    )

    assert tuple(result) == ("concatenated_budget",)
    assert result["concatenated_budget"]["feature"]["x"].tolist() == [
        1.0,
        2.0,
        3.0,
    ]
    assert result["concatenated_budget"]["target"].tolist() == [
        10.0,
        20.0,
        30.0,
    ]


def test_concatenated_training_partition_preserves_sample_weights() -> None:
    partitions = {
        "chunk_0": {
            "feature": pd.DataFrame({"x": [1.0, 2.0]}),
            "target": pd.Series([10.0, 20.0]),
            "sample_weight": np.asarray([0.5, 1.5]),
        },
        "chunk_1": {
            "feature": pd.DataFrame({"x": [3.0]}),
            "target": pd.Series([30.0]),
            "sample_weight": np.asarray([2.0]),
        },
    }

    result = build_model_training_partitions(
        partitions,
        PartitionModelMode.CONCATENATED,
    )

    assert np.array_equal(
        result["concatenated_budget"]["sample_weight"],
        np.asarray([0.5, 1.5, 2.0]),
    )


def test_sampling_ensemble_passes_partition_sample_weights_to_model() -> None:
    partition = {
        "feature": pd.DataFrame({"x": [1.0, 2.0, 3.0]}),
        "target": pd.Series([2.0, 4.0, 9.0]),
        "sample_weight": np.asarray([1.0, 1.0, 4.0]),
    }
    model = _WeightedMeanRegressor()

    SamplingEnsemble._fit_partition_model(model, partition)

    assert np.array_equal(model.sample_weight_, partition["sample_weight"])
    assert model.mean_ == pytest.approx(7.0)


def test_sampling_ensemble_trains_one_model_on_concatenated_budget() -> None:
    rng = np.random.default_rng(42)
    X_train = pd.DataFrame(rng.normal(size=(60, 4)))
    y_train = pd.Series(2.0 * X_train[0] - X_train[1])
    X_val = pd.DataFrame(rng.normal(size=(20, 4)))
    y_val = pd.Series(2.0 * X_val[0] - X_val[1])
    ensemble = SamplingEnsemble(
        problem="regression",
        partitioner_config={
            "strategy": "random",
            "n_partitions": 3,
            "random_state": 42,
            "budget_ratio": 0.5,
            "partition_model_mode": "concatenated",
        },
        model_factory=_MeanRegressor,
        ensemble_method="voting",
        show_progress=False,
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

    contract = ensemble.partition_training_contract_
    assert contract.request.partition_model_mode == "concatenated"
    assert contract.request.n_partitions == 3
    assert contract.request.n_training_partitions == 1
    assert contract.partitions.n_partitions == 3
    assert len(ensemble.models) == 1
    assert ensemble.models[0]["name"] == "concatenated_budget"
    assert ensemble.models[0]["data_size"] == 30
    assert ensemble.validation_diagnostics_["partition_model_mode"] == (
        "concatenated"
    )


def test_concatenated_mode_rejects_routed_weighted() -> None:
    with pytest.raises(ValueError, match="incompatible"):
        SamplingEnsemble(
            problem="regression",
            partitioner_config={
                "strategy": "random",
                "partition_model_mode": "concatenated",
            },
            model_factory=_MeanRegressor,
            ensemble_method="routed_weighted",
            show_progress=False,
        )


def test_partition_model_mode_validation_is_explicit() -> None:
    assert normalize_partition_model_mode(None) is PartitionModelMode.INDEPENDENT
    with pytest.raises(ValueError, match="partition_model_mode"):
        normalize_partition_model_mode("unknown")
