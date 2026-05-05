from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

from sampling_zoo.core.sampling_strategies.spectral.rmt_contraction_sampler import (
    RMTContractionConfig,
    RMTContractionTensorSampler,
)


def _frame(n_samples: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    X = pd.DataFrame(
        rng.normal(size=(n_samples, 6)),
        columns=[f"x_{idx}" for idx in range(6)],
    )
    X["cat"] = pd.Series([f"c_{idx % 12}" for idx in range(n_samples)], dtype="string")
    return X


def test_numpy_backend_predict_partition_proba_rows_sum_to_one() -> None:
    X = _frame()
    sampler = RMTContractionTensorSampler(
        n_partitions=4,
        n_views=5,
        projection_dim=3,
        approx_rank=3,
        max_encoded_features=10,
        backend="numpy",
        random_state=7,
        show_progress=False,
    )

    sampler.fit(X)
    proba = sampler.predict_partition_proba(X.iloc[:10])

    assert proba.shape == (10, len(sampler.partition_names_))
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert sampler.diagnostics_["backend"] == "numpy"


def test_config_constructor_and_legacy_positional_arguments() -> None:
    config = RMTContractionConfig(
        n_partitions=2,
        n_views=3,
        projection_dim=2,
        backend="numpy",
        show_progress=False,
    )

    from_config = RMTContractionTensorSampler(config=config, n_partitions=4)
    from_positionals = RMTContractionTensorSampler(3, 4, projection_dim=2, backend="numpy", show_progress=False)

    assert from_config.n_partitions == 4
    assert from_config.n_views == 3
    assert from_config.backend == "numpy"
    assert from_positionals.n_partitions == 3
    assert from_positionals.n_views == 4


def test_auto_backend_resolves_to_available_backend() -> None:
    X = _frame()
    sampler = RMTContractionTensorSampler(
        n_partitions=3,
        n_views=4,
        projection_dim=2,
        approx_rank=2,
        backend="auto",
        random_state=11,
        show_progress=False,
    )

    sampler.fit(X)

    assert sampler.backend_ in {"numpy", "torch"}
    if importlib.util.find_spec("torch") is None:
        assert sampler.backend_ == "numpy"


def test_sparse_encoded_features_are_capped_before_dense_math() -> None:
    X = _frame(n_samples=120)
    sampler = RMTContractionTensorSampler(
        n_partitions=3,
        n_views=3,
        projection_dim=2,
        approx_rank=2,
        max_one_hot_cardinality=20,
        max_encoded_features=5,
        backend="numpy",
        random_state=13,
        show_progress=False,
    )

    sampler.fit(X)

    assert sampler.encoded_feature_subset_ is not None
    assert len(sampler.encoded_feature_subset_) == 5
    assert sampler.diagnostics_["raw_encoded_feature_count"] > sampler.diagnostics_["encoded_feature_count"]
    assert sampler.diagnostics_["encoded_feature_count"] == 5


def test_partition_probability_columns_align_with_partition_names() -> None:
    X = _frame()
    sampler = RMTContractionTensorSampler(
        n_partitions=5,
        n_views=4,
        projection_dim=3,
        approx_rank=3,
        chunks_percent=60,
        backend="numpy",
        random_state=17,
        show_progress=False,
    )

    sampler.fit(X)
    proba = sampler.predict_partition_proba(X.iloc[:7])

    assert proba.shape[1] == len(sampler.partition_names_)
    assert set(sampler.partition_names_) == set(sampler.partitions)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_backend_if_available() -> None:
    X = _frame()
    sampler = RMTContractionTensorSampler(
        n_partitions=3,
        n_views=4,
        projection_dim=2,
        approx_rank=2,
        backend="torch",
        device="cpu",
        random_state=19,
        show_progress=False,
    )

    sampler.fit(X)
    proba = sampler.predict_partition_proba(X.iloc[:5])

    assert sampler.backend_ == "torch"
    assert np.allclose(proba.sum(axis=1), 1.0)
