from __future__ import annotations

import numpy as np
import pandas as pd

from sampling_zoo.core.utils.ensemble_routing import RoutedWeightedRouter


class _PartiallyAlignedPartitioner:
    partition_names_ = ["chunk_0", "chunk_1"]

    def predict_partition_proba(self, features):
        return np.asarray([[0.2, 0.8], [0.0, 0.0]])


def test_zero_routing_row_uses_uniform_fallback_without_warning() -> None:
    router = RoutedWeightedRouter(problem="regression")

    with np.errstate(divide="raise", invalid="raise"):
        weights = router.base_weights(
            features=pd.DataFrame({"x": [0.0, 1.0]}),
            active_models=[{"name": "chunk_0"}, {"name": "chunk_1"}],
            partitioner=_PartiallyAlignedPartitioner(),
        )

    np.testing.assert_allclose(weights[0], [0.2, 0.8])
    np.testing.assert_allclose(weights[1], [0.5, 0.5])
    np.testing.assert_allclose(weights.sum(axis=1), 1.0)
