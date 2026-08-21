from __future__ import annotations

import numpy as np
import pandas as pd

from sampling_zoo.core.utils.sampling_ensemble import SamplingEnsemble


class _IdentityRegressor:
    def predict(self, features):
        return np.asarray(features.iloc[:, 0])


def test_internal_batch_inference_uses_requested_batch_size_without_dropping_rows() -> None:
    ensemble = SamplingEnsemble(
        problem="regression",
        model_factory=_IdentityRegressor,
        show_progress=False,
    )
    features = pd.DataFrame({"value": np.arange(25)})

    labels, predictions = ensemble._run_inference(
        _IdentityRegressor(),
        features,
        calculation_mode="batch",
        batch_size=10,
    )

    np.testing.assert_array_equal(labels, np.arange(25))
    np.testing.assert_array_equal(predictions, np.arange(25))
