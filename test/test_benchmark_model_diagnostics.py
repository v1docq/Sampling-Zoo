from __future__ import annotations

import numpy as np
import pandas as pd

from examples.benchmark.benchmark_model_diagnostics import (
    summarize_ensemble_complexity,
    summarize_model_complexity,
)


class _FakeBooster:
    def dump_model(self):
        return {
            "tree_info": [
                {
                    "tree_structure": {
                        "split_index": 0,
                        "left_child": {"leaf_index": 0},
                        "right_child": {"leaf_index": 1},
                    }
                },
                {"tree_structure": {"leaf_index": 0}},
            ]
        }

    def feature_importance(self, importance_type):
        assert importance_type == "gain"
        return np.asarray([3.0, 1.0, 0.0])


class _FakeLightGBMModel:
    booster_ = _FakeBooster()

    def predict(self, X, pred_contrib=False):
        assert pred_contrib
        return np.tile(np.asarray([2.0, 1.0, 0.0]), (len(X), 1))


def test_lightgbm_complexity_summarizes_trees_importance_and_shap() -> None:
    diagnostics = summarize_model_complexity(
        _FakeLightGBMModel(),
        pd.DataFrame({"x0": [0.0, 1.0], "x1": [1.0, 0.0]}),
    )

    assert diagnostics["status"] == "ok"
    assert diagnostics["tree_count"] == 2
    assert diagnostics["leaf_count"] == 3
    assert diagnostics["split_count"] == 1
    assert diagnostics["max_tree_depth"] == 1
    assert diagnostics["gain_top_5_share"] == 1.0
    assert diagnostics["shap"]["status"] == "ok"
    assert diagnostics["shap"]["sample_size"] == 2


def test_unsupported_model_complexity_is_non_fatal() -> None:
    diagnostics = summarize_model_complexity(object())

    assert diagnostics == {
        "status": "unsupported",
        "model_type": "object",
        "model_module": "builtins",
        "model_params": {},
    }


def test_ensemble_complexity_aggregates_supported_models() -> None:
    diagnostics = summarize_ensemble_complexity(
        (
            {"name": "chunk_0", "model": _FakeLightGBMModel()},
            {"name": "chunk_1", "model": _FakeLightGBMModel()},
        ),
        np.ones((3, 2)),
    )

    assert diagnostics["status"] == "ok"
    assert diagnostics["model_count"] == 2
    assert diagnostics["tree_count_total"] == 4
    assert diagnostics["leaf_count_total"] == 6
    assert diagnostics["split_count_total"] == 2
    assert diagnostics["max_tree_depth"] == 1
    assert diagnostics["shap_top_5_share"] == 1.0
