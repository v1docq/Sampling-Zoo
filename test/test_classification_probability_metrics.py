from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from examples.benchmark.rmt_classification_amlb_datasets import (
    RMTClassificationExperimentConfig,
    RMTClassificationExperimentOrchestrator,
    _resolve_classification_tasks,
)
from examples.benchmark.rmt_regression_em_refinement import (
    RMTRegressionEMExperimentConfig,
    RMTRegressionEMExperimentOrchestrator,
)
from sampling_zoo.core.experiment.errors import ClassificationProbabilitiesRequiredError
from sampling_zoo.core.metrics.eval_metrics import calculate_metrics, metric_drop
from sampling_zoo.core.utils.sampling_ensemble import SamplingEnsemble


def test_binary_classification_metrics_use_positive_class_probability() -> None:
    y_true = np.asarray([0, 1, 1, 0])
    y_proba = np.asarray([
        [0.90, 0.10],
        [0.20, 0.80],
        [0.35, 0.65],
        [0.70, 0.30],
    ])
    metrics = calculate_metrics(
        y_true=y_true,
        y_labels=np.argmax(y_proba, axis=1),
        y_proba=y_proba,
        problem_type="classification",
        classes=np.asarray([0, 1]),
    )

    assert metrics["roc_auc"] == 1.0
    assert np.isfinite(metrics["log_loss"])


def test_multiclass_classification_metrics_use_log_loss_without_auc() -> None:
    y_true = np.asarray([0, 1, 2, 1])
    y_proba = np.asarray([
        [0.80, 0.10, 0.10],
        [0.10, 0.80, 0.10],
        [0.15, 0.15, 0.70],
        [0.20, 0.60, 0.20],
    ])
    metrics = calculate_metrics(
        y_true=y_true,
        y_labels=np.argmax(y_proba, axis=1),
        y_proba=y_proba,
        problem_type="classification",
        classes=np.asarray([0, 1, 2]),
    )

    assert np.isnan(metrics["roc_auc"])
    assert metrics["log_loss"] < 0.5


def test_metric_drop_respects_auc_and_logloss_direction() -> None:
    assert metric_drop("roc_auc", score=0.91, reference_score=0.95) == pytest.approx(0.04)
    assert metric_drop("log_loss", score=0.44, reference_score=0.40) == pytest.approx(0.04)


class _StaticProbaModel:
    def __init__(self, classes, proba):
        self.classes_ = np.asarray(classes)
        self.proba = np.asarray(proba, dtype=float)

    def predict_proba(self, X):
        return self.proba[:len(X)]

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class _MissingProbaModel:
    def predict(self, X):
        return np.zeros(len(X), dtype=int)


def test_ensemble_predict_proba_aligns_global_classes_and_sums_to_one() -> None:
    X = pd.DataFrame({"x": [0, 1, 2]})
    ensemble = SamplingEnsemble(
        problem="classification",
        ensemble_method="weighted",
        show_progress=False,
    )
    ensemble.classes_ = np.asarray([0, 1, 2])
    ensemble.models = [
        {
            "name": "chunk_0",
            "model": _StaticProbaModel([0, 1, 2], [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]]),
            "metrics": {"log_loss": 0.4, "f1_weighted": 0.8},
            "classes": np.asarray([0, 1, 2]),
        },
        {
            "name": "chunk_1",
            "model": _StaticProbaModel([2, 1, 0], [[0.1, 0.3, 0.6], [0.2, 0.7, 0.1], [0.7, 0.2, 0.1]]),
            "metrics": {"log_loss": 0.5, "f1_weighted": 0.7},
            "classes": np.asarray([2, 1, 0]),
        },
    ]

    proba = ensemble.ensemble_predict_proba(X)

    assert proba.shape == (3, 3)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert np.array_equal(ensemble.ensemble_predict(X), np.asarray([0, 1, 2]))


def test_missing_predict_proba_fails_explicitly_for_classification() -> None:
    ensemble = SamplingEnsemble(problem="classification", ensemble_method="voting", show_progress=False)
    ensemble.classes_ = np.asarray([0, 1])
    ensemble.models = [{
        "name": "chunk_0",
        "model": _MissingProbaModel(),
        "metrics": {"f1_weighted": 0.5},
        "classes": np.asarray([0, 1]),
    }]

    with pytest.raises(ClassificationProbabilitiesRequiredError) as exc:
        ensemble.ensemble_predict_proba(pd.DataFrame({"x": [1, 2]}))

    assert exc.value.code == "classification_probabilities_required"


def test_classification_partition_diagnostics_detect_missing_and_single_class_chunks() -> None:
    ensemble = SamplingEnsemble(problem="classification", show_progress=False)
    diagnostics = ensemble._build_partition_target_diagnostics(
        {
            "chunk_0": {"target": pd.Series([0, 0, 0])},
            "chunk_1": {"target": pd.Series([0, 1, 2])},
        },
        pd.Series([0, 0, 1, 1, 2, 2]),
    )

    assert diagnostics["class_balance_summary"]["chunks_with_missing_classes"] == 1
    assert diagnostics["class_balance_summary"]["single_class_chunks"] == 1
    assert diagnostics["chunks"]["chunk_0"]["missing_classes"] == ["1", "2"]


def test_classification_runner_resolves_amlb_aliases_and_builds_probability_grid() -> None:
    assert _resolve_classification_tasks(("amlb_adult", "optdigits")) == ("adult", "optdigits")

    orchestrator = RMTClassificationExperimentOrchestrator(
        RMTClassificationExperimentConfig(
            classification_tasks=("amlb_adult",),
            budget_ratios=(0.1,),
            router_modes=("spectral", "constrained_gating"),
            show_progress=False,
            synthetic_smoke=True,
        )
    )
    grid = orchestrator._build_strategy_grid().materialize()

    assert "full_dataset" in grid
    assert any(config.get("strategy") == "rmt_contraction" for config in grid.values())
    assert any(config.get("router") == "constrained_gating" for config in grid.values())
    assert all(config.get("strategy") != "feature_clustering" for config in grid.values())


def test_regression_em_runner_grid_is_targeted_to_rmt_routed_configs() -> None:
    orchestrator = RMTRegressionEMExperimentOrchestrator(
        RMTRegressionEMExperimentConfig(
            regression_tasks=("diamonds",),
            budget_ratios=(0.1,),
            router_modes=("spectral", "constrained_gating"),
            show_progress=False,
            synthetic_smoke=True,
        )
    )
    grid = orchestrator._build_strategy_grid().materialize()
    non_full = [config for name, config in grid.items() if name != "full_dataset"]

    assert non_full
    assert {config["strategy"] for config in non_full} == {"rmt_contraction"}
    assert {config["ensemble_method"] for config in non_full} == {"routed_weighted"}
    assert all(config["routing_refinement"] == "em_retraining" for config in non_full)
