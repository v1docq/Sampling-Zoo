"""Shared metric calculation and comparison helpers."""

from __future__ import annotations

import operator
from typing import Any, Optional, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

LOWER_IS_BETTER = {
    "rmse",
    "mae",
    "mse",
    "logloss",
    "log_loss",
    "cross_entropy",
    "brier_score",
    "expected_calibration_error",
    "ece",
}
HIGHER_IS_BETTER = {
    "accuracy",
    "f1",
    "f1_macro",
    "f1_weighted",
    "recall",
    "precision",
    "roc_auc",
}


def calculate_metrics(
    y_true: Any,
    y_labels: Any,
    y_proba: Optional[np.ndarray],
    problem_type: str,
    classes: Optional[Sequence[Any]] = None,
) -> dict[str, float]:
    """Calculate regression metrics or probability-first classification metrics."""

    if problem_type == "classification":
        return _classification_metrics(
            y_true=y_true,
            y_labels=y_labels,
            y_proba=y_proba,
            classes=classes,
        )

    y_pred = np.asarray(y_labels)
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _classification_metrics(
    *,
    y_true: Any,
    y_labels: Any,
    y_proba: Optional[np.ndarray],
    classes: Optional[Sequence[Any]],
) -> dict[str, float]:
    y_true_array = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_labels).reshape(-1)
    global_classes = (
        np.asarray(classes)
        if classes is not None
        else np.unique(y_true_array)
    )
    metrics = {
        "accuracy": float(accuracy_score(y_true_array, y_pred)),
        "f1_macro": float(
            f1_score(y_true_array, y_pred, average="macro", zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(y_true_array, y_pred, average="weighted", zero_division=0)
        ),
        "roc_auc": float("nan"),
        "log_loss": float("nan"),
        "brier_score": float("nan"),
        "expected_calibration_error": float("nan"),
    }
    if y_proba is None:
        return metrics

    probabilities = np.asarray(y_proba, dtype=float)
    if (
        probabilities.ndim != 2
        or probabilities.shape[0] != y_true_array.size
        or probabilities.shape[1] != global_classes.size
    ):
        return metrics

    try:
        metrics["log_loss"] = float(
            log_loss(y_true_array, probabilities, labels=global_classes)
        )
        metrics["brier_score"] = classification_brier_score(
            y_true_array,
            probabilities,
            global_classes,
        )
        metrics["expected_calibration_error"] = (
            expected_calibration_error(
                y_true_array,
                probabilities,
                global_classes,
            )
        )
        if global_classes.size == 2:
            positive_class = global_classes[-1]
            positive_index = int(
                np.where(global_classes == positive_class)[0][0]
            )
            binary_target = (y_true_array == positive_class).astype(int)
            metrics["roc_auc"] = float(
                roc_auc_score(
                    binary_target,
                    probabilities[:, positive_index],
                )
            )
    except ValueError:
        pass
    return metrics


def classification_brier_score(
    y_true: Any,
    probabilities: np.ndarray,
    classes: Sequence[Any],
) -> float:
    """Return binary or multiclass Brier score for class-aligned probabilities."""

    target = np.asarray(y_true).reshape(-1)
    proba = np.asarray(probabilities, dtype=float)
    class_labels = np.asarray(classes)
    encoded = _encode_class_targets(target, class_labels)
    if class_labels.size == 2:
        positive_target = (encoded == 1).astype(float)
        return float(np.mean((proba[:, 1] - positive_target) ** 2))

    one_hot = np.eye(class_labels.size, dtype=float)[encoded]
    return float(np.mean(np.sum((proba - one_hot) ** 2, axis=1)))


def expected_calibration_error(
    y_true: Any,
    probabilities: np.ndarray,
    classes: Sequence[Any],
    *,
    n_bins: int = 10,
) -> float:
    """Measure confidence/accuracy mismatch using equal-width confidence bins."""

    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    target = np.asarray(y_true).reshape(-1)
    proba = np.asarray(probabilities, dtype=float)
    class_labels = np.asarray(classes)
    encoded = _encode_class_targets(target, class_labels)
    confidence = np.max(proba, axis=1)
    predicted = np.argmax(proba, axis=1)
    correct = (predicted == encoded).astype(float)
    bin_index = np.minimum(
        (np.clip(confidence, 0.0, 1.0) * n_bins).astype(int),
        n_bins - 1,
    )
    error = 0.0
    for current_bin in range(n_bins):
        mask = bin_index == current_bin
        if not np.any(mask):
            continue
        weight = float(np.mean(mask))
        error += weight * abs(
            float(np.mean(correct[mask]))
            - float(np.mean(confidence[mask]))
        )
    return float(error)


def _encode_class_targets(
    target: np.ndarray,
    classes: np.ndarray,
) -> np.ndarray:
    class_to_index = {value: index for index, value in enumerate(classes.tolist())}
    try:
        return np.asarray(
            [class_to_index[value] for value in target.tolist()],
            dtype=int,
        )
    except KeyError as error:
        raise ValueError("y_true contains a class absent from classes") from error


def get_metric_comparator(metric_name: str):
    """Return comparator(a, b) that is true when metric value a is better."""

    name = metric_name.lower()
    if name in LOWER_IS_BETTER:
        return operator.lt
    if name in HIGHER_IS_BETTER:
        return operator.gt
    raise ValueError(f"Unknown metric type for '{metric_name}'")


def primary_classification_metric(classes: Sequence[Any]) -> str:
    """Return the AMLB-compatible primary metric for a class set."""

    return "roc_auc" if np.asarray(classes).size == 2 else "log_loss"


def metric_drop(metric_name: str, score: float, reference_score: float) -> float:
    """Return positive degradation relative to a reference metric value."""

    normalized = str(metric_name).strip().lower()
    if normalized in HIGHER_IS_BETTER:
        return float(reference_score - score)
    if normalized in LOWER_IS_BETTER:
        return float(score - reference_score)
    raise ValueError(f"Unknown metric type for '{metric_name}'")
