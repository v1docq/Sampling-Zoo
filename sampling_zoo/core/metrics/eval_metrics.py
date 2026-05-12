import operator
from typing import Any, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss, mean_squared_error, r2_score, roc_auc_score


LOWER_IS_BETTER = {"rmse", "mae", "mse", "log_loss", "logloss", "cross_entropy"}
HIGHER_IS_BETTER = {"accuracy", "f1", "f1_macro", "f1_weighted", "recall", "precision", "roc_auc"}


def calculate_metrics(
    y_true,
    y_labels,
    y_proba,
    problem_type,
    classes: Sequence[Any] | None = None,
):
    """Compute benchmark metrics with probability-first classification semantics."""
    if problem_type == "classification":
        return _calculate_classification_metrics(y_true, y_labels, y_proba, classes=classes)

    y_pred = np.asarray(y_labels)
    return {
        "mse": mean_squared_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
    }


def _calculate_classification_metrics(
    y_true,
    y_labels,
    y_proba,
    *,
    classes: Sequence[Any] | None,
) -> dict[str, float]:
    y_true_arr = np.asarray(y_true)
    y_pred = np.asarray(y_labels).reshape(-1)
    class_labels = _resolve_class_labels(y_true_arr, classes, y_proba)
    metrics = {
        "accuracy": float(accuracy_score(y_true_arr, y_pred)),
        "f1_macro": float(f1_score(y_true_arr, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true_arr, y_pred, average="weighted", zero_division=0)),
        "roc_auc": float("nan"),
        "log_loss": float("nan"),
    }

    if y_proba is None:
        return metrics

    proba = _normalize_probability_matrix(np.asarray(y_proba, dtype=float))
    if proba.ndim != 2 or proba.shape[0] != y_true_arr.shape[0] or proba.shape[1] == 0:
        return metrics

    try:
        metrics["log_loss"] = float(log_loss(y_true_arr, proba, labels=class_labels))
    except ValueError:
        metrics["log_loss"] = float("nan")

    if len(class_labels) == 2:
        positive_class = class_labels[-1]
        positive_idx = _class_index(class_labels, positive_class)
        if positive_idx is not None and positive_idx < proba.shape[1]:
            try:
                metrics["roc_auc"] = float(roc_auc_score(y_true_arr, proba[:, positive_idx]))
            except ValueError:
                metrics["roc_auc"] = float("nan")

    return metrics


def _resolve_class_labels(
    y_true: np.ndarray,
    classes: Sequence[Any] | None,
    y_proba: Any,
) -> np.ndarray:
    if classes is not None:
        return np.asarray(classes)
    if y_proba is not None and getattr(y_proba, "ndim", 0) == 2:
        n_columns = int(np.asarray(y_proba).shape[1])
        unique = np.unique(y_true)
        if unique.shape[0] == n_columns:
            return unique
        return np.arange(n_columns)
    return np.unique(y_true)


def _class_index(classes: np.ndarray, label: Any) -> int | None:
    matches = np.where(classes == label)[0]
    if matches.size != 1:
        return None
    return int(matches[0])


def _normalize_probability_matrix(proba: np.ndarray) -> np.ndarray:
    proba = np.asarray(proba, dtype=float)
    proba = np.clip(proba, 1e-15, 1.0)
    row_sums = proba.sum(axis=1, keepdims=True)
    return np.where(row_sums > 0, proba / row_sums, 1.0 / max(proba.shape[1], 1))


def get_metric_comparator(metric_name: str):
    """Return comparator(a, b), where True means a is better than b."""
    direction = metric_direction(metric_name)
    return operator.lt if direction == "lower" else operator.gt


def metric_direction(metric_name: str) -> str:
    name = metric_name.lower()
    if name in LOWER_IS_BETTER:
        return "lower"
    if name in HIGHER_IS_BETTER:
        return "higher"
    raise ValueError(f"Unknown metric type for '{metric_name}'")


def metric_drop(metric_name: str, score: float, reference_score: float) -> float:
    """Positive values mean degradation relative to the reference score."""
    if metric_direction(metric_name) == "lower":
        return float(score - reference_score)
    return float(reference_score - score)
