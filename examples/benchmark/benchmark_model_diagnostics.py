"""Best-effort model complexity diagnostics for benchmark reports."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


def summarize_model_complexity(
    model: Any,
    X_reference: Any = None,
    *,
    shap_sample_size: int = 256,
) -> dict[str, Any]:
    booster = getattr(model, "booster_", None)
    if booster is None:
        return {
            "status": "unsupported",
            "model_type": type(model).__name__,
        }

    try:
        tree_diagnostics = _lightgbm_tree_diagnostics(booster)
        importance_diagnostics = _lightgbm_importance_diagnostics(booster)
    except Exception as exc:
        return {
            "status": "failed",
            "model_type": type(model).__name__,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }

    shap_diagnostics = _lightgbm_shap_diagnostics(
        model,
        X_reference,
        sample_size=shap_sample_size,
    )
    return {
        "status": "ok",
        "model_type": type(model).__name__,
        **tree_diagnostics,
        **importance_diagnostics,
        "shap": shap_diagnostics,
    }


def summarize_ensemble_complexity(
    models: Sequence[Mapping[str, Any]],
    X_reference: Any = None,
) -> dict[str, Any]:
    per_model = []
    for index, model_info in enumerate(models):
        diagnostics = summarize_model_complexity(
            model_info.get("model"),
            X_reference,
        )
        per_model.append(
            {
                "name": str(model_info.get("name", f"model_{index}")),
                **diagnostics,
            }
        )

    supported = [
        diagnostics
        for diagnostics in per_model
        if diagnostics.get("status") == "ok"
    ]
    if not supported:
        return {
            "status": "unsupported",
            "model_count": len(per_model),
            "supported_model_count": 0,
            "models": per_model,
        }

    return {
        "status": "ok",
        "model_count": len(per_model),
        "supported_model_count": len(supported),
        "tree_count_total": _sum_metric(supported, "tree_count"),
        "leaf_count_total": _sum_metric(supported, "leaf_count"),
        "split_count_total": _sum_metric(supported, "split_count"),
        "max_tree_depth": _max_metric(supported, "max_tree_depth"),
        "mean_tree_depth": _mean_metric(supported, "mean_tree_depth"),
        "gain_importance_entropy": _mean_metric(
            supported,
            "gain_importance_entropy",
        ),
        "shap_mean_abs_entropy": _mean_nested_metric(
            supported,
            "shap",
            "mean_abs_entropy",
        ),
        "shap_top_5_share": _mean_nested_metric(
            supported,
            "shap",
            "top_5_share",
        ),
        "models": per_model,
    }


def _lightgbm_tree_diagnostics(booster: Any) -> dict[str, Any]:
    tree_info = booster.dump_model().get("tree_info", [])
    stats = [
        _tree_structure_stats(tree.get("tree_structure", {}))
        for tree in tree_info
    ]
    return {
        "tree_count": len(stats),
        "leaf_count": int(sum(item[0] for item in stats)),
        "split_count": int(sum(item[1] for item in stats)),
        "mean_tree_depth": (
            float(np.mean([item[2] for item in stats]))
            if stats
            else 0.0
        ),
        "max_tree_depth": int(max((item[2] for item in stats), default=0)),
    }


def _tree_structure_stats(node: Mapping[str, Any], depth: int = 0) -> tuple[int, int, int]:
    if "split_index" not in node:
        return 1, 0, depth
    left = _tree_structure_stats(node.get("left_child", {}), depth + 1)
    right = _tree_structure_stats(node.get("right_child", {}), depth + 1)
    return (
        left[0] + right[0],
        1 + left[1] + right[1],
        max(left[2], right[2]),
    )


def _lightgbm_importance_diagnostics(booster: Any) -> dict[str, Any]:
    gains = np.asarray(booster.feature_importance("gain"), dtype=float)
    positive = gains[gains > 0]
    return {
        "nonzero_gain_feature_count": int(positive.size),
        "gain_importance_entropy": _normalized_entropy(positive),
        "gain_top_5_share": _top_k_share(positive, 5),
    }


def _lightgbm_shap_diagnostics(
    model: Any,
    X_reference: Any,
    *,
    sample_size: int,
) -> dict[str, Any]:
    if X_reference is None:
        return {"status": "not_requested"}
    try:
        sample = X_reference.iloc[:sample_size]
    except AttributeError:
        sample = np.asarray(X_reference)[:sample_size]
    try:
        contributions = np.asarray(
            model.predict(sample, pred_contrib=True),
            dtype=float,
        )
        if contributions.ndim != 2 or contributions.shape[1] < 2:
            raise ValueError("Unexpected pred_contrib output shape")
        mean_abs = np.mean(np.abs(contributions[:, :-1]), axis=0)
        return {
            "status": "ok",
            "sample_size": int(contributions.shape[0]),
            "nonzero_feature_count": int(np.count_nonzero(mean_abs)),
            "mean_abs_entropy": _normalized_entropy(mean_abs),
            "top_5_share": _top_k_share(mean_abs, 5),
        }
    except Exception as exc:
        return {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def _normalized_entropy(values: np.ndarray) -> float | None:
    positive = np.asarray(values, dtype=float)
    positive = positive[np.isfinite(positive) & (positive > 0)]
    if positive.size == 0:
        return None
    if positive.size == 1:
        return 0.0
    probabilities = positive / positive.sum()
    entropy = -np.sum(probabilities * np.log(probabilities))
    return float(entropy / np.log(positive.size))


def _top_k_share(values: np.ndarray, k: int) -> float | None:
    positive = np.asarray(values, dtype=float)
    positive = positive[np.isfinite(positive) & (positive > 0)]
    total = positive.sum()
    if total <= 0:
        return None
    return float(np.sort(positive)[-k:].sum() / total)


def _sum_metric(values: Sequence[Mapping[str, Any]], key: str) -> int:
    return int(sum(int(value.get(key, 0)) for value in values))


def _max_metric(values: Sequence[Mapping[str, Any]], key: str) -> int:
    return int(max((int(value.get(key, 0)) for value in values), default=0))


def _mean_metric(values: Sequence[Mapping[str, Any]], key: str) -> float | None:
    numeric = [float(value[key]) for value in values if value.get(key) is not None]
    return float(np.mean(numeric)) if numeric else None


def _mean_nested_metric(
    values: Sequence[Mapping[str, Any]],
    parent: str,
    key: str,
) -> float | None:
    numeric = [
        float(value[parent][key])
        for value in values
        if isinstance(value.get(parent), Mapping)
        and value[parent].get(key) is not None
    ]
    return float(np.mean(numeric)) if numeric else None
