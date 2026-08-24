"""Адаптеры стратегий для унифицированного benchmark-пайплайна.

Цели:
1. Спрятать различия в API стратегий (`sample_indices`, `predict_partitions`, `get_partitions`).
2. Нормализовать выход в единый формат для метрик benchmark.
3. Изолировать ошибки конкретной стратегии через fallback, чтобы benchmark продолжал работу.
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

@dataclass
class AdapterResult:
    """Единый формат результатов для benchmark."""

    selected_indices: Optional[List[Any]] = None
    partition_labels: Optional[np.ndarray] = None
    weights: Optional[np.ndarray] = None
    extra: Dict[str, Any] = field(default_factory=dict)


class StrategyAdapter:
    """Базовый адаптер с централизованной обработкой ошибок и fallback-логикой."""

    def __init__(self, strategy: Any, strategy_name: Optional[str] = None):
        self.strategy = strategy
        self.strategy_name = strategy_name or strategy.__class__.__name__
        self._diagnostics: Dict[str, Any] = {
            "strategy": self.strategy_name,
            "fit_status": "not_run",
            "fit_error": None,
            "sample_status": "not_run",
            "sample_error": None,
            "fallback_used": False,
            "traceback": None,
        }
        self._is_fitted = False

    def fit(self, X_train: Any, y_train: Any = None) -> "StrategyAdapter":
        """Безопасный вызов fit с поддержкой разных сигнатур."""
        try:
            self._fit_impl(X_train, y_train)
            self._is_fitted = True
            self._diagnostics["fit_status"] = "ok"
        except Exception as ex:
            self._diagnostics["fit_status"] = "error"
            self._diagnostics["fit_error"] = str(ex)
            self._diagnostics["traceback"] = traceback.format_exc()
            self._diagnostics["fallback_used"] = True
        return self

    def sample_or_partition(self, X_eval: Any) -> AdapterResult:
        """Безопасный вызов inference-шагов стратегии."""
        if not self._is_fitted:
            self._diagnostics["sample_status"] = "skipped"
            self._diagnostics["sample_error"] = "fit step failed or was not executed"
            self._diagnostics["fallback_used"] = True
            return self._fallback_result(X_eval)

        try:
            result = self._sample_or_partition_impl(X_eval)
            self._diagnostics["sample_status"] = "ok"
            return result
        except Exception as ex:
            self._diagnostics["sample_status"] = "error"
            self._diagnostics["sample_error"] = str(ex)
            self._diagnostics["traceback"] = traceback.format_exc()
            self._diagnostics["fallback_used"] = True
            return self._fallback_result(X_eval)

    def collect_diagnostics(self) -> Dict[str, Any]:
        return dict(self._diagnostics)

    def _fit_impl(self, X_train: Any, y_train: Any = None) -> None:
        """Дефолтная попытка fit для стратегий с разными сигнатурами."""
        try:
            self.strategy.fit(X_train, y_train)
        except TypeError:
            try:
                self.strategy.fit(X_train, target=y_train)
            except TypeError:
                self.strategy.fit(X_train)

    def _sample_or_partition_impl(self, X_eval: Any) -> AdapterResult:
        raise NotImplementedError

    def _fallback_result(self, X_eval: Any) -> AdapterResult:
        n_samples = _get_n_samples(X_eval)
        return AdapterResult(
            selected_indices=[],
            partition_labels=np.full(n_samples, -1, dtype=int),
            weights=None,
            extra={"fallback": True},
        )


class SamplingOnlyAdapter(StrategyAdapter):
    """Адаптер для стратегий с `sample_indices` (SpectralLeverage/TensorEnergy)."""

    def _sample_or_partition_impl(self, X_eval: Any) -> AdapterResult:
        sampled = self.strategy.sample_indices()

        weights = None
        if isinstance(sampled, tuple) and len(sampled) == 2:
            selected_indices, weights = sampled
            weights = np.asarray(weights)
        else:
            selected_indices = sampled

        return AdapterResult(
            selected_indices=list(selected_indices),
            partition_labels=None,
            weights=weights,
            extra={"method": "sample_indices"},
        )


class PartitioningAdapter(StrategyAdapter):
    """Адаптер для стратегий с `predict_partitions` и/или `get_partitions`."""

    def _sample_or_partition_impl(self, X_eval: Any) -> AdapterResult:
        labels = None
        partitions_payload = None

        if hasattr(self.strategy, "predict_partitions"):
            labels = self.strategy.predict_partitions(X_eval)

        if labels is None and hasattr(self.strategy, "get_partitions"):
            try:
                partitions_payload = self.strategy.get_partitions()
            except TypeError:
                partitions_payload = getattr(self.strategy, "partitions", None)

            labels = _labels_from_partitions(partitions_payload, _get_n_samples(X_eval))

        if labels is None:
            raise RuntimeError(
                f"{self.strategy_name}: не удалось получить partition labels через "
                "predict_partitions/get_partitions"
            )

        labels = _normalize_labels(labels, _get_n_samples(X_eval))

        return AdapterResult(
            selected_indices=None,
            partition_labels=labels,
            weights=None,
            extra={
                "method": "predict_partitions_or_get_partitions",
                "partitions_payload_type": type(partitions_payload).__name__ if partitions_payload is not None else None,
            },
        )


class GenericAdapter(StrategyAdapter):
    """Универсальный адаптер: сначала пытается sample_indices, затем partition API."""

    def _sample_or_partition_impl(self, X_eval: Any) -> AdapterResult:
        if hasattr(self.strategy, "sample_indices"):
            return SamplingOnlyAdapter(self.strategy, self.strategy_name)._sample_or_partition_impl(X_eval)
        return PartitioningAdapter(self.strategy, self.strategy_name)._sample_or_partition_impl(X_eval)


class BenchmarkRunner:
    """Запускает набор адаптеров и изолирует сбои отдельных стратегий."""

    def __init__(self, adapters: Mapping[str, StrategyAdapter]):
        self.adapters = dict(adapters)

    def run(self, X_train: Any, y_train: Any, X_eval: Any) -> Dict[str, Dict[str, Any]]:
        results: Dict[str, Dict[str, Any]] = {}

        for name, adapter in self.adapters.items():
            adapter.fit(X_train, y_train)
            normalized = adapter.sample_or_partition(X_eval)

            results[name] = {
                "selected_indices": normalized.selected_indices,
                "partition_labels": normalized.partition_labels,
                "weights": normalized.weights,
                "extra": normalized.extra,
                "diagnostics": adapter.collect_diagnostics(),
            }

        return results


def build_default_adapters(strategies: Mapping[str, Any]) -> Dict[str, StrategyAdapter]:
    """Фабрика адаптеров с автодетекцией специальных стратегий."""
    adapters: Dict[str, StrategyAdapter] = {}

    for name, strategy in strategies.items():
        strategy_class_name = strategy.__class__.__name__

        if strategy_class_name in {"SpectralLeverageSampler", "TensorEnergySampler"}:
            adapters[name] = SamplingOnlyAdapter(strategy, name)
        elif strategy_class_name in {"VoronoiSampler", "HDBScanSampler", "DelaunaySampler"}:
            adapters[name] = PartitioningAdapter(strategy, name)
        else:
            adapters[name] = GenericAdapter(strategy, name)

    return adapters

def _get_n_samples(X: Any) -> int:
    if isinstance(X, (pd.DataFrame, pd.Series)):
        return int(X.shape[0])
    arr = np.asarray(X)
    return int(arr.shape[0])


def _labels_from_partitions(partitions: Any, n_samples: int) -> np.ndarray:
    labels = np.full(n_samples, -1, dtype=int)
    if not isinstance(partitions, Mapping):
        return labels

    for label_id, (_, indices) in enumerate(partitions.items()):
        try:
            idx = np.asarray(indices, dtype=int)
            labels[idx] = label_id
        except Exception:
            continue
    return labels


def _normalize_labels(raw_labels: Any, n_samples: int) -> np.ndarray:
    """Нормализация меток в shape=(n_samples,), неизвестное -> -1."""
    labels = np.full(n_samples, -1, dtype=int)

    if raw_labels is None:
        return labels

    # one_cluster=True case: ndarray[int]
    if isinstance(raw_labels, np.ndarray) and raw_labels.ndim == 1:
        labels[: min(n_samples, raw_labels.shape[0])] = raw_labels[:n_samples]
        return labels

    # one_cluster=False case in HDBSCAN: list[np.ndarray]
    if isinstance(raw_labels, Iterable) and not isinstance(raw_labels, (str, bytes, dict)):
        normalized: List[int] = []
        for item in list(raw_labels)[:n_samples]:
            if isinstance(item, np.ndarray):
                normalized.append(int(item[0]) if item.size > 0 else -1)
            elif isinstance(item, (list, tuple)):
                normalized.append(int(item[0]) if len(item) > 0 else -1)
            else:
                normalized.append(int(item))

        labels[: len(normalized)] = np.asarray(normalized, dtype=int)
        return labels

    return labels
def _strategy_base_name(strategy_name: str) -> str:
    budget_suffix = "__budget_"
    if budget_suffix in strategy_name:
        return strategy_name.split(budget_suffix, 1)[0]
    return strategy_name


def _select_top_k_by_importance(
    informative_indices: np.ndarray,
    informative_scores: np.ndarray | None,
    budget_size: int,
) -> np.ndarray:
    if informative_indices.size <= budget_size:
        return informative_indices

    if informative_scores is None:
        return informative_indices[:budget_size]

    ranked_positions = np.argsort(-informative_scores, kind="mergesort")
    return informative_indices[ranked_positions[:budget_size]]


def _normalize_scores(
    raw_scores: Sequence[float] | None,
    informative_indices: np.ndarray,
    train_size: int,
) -> np.ndarray | None:
    if raw_scores is None:
        return None

    scores = np.asarray(raw_scores, dtype=float).reshape(-1)
    if scores.shape[0] == informative_indices.shape[0]:
        return scores

    if scores.shape[0] == train_size:
        return scores[informative_indices]

    return None