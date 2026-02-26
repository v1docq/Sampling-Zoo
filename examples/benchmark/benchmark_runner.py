from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
from sklearn.base import ClassifierMixin, clone
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from examples.special_strategy.benchmark_logging import BenchmarkLogger, build_sample_stats


@dataclass
class DatasetBundle:
    name: str
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


class SpecialStrategyBenchmarkRunner:
    """Runner for benchmark of sampling strategies with structured artifact logging."""

    def __init__(self, logger: Optional[BenchmarkLogger] = None) -> None:
        self.logger = logger or BenchmarkLogger()
        self.run_records: List[Dict[str, Any]] = []

    def run(
        self,
        datasets: Iterable[DatasetBundle],
        strategies: Mapping[str, Callable[[np.ndarray, np.ndarray], Mapping[str, Any]]],
        base_model: ClassifierMixin,
    ) -> List[Dict[str, Any]]:
        for dataset in datasets:
            for strategy_name, strategy_fn in strategies.items():
                run_payload = self._run_single_strategy(dataset, strategy_name, strategy_fn, base_model)
                self.run_records.append(run_payload)

        self.logger.create_markdown_report(self.run_records)
        return self.run_records

    def _run_single_strategy(
        self,
        dataset: DatasetBundle,
        strategy_name: str,
        strategy_fn: Callable[[np.ndarray, np.ndarray], Mapping[str, Any]],
        base_model: ClassifierMixin,
    ) -> Dict[str, Any]:
        sample_started = perf_counter()
        strategy_output = dict(strategy_fn(dataset.x_train, dataset.y_train))
        sample_time = perf_counter() - sample_started

        sampled_indices = _resolve_sample_indices(strategy_output, len(dataset.x_train))
        x_sampled = dataset.x_train[sampled_indices]
        y_sampled = dataset.y_train[sampled_indices]

        fit_started = perf_counter()
        model = clone(base_model)
        model.fit(x_sampled, y_sampled)
        fit_time = perf_counter() - fit_started

        infer_started = perf_counter()
        y_pred = model.predict(dataset.x_test)
        y_proba = model.predict_proba(dataset.x_test) if hasattr(model, "predict_proba") else None
        inference_time = perf_counter() - infer_started

        model_metrics = _collect_metrics(dataset.y_test, y_pred, y_proba)
        sample_stats = build_sample_stats(
            y_sampled=y_sampled,
            total_train_size=len(dataset.y_train),
            cluster_labels=_from_output(strategy_output, "cluster_labels", sampled_indices),
            cell_ids=_from_output(strategy_output, "cell_ids", sampled_indices),
            simplex_ids=_from_output(strategy_output, "simplex_ids", sampled_indices),
        )

        strategy_params = strategy_output.get("strategy_params", {})
        payload = self.logger.log_strategy_run(
            dataset_name=dataset.name,
            strategy_name=strategy_name,
            strategy_params=strategy_params,
            model_metrics=model_metrics,
            timings={"fit": fit_time, "sample": sample_time, "inference": inference_time},
            sample_stats=sample_stats,
            extra={"sample_indices_path": str(self.logger.save_sample_dump(dataset.name, strategy_name, sampled_indices))},
        )

        score_values = _resolve_score_values(strategy_output, sampled_indices)
        if score_values is not None:
            self.logger.save_probability_distribution_plot(score_values, dataset.name, strategy_name)

        self.logger.save_class_coverage_plot(sample_stats["class_distribution"], dataset.name, strategy_name)
        self.logger.save_2d_projection_plot(x_sampled, y_sampled, dataset.name, strategy_name, method="pca")
        return payload


def _resolve_sample_indices(strategy_output: Mapping[str, Any], train_size: int) -> np.ndarray:
    if "sample_indices" in strategy_output:
        return np.asarray(strategy_output["sample_indices"], dtype=int)

    if "mask" in strategy_output:
        mask = np.asarray(strategy_output["mask"], dtype=bool)
        return np.where(mask)[0]

    if "sampled_x" in strategy_output:
        sampled_x = np.asarray(strategy_output["sampled_x"])
        if sampled_x.shape[0] == train_size:
            return np.arange(train_size)

    raise ValueError("Strategy output must include either 'sample_indices' or 'mask'.")


def _resolve_score_values(strategy_output: Mapping[str, Any], sampled_indices: Sequence[int]) -> Optional[np.ndarray]:
    if "sample_scores" in strategy_output:
        return np.asarray(strategy_output["sample_scores"], dtype=float)

    if "weights" in strategy_output:
        weights = np.asarray(strategy_output["weights"], dtype=float)
        if weights.shape[0] == len(sampled_indices):
            return weights
        if len(sampled_indices) > 0 and weights.shape[0] > int(np.max(sampled_indices)):
            return weights[np.asarray(sampled_indices, dtype=int)]

    return None


def _from_output(strategy_output: Mapping[str, Any], key: str, sampled_indices: Sequence[int]) -> Optional[np.ndarray]:
    values = strategy_output.get(key)
    if values is None:
        return None

    arr = np.asarray(values)
    if arr.shape[0] == len(sampled_indices):
        return arr

    if len(sampled_indices) > 0 and arr.shape[0] > int(np.max(sampled_indices)):
        return arr[np.asarray(sampled_indices, dtype=int)]

    return None


def _collect_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray]) -> Dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }

    if y_proba is None:
        metrics["roc_auc"] = float("nan")
        return metrics

    classes_count = len(np.unique(y_true))
    if classes_count == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
    else:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro"))

    return metrics
