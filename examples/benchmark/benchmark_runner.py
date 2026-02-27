from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import ClassifierMixin, clone
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from tqdm.auto import tqdm

from benchmark_datasets import DatasetBundle
from benchmark_logging import BenchmarkLogger, build_sample_stats
from benchmark_adapters import _normalize_scores,_select_top_k_by_importance,_strategy_base_name
from benchmar_repo import AMLB_CATEGORY_PROFILES
from benchmark_sampling_strategies import make_strategies
from bechmark_models import _to_dense


@dataclass
class StrategyOutput:
    sample_indices: Sequence[int]
    strategy_params: Optional[Mapping[str, Any]] = None
    sample_scores: Optional[Sequence[float]] = None
    cluster_labels: Optional[Sequence[Any]] = None
    cell_ids: Optional[Sequence[Any]] = None
    simplex_ids: Optional[Sequence[Any]] = None
    extra: Optional[Mapping[str, Any]] = None


class SpecialStrategyBenchmarkRunner:
    """Benchmark runner for sampling strategies with consistent artifact logging."""

    def __init__(
        self,
        logger: Optional[BenchmarkLogger] = None,
        enable_diagnostic_plots: bool = False,
        show_progress: bool = True,
    ) -> None:
        self.logger = logger or BenchmarkLogger()
        self.enable_diagnostic_plots = enable_diagnostic_plots
        self.show_progress = show_progress
        self.run_records: List[Dict[str, Any]] = []

    def run(
        self,
        datasets: Iterable[DatasetBundle],
        strategies: Mapping[str, Callable[[DatasetBundle], Mapping[str, Any]]],
        base_model: ClassifierMixin,
    ) -> List[Dict[str, Any]]:
        self.run_records = []
        datasets_seq = list(datasets)

        dataset_iter = tqdm(
            datasets_seq,
            disable=not self.show_progress,
            desc="Datasets",
            leave=False,
        )
        for dataset in dataset_iter:
            x_train_dense = _to_dense(dataset.X_train_processed)
            x_test_dense = _to_dense(dataset.X_test_processed)
            y_train = dataset.y_train.to_numpy()
            y_test = dataset.y_test.to_numpy()

            strategy_iter = tqdm(
                strategies.items(),
                total=len(strategies),
                disable=not self.show_progress,
                desc=f"Strategies ({dataset.name})",
                leave=False,
            )
            for strategy_name, strategy_fn in strategy_iter:
                run_payload = self._run_single_strategy(
                    dataset=dataset,
                    strategy_name=strategy_name,
                    strategy_fn=strategy_fn,
                    base_model=base_model,
                    x_train_dense=x_train_dense,
                    y_train=y_train,
                    x_test_dense=x_test_dense,
                    y_test=y_test,
                )
                self.run_records.append(run_payload)

        self.logger.create_markdown_report(self.run_records)
        return self.run_records

    def _run_single_strategy(
        self,
        dataset: DatasetBundle,
        strategy_name: str,
        strategy_fn: Callable[[DatasetBundle], Mapping[str, Any]],
        base_model: ClassifierMixin,
        x_train_dense: np.ndarray,
        y_train: np.ndarray,
        x_test_dense: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, Any]:
        sample_started = perf_counter()
        strategy_output = dict(strategy_fn(dataset))
        sample_time = perf_counter() - sample_started

        sampled_indices = _resolve_sample_indices(strategy_output, len(y_train))
        x_sampled = x_train_dense[sampled_indices]
        y_sampled = y_train[sampled_indices]

        fit_started = perf_counter()
        model = clone(base_model)
        model.fit(x_sampled, y_sampled)
        fit_time = perf_counter() - fit_started

        infer_started = perf_counter()
        y_pred = model.predict(x_test_dense)
        y_proba = model.predict_proba(x_test_dense) if hasattr(model, "predict_proba") else None
        inference_time = perf_counter() - infer_started

        model_metrics = _collect_metrics(y_test, y_pred, y_proba, getattr(model, "classes_", None))
        sample_stats = build_sample_stats(
            y_sampled=y_sampled,
            total_train_size=len(y_train),
            cluster_labels=_from_output(strategy_output, "cluster_labels", sampled_indices),
            cell_ids=_from_output(strategy_output, "cell_ids", sampled_indices),
            simplex_ids=_from_output(strategy_output, "simplex_ids", sampled_indices),
        )

        payload = self.logger.log_strategy_run(
            dataset_name=dataset.name,
            strategy_name=strategy_name,
            strategy_params=strategy_output.get("strategy_params", {}),
            model_metrics=model_metrics,
            timings={"fit": fit_time, "sample": sample_time, "inference": inference_time},
            sample_stats=sample_stats,
            extra={
                "sample_indices_path": str(self.logger.save_sample_dump(dataset.name, strategy_name, sampled_indices)),
                **(strategy_output.get("extra", {}) or {}),
            },
        )

        if self.enable_diagnostic_plots:
            score_values = _resolve_score_values(strategy_output, sampled_indices)
            if score_values is not None:
                self.logger.save_probability_distribution_plot(score_values, dataset.name, strategy_name)

            self.logger.save_class_coverage_plot(sample_stats["class_distribution"], dataset.name, strategy_name)
            self.logger.save_2d_projection_plot(x_sampled, y_sampled, dataset.name, strategy_name, method="pca")
        return payload



def _resolve_sample_indices(strategy_output: Mapping[str, Any], train_size: int) -> np.ndarray:
    sample_indices = np.asarray(strategy_output.get("sample_indices", []), dtype=int)
    if sample_indices.size == 0:
        raise ValueError("Strategy output must include non-empty 'sample_indices'.")

    sample_indices = np.unique(sample_indices)
    sample_indices = sample_indices[(sample_indices >= 0) & (sample_indices < train_size)]
    if sample_indices.size == 0:
        raise ValueError("All sample indices are out of valid train range.")
    return sample_indices


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


def _collect_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray],
    model_classes: Optional[Sequence[Any]] = None,
) -> Dict[str, float]:
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

    classes_true = np.unique(y_true)
    try:
        if len(classes_true) == 2:
            if y_proba.ndim == 2 and y_proba.shape[1] > 1:
                if model_classes is not None:
                    classes_arr = np.asarray(model_classes)
                    positive_class = classes_true[-1]
                    pos_idx = np.where(classes_arr == positive_class)[0]
                    if pos_idx.size == 1:
                        metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba[:, int(pos_idx[0])]))
                    else:
                        metrics["roc_auc"] = float("nan")
                else:
                    metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba[:, -1]))
            else:
                metrics["roc_auc"] = float("nan")
            return metrics

        if model_classes is None:
            metrics["roc_auc"] = float("nan")
            return metrics

        classes_arr = np.asarray(model_classes)
        missing_classes = [cls for cls in classes_true.tolist() if cls not in set(classes_arr.tolist())]
        if missing_classes:
            metrics["roc_auc"] = float("nan")
            return metrics

        target_classes = np.sort(classes_true)
        target_indices = [int(np.where(classes_arr == cls)[0][0]) for cls in target_classes]
        y_proba_selected = y_proba[:, target_indices]

        row_sums = y_proba_selected.sum(axis=1, keepdims=True)
        safe_row_sums = np.where(row_sums > 0, row_sums, 1.0)
        y_proba_selected = y_proba_selected / safe_row_sums

        metrics["roc_auc"] = float(
            roc_auc_score(
                y_true,
                y_proba_selected,
                labels=target_classes,
                multi_class="ovr",
                average="macro",
            )
        )
    except ValueError:
        metrics["roc_auc"] = float("nan")

    return metrics

def _apply_budget_policy(
    informative_indices: Sequence[int],
    informative_scores: Sequence[float] | None,
    train_size: int,
    budget_ratio: float,
    seed: int,
) -> dict[str, Any]:
    informative = np.unique(np.asarray(informative_indices, dtype=int))
    informative = informative[(informative >= 0) & (informative < train_size)]
    if informative.size == 0:
        raise ValueError("Sampling strategy returned no informative indices.")

    budget_size = max(1, int(round(train_size * budget_ratio)))
    budget_size = min(budget_size, train_size)

    normalized_scores = _normalize_scores(informative_scores, informative, train_size)

    selected = _select_top_k_by_importance(informative, normalized_scores, budget_size)
    action = "truncate_to_top_k" if informative.size > budget_size else "keep_informative"

    if selected.size < budget_size:
        rng = np.random.default_rng(seed)
        remaining = np.setdiff1d(np.arange(train_size, dtype=int), selected, assume_unique=False)
        needed = min(budget_size - selected.size, remaining.size)
        if needed > 0:
            filled = rng.choice(remaining, size=needed, replace=False)
            selected = np.concatenate([selected, filled])
            action = "top_up_with_raw_samples"

    selected = np.unique(selected)
    if selected.size > budget_size:
        selected = selected[:budget_size]

    return {
        "sample_indices": selected,
        "budget_size": int(budget_size),
        "budget_ratio": float(budget_ratio),
        "informative_size": int(informative.size),
        "policy_action": action,
    }


def with_budget_variants(
    strategy_name: str,
    strategy_fn,
    budget_ratios: Sequence[float],
    seed: int,
):
    def _runner(bundle: DatasetBundle, budget_ratio: float) -> dict[str, Any]:
        result = dict(strategy_fn(bundle))
        raw_indices = np.asarray(result.get("sample_indices", []), dtype=int)
        raw_scores = result.get("sample_scores")

        budgeted = _apply_budget_policy(
            informative_indices=raw_indices,
            informative_scores=raw_scores,
            train_size=bundle.y_train.shape[0],
            budget_ratio=budget_ratio,
            seed=seed,
        )

        result["sample_indices"] = budgeted["sample_indices"]
        result["strategy_params"] = {
            **(result.get("strategy_params") or {}),
            "budget_ratio": budgeted["budget_ratio"],
            "budget_size": budgeted["budget_size"],
            "informative_size": budgeted["informative_size"],
            "budget_policy_action": budgeted["policy_action"],
        }

        result["extra"] = {
            **(result.get("extra") or {}),
            "informative_indices": np.unique(raw_indices).tolist(),
            "budget_ratio": budgeted["budget_ratio"],
            "budget_size": budgeted["budget_size"],
        }
        return result

    wrapped = {}
    for ratio in budget_ratios:
        ratio_tag = f"{int(round(ratio * 100)):02d}"

        def _factory(bundle: DatasetBundle, current_ratio: float = ratio):
            return _runner(bundle, current_ratio)

        wrapped[f"{strategy_name}__budget_{ratio_tag}"] = _factory
    return wrapped


def make_benchmark_strategies(seed: int, budget_ratios: Sequence[float]) -> dict[str, Any]:
    strategy_pool = make_strategies(seed=seed)
    wrapped: dict[str, Any] = {}
    for strategy_name, strategy_fn in strategy_pool.items():
        wrapped.update(with_budget_variants(strategy_name, strategy_fn, budget_ratios, seed))

    def full_dataset_strategy(bundle: DatasetBundle) -> dict[str, Any]:
        full_indices = np.arange(bundle.y_train.shape[0], dtype=int)
        return {
            "sample_indices": full_indices,
            "strategy_params": {
                "budget_ratio": 1.0,
                "budget_size": int(full_indices.size),
                "informative_size": int(full_indices.size),
                "budget_policy_action": "full_dataset_baseline",
            },
            "extra": {"informative_indices": full_indices.tolist(), "budget_ratio": 1.0},
        }

    wrapped["full_dataset"] = full_dataset_strategy
    return wrapped


def _with_enriched_dimensions(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    enriched["strategy_base"] = enriched["strategy"].map(_strategy_base_name)
    enriched["budget_ratio"] = pd.to_numeric(enriched.get("strategy_params.budget_ratio"), errors="coerce")
    enriched["budget_ratio"] = enriched["budget_ratio"].fillna(pd.to_numeric(enriched.get("extra.budget_ratio"), errors="coerce"))
    enriched["budget_percent"] = (enriched["budget_ratio"] * 100.0).round(2)
    enriched["model"] = enriched["strategy"].str.rsplit("__", n=1).str[-1]
    return enriched


def resolve_datasets(full_benchmark: bool, include_amlb: bool, amlb_categories: Sequence[str] | None) -> list[str]:
    dataset_names = ["mixed_hard"]
    if full_benchmark:
        dataset_names = ["high_cardinality_categorical", "large_numeric", "mixed_hard"]

    if include_amlb:
        dataset_names.extend(["amlb_adult", "amlb_covertype"])

    for category in amlb_categories or []:
        datasets_for_category = AMLB_CATEGORY_PROFILES.get(category)
        if not datasets_for_category:
            raise ValueError(f"Unknown AMLB category: {category}. Available: {sorted(AMLB_CATEGORY_PROFILES)}")
        dataset_names.extend(datasets_for_category)

    unique_names = []
    seen = set()
    for name in dataset_names:
        if name not in seen:
            seen.add(name)
            unique_names.append(name)
    return unique_names