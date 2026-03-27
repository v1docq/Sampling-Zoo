from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence
import sys

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, log_loss
from sklearn.model_selection import KFold
from tqdm.auto import tqdm

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.metrics.eval_metrics import calculate_metrics
from core.utils.sampling_ensemble import SamplingEnsemble
from core.utils.amlb_dataloader import AMLBDatasetLoader
from core.utils.utils import safe_index
from benchmark_datasets import DatasetBundle, RawDatasetBundle
from benchmark_logging import BenchmarkLogger, build_sample_stats
from benchmark_adapters import _normalize_scores,_select_top_k_by_importance,_strategy_base_name
from benchmark_repo import AMLB_CATEGORY_PROFILES
from benchmark_sampling_strategies import make_strategies
from benchmark_models import _to_dense


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
        model_factory: Callable[[], ClassifierMixin],
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
                    model_factory=model_factory,
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
        model_factory: Callable[[], ClassifierMixin],
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
        model = model_factory()
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
        metrics["log_loss"] = float("nan")
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
                        metrics["log_loss"] = float(log_loss(y_true, y_proba))
                    else:
                        metrics["roc_auc"] = float("nan")
                        metrics["log_loss"] = float("nan")
                else:
                    metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba[:, -1]))
                    metrics["log_loss"] = float(log_loss(y_true, y_proba))
            else:
                metrics["roc_auc"] = float("nan")
                metrics["log_loss"] = float("nan")
            return metrics

        if model_classes is None:
            metrics["roc_auc"] = float("nan")
            metrics["log_loss"] = float("nan")
            return metrics

        classes_arr = np.asarray(model_classes)
        missing_classes = [cls for cls in classes_true.tolist() if cls not in set(classes_arr.tolist())]
        if missing_classes:
            metrics["roc_auc"] = float("nan")
            metrics["log_loss"] = float("nan")
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
        metrics["log_loss"] = float(log_loss(y_true, y_proba_selected, labels=target_classes))
    except ValueError:
        metrics["roc_auc"] = float("nan")
        metrics["log_loss"] = float("nan")

    return metrics


class EnsembleChunkBenchmarkRunner:
    """Runner for chunk-based SamplingEnsemble benchmarks on raw AMLB datasets."""

    def __init__(
        self,
        logger: Optional[BenchmarkLogger] = None,
        cv_folds: int = 3,
        seed: int = 42,
        show_progress: bool = True,
        on_record: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        self.logger = logger or BenchmarkLogger()
        self.cv_folds = cv_folds
        self.seed = seed
        self.show_progress = show_progress
        self.loader = AMLBDatasetLoader()
        self.on_record = on_record

    def run_dataset(
        self,
        dataset: RawDatasetBundle,
        strategy_configs: Mapping[str, Mapping[str, Any]],
        model_pool: Mapping[str, Callable[[], Any]],
    ) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        model_iter = tqdm(
            model_pool.items(),
            total=len(model_pool),
            disable=not self.show_progress,
            desc=f"Models ({dataset.name})",
            leave=False,
        )
        for model_name, model_factory in model_iter:
            strategy_iter = tqdm(
                strategy_configs.items(),
                total=len(strategy_configs),
                disable=not self.show_progress,
                desc=f"Strategies ({dataset.name}/{model_name})",
                leave=False,
            )
            for strategy_name, partitioner_config in strategy_iter:
                fold_iter = tqdm(
                    self._iter_folds(dataset),
                    total=self.cv_folds,
                    disable=not self.show_progress,
                    desc=f"Folds ({dataset.name}/{strategy_name}/{model_name})",
                    leave=False,
                )
                for fold_idx, X_train, X_val, X_test, y_train, y_val, y_test in fold_iter:
                    record = self._run_single_fold(
                        dataset=dataset,
                        strategy_name=strategy_name,
                        partitioner_config=partitioner_config,
                        model_name=model_name,
                        model_factory=model_factory,
                        fold_idx=fold_idx,
                        X_train=X_train,
                        X_val=X_val,
                        X_test=X_test,
                        y_train=y_train,
                        y_val=y_val,
                        y_test=y_test,
                    )
                    if self.on_record is not None:
                        self.on_record(record)
                    records.append(record)

        return records

    def _iter_folds(
        self,
        dataset: RawDatasetBundle,
    ) -> Iterable[tuple[int, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]]:
        X = dataset.X.to_numpy() if isinstance(dataset.X, pd.DataFrame) else np.asarray(dataset.X)
        y = dataset.y.to_numpy() if isinstance(dataset.y, pd.Series) else np.asarray(dataset.y)

        splitter = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.seed)
        split_iter = splitter.split(X)
        for fold_idx, (train_idx, test_idx) in enumerate(split_iter, start=1):
            X_train_full = dataset.X.iloc[train_idx].reset_index(drop=True)
            y_train_full = dataset.y.iloc[train_idx].reset_index(drop=True)
            X_test = dataset.X.iloc[test_idx].reset_index(drop=True)
            y_test = dataset.y.iloc[test_idx].reset_index(drop=True)

            X_train, _, X_val, y_train, _, y_val = self.loader.prepare_train_val_test_balanced(
                X_train_full,
                y_train_full,
                test_size=0.2,
                val_size=0,
                min_samples=20,
                problem=dataset.problem_type,
                random_state=self.seed + fold_idx,
            )
            yield (
                fold_idx,
                X_train,
                X_val,
                X_test,
                y_train,
                y_val,
                y_test,
            )

    @staticmethod
    def _ensure_dataframe(X: Any, dataset: RawDatasetBundle) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            df = pd.DataFrame(X, columns=dataset.feature_columns)
        for col in dataset.categorical_columns:
            if col in df.columns:
                df[col] = df[col].astype("category")
        return df

    @staticmethod
    def _class_representatives(
        X_train: Any,
        y_train: Any,
        seed: int,
    ) -> Dict[Any, tuple[pd.Series, Any]]:
        rng = np.random.default_rng(seed)
        y_array = np.asarray(y_train)
        representatives: Dict[Any, tuple[pd.Series, Any]] = {}
        for cls in np.unique(y_array):
            cls_indices = np.where(y_array == cls)[0]
            picked = int(rng.choice(cls_indices))
            representatives[cls] = (safe_index(X_train, picked), safe_index(y_train, picked))
        return representatives

    def _run_single_fold(
        self,
        dataset: RawDatasetBundle,
        strategy_name: str,
        partitioner_config: Mapping[str, Any],
        model_name: str,
        model_factory: Callable[[], Any],
        fold_idx: int,
        X_train: Any,
        X_val: Any,
        X_test: Any,
        y_train: Any,
        y_val: Any,
        y_test: Any,
    ) -> Dict[str, Any]:
        try:
            X_train_df = self._ensure_dataframe(X_train, dataset)
            X_val_df = self._ensure_dataframe(X_val, dataset)
            X_test_df = self._ensure_dataframe(X_test, dataset)

            class_samples = None
            if dataset.problem_type == "classification":
                class_samples = self._class_representatives(X_train_df, y_train, seed=self.seed + fold_idx)

            effective_partitions = max(1, int(np.ceil(len(y_train) / 20000)))
            target_chunks = 10
            chunks_percent = min(100.0, 100.0 * target_chunks / max(1, effective_partitions))
            tuned_partitioner_config = dict(partitioner_config)
            tuned_partitioner_config["n_partitions"] = effective_partitions
            tuned_partitioner_config["chunks_percent"] = chunks_percent

            ensemble = SamplingEnsemble(
                problem=dataset.problem_type,
                partitioner_config=tuned_partitioner_config,
                model_factory=model_factory,
                ensemble_method="voting",
            )

            fit_started = perf_counter()
            ensemble.train_partition_models(
                X_train=X_train_df,
                y_train=y_train,
                X_val=X_val_df,
                y_val=y_val,
                class_samples=class_samples,
                cv_fold=fold_idx,
                validation_metric="f1_weighted" if dataset.problem_type == "classification" else "rmse",
                train_all_chunks=True,
                save_models_to_disk=False,
            )
            fit_time = perf_counter() - fit_started

            infer_started = perf_counter()
            predictions = ensemble.ensemble_predict(X_test_df)
            infer_time = perf_counter() - infer_started

            model_metrics = calculate_metrics(
                y_true=y_test,
                y_labels=predictions,
                y_proba=None,
                problem_type=dataset.problem_type,
            )

            chunk_sizes = [int(model_info.get("data_size", 0)) for model_info in ensemble.models]
            sample_stats = build_sample_stats(
                y_sampled=np.asarray(y_train),
                total_train_size=len(y_train),
            )
            sample_stats["chunk_count"] = int(len(chunk_sizes))
            sample_stats["chunk_size_mean"] = float(np.mean(chunk_sizes)) if chunk_sizes else 0.0

            return self.logger.log_strategy_run(
                dataset_name=dataset.name,
                strategy_name=f"{strategy_name}__{model_name}__fold_{fold_idx}",
                strategy_params={
                    **tuned_partitioner_config,
                    "model": model_name,
                    "cv_fold": fold_idx,
                },
                model_metrics=model_metrics,
                timings={"fit": fit_time, "sample": 0.0, "inference": infer_time},
                sample_stats=sample_stats,
                extra={
                    "problem_type": dataset.problem_type,
                    "strategy": strategy_name,
                    "model": model_name,
                    "cv_fold": fold_idx,
                    "effective_partitions": effective_partitions,
                    "chunks_percent": chunks_percent,
                    "n_chunks": len(ensemble.models),
                    "chunk_sizes": chunk_sizes,
                    "partition_metrics": ensemble.partition_metrics,
                    "source_path": dataset.source_path,
                    "n_train": int(len(X_train)),
                    "n_val": int(len(X_val)),
                    "n_test": int(len(X_test)),
                },
            )
        except Exception as ex:
            return self.logger.log_strategy_run(
                dataset_name=dataset.name,
                strategy_name=f"{strategy_name}__{model_name}__fold_{fold_idx}",
                strategy_params={
                    **dict(partitioner_config),
                    "model": model_name,
                    "cv_fold": fold_idx,
                },
                model_metrics={},
                timings={"fit": 0.0, "sample": 0.0, "inference": 0.0},
                sample_stats=build_sample_stats(
                    y_sampled=np.array([], dtype=float),
                    total_train_size=max(len(y_train), 1),
                ),
                extra={
                    "problem_type": dataset.problem_type,
                    "strategy": strategy_name,
                    "model": model_name,
                    "cv_fold": fold_idx,
                    "effective_partitions": effective_partitions,
                    "chunks_percent": chunks_percent,
                    "source_path": dataset.source_path,
                    "error": str(ex),
                },
            )

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
