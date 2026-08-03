from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence
import sys
import gc

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, log_loss
from sklearn.model_selection import KFold, train_test_split
from tqdm.auto import tqdm

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from sampling_zoo.core.metrics.eval_metrics import (
    calculate_metrics,
    primary_classification_metric,
)
from sampling_zoo.core.experiment.errors import (
    ClassificationProbabilitiesRequiredError,
    InvalidExperimentConfigError,
)
from sampling_zoo.core.experiment.contracts import (
    ModelStrategyScenarioGridContract,
    StrategyGridContract,
)
from sampling_zoo.core.experiment.morphisms import (
    dataset_to_contract,
    evaluation_to_contract,
    fold_to_contract,
    materialize_strategy_grid,
    normalize_strategy_grid,
)
from sampling_zoo.core.experiment.resume import (
    LeafRunKey,
    ResumePlan,
    leaf_run_key_from_components,
)
from sampling_zoo.core.utils.sampling_ensemble import SamplingEnsemble
from sampling_zoo.core.utils.amlb_dataloader import AMLBDatasetLoader
from sampling_zoo.core.utils.progress import progress_bar
from sampling_zoo.core.utils.utils import safe_index
from benchmark_datasets import DatasetBundle, OpenMLRawDatasetBundle, RawDatasetBundle
from benchmark_logging import BenchmarkLogger, build_sample_stats
from benchmark_adapters import _normalize_scores,_select_top_k_by_importance,_strategy_base_name
from benchmark_repo import AMLB_CATEGORY_PROFILES
from benchmark_sampling_strategies import make_strategies
from benchmark_models import _to_dense
from benchmark_model_diagnostics import summarize_ensemble_complexity


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



@dataclass(frozen=True)
class FoldSplit:
    fold_idx: int
    split_label: str
    X_train: Any
    X_val: Any
    X_test: Any
    y_train: Any
    y_val: Any
    y_test: Any


@dataclass(frozen=True)
class FoldExecutionPlan:
    force_chunking: bool
    force_direct_model: bool
    configured_partitions: int
    effective_partitions: int
    chunks_percent: float
    use_direct_model: bool


@dataclass(frozen=True)
class EnsembleSampleAccounting:
    """Separate allocated sampling budget from the retained ensemble size."""

    selected_rows: int
    selected_partition_count: int
    active_model_rows: int
    active_model_count: int
    class_coverage_rows_added: int

    @property
    def model_fit_rows_total(self) -> int:
        return self.selected_rows + self.class_coverage_rows_added


def _partition_row_count(partition: Any) -> int:
    if isinstance(partition, Mapping) and "feature" in partition:
        return int(len(partition["feature"]))
    return int(len(partition))


def _build_ensemble_sample_accounting(
    ensemble: SamplingEnsemble,
) -> EnsembleSampleAccounting:
    size_contract = getattr(
        ensemble,
        "partition_size_diagnostics_contract_",
        None,
    )
    if size_contract is not None:
        selected_rows = int(
            size_contract.selected_rows
            if size_contract.selected_rows is not None
            else sum(size_contract.post_budget_sizes.values())
        )
        selected_partition_count = len(size_contract.post_budget_sizes)
    else:
        partitions = getattr(ensemble, "partitions", {}) or {}
        selected_rows = sum(
            _partition_row_count(partition)
            for partition in partitions.values()
        )
        selected_partition_count = len(partitions)

    active_models = getattr(ensemble, "models", ()) or ()
    active_model_rows = sum(
        int(model_info.get("data_size", 0))
        for model_info in active_models
    )
    repairs = getattr(ensemble, "class_coverage_repairs_", {}) or {}
    class_coverage_rows_added = sum(
        int(repair.get("rows_added", 0))
        for repair in repairs.values()
        if isinstance(repair, Mapping)
    )
    return EnsembleSampleAccounting(
        selected_rows=int(selected_rows),
        selected_partition_count=int(selected_partition_count),
        active_model_rows=int(active_model_rows),
        active_model_count=len(active_models),
        class_coverage_rows_added=int(class_coverage_rows_added),
    )


def _selected_class_distribution(
    ensemble: SamplingEnsemble,
) -> dict[str, int]:
    diagnostics = getattr(ensemble, "partition_diagnostics_", {}) or {}
    chunks = diagnostics.get("chunks", {})
    distribution: dict[str, int] = {}
    if isinstance(chunks, Mapping):
        for chunk in chunks.values():
            if not isinstance(chunk, Mapping):
                continue
            for label, count in chunk.get("class_counts", {}).items():
                key = str(label)
                distribution[key] = distribution.get(key, 0) + int(count)
    return distribution


class EnsembleFoldBenchmarkExecutor:
    """Owns fold splitting, fold-level model execution, and fold result logging."""

    def __init__(
        self,
        logger: BenchmarkLogger,
        loader: AMLBDatasetLoader,
        cv_folds: int,
        seed: int,
        show_progress: bool,
        resume_plan: ResumePlan | None = None,
    ) -> None:
        self.logger = logger
        self.loader = loader
        self.cv_folds = cv_folds
        self.seed = seed
        self.show_progress = show_progress
        self.resume_plan = resume_plan
        self.skipped_leaf_keys: set[str] = set()

    def run_strategy_folds(
        self,
        dataset: RawDatasetBundle,
        strategy_name: str,
        partitioner_config: Mapping[str, Any],
        model_name: str,
        model_factory: Callable[[], Any],
        openml_split_data: Optional[tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, list[str], list[str], list[str]]] = None,
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        split_total = self.split_count(dataset)
        split_desc = "Splits" if split_total == 1 else "Folds"
        fold_iter = tqdm(
            self.iter_folds(dataset, openml_split_data=openml_split_data),
            total=split_total,
            disable=not self.show_progress,
            desc=f"{split_desc} ({dataset.name}/{strategy_name}/{model_name})",
            leave=False,
        )
        for fold in fold_iter:
            leaf_run = self._leaf_run_key(
                dataset=dataset,
                strategy_name=strategy_name,
                partitioner_config=partitioner_config,
                model_name=model_name,
                split_label=fold.split_label,
            )
            if (
                self.resume_plan is not None
                and not self.resume_plan.should_execute(leaf_run)
            ):
                self.skipped_leaf_keys.add(leaf_run.key)
                continue
            records.append(
                self.run_single_fold(
                    dataset=dataset,
                    strategy_name=strategy_name,
                    partitioner_config=partitioner_config,
                    model_name=model_name,
                    model_factory=model_factory,
                    fold=fold,
                )
            )
        return records

    def _leaf_run_key(
        self,
        *,
        dataset: RawDatasetBundle,
        strategy_name: str,
        partitioner_config: Mapping[str, Any],
        model_name: str,
        split_label: str,
    ) -> LeafRunKey:
        return leaf_run_key_from_components(
            dataset=dataset.name,
            split=split_label,
            model=model_name,
            strategy=strategy_name,
            strategy_config=partitioner_config,
            seed=self.seed,
        )

    def split_count(self, dataset: RawDatasetBundle) -> int:
        return 1 if isinstance(dataset, OpenMLRawDatasetBundle) else self.cv_folds

    def iter_folds(
        self,
        dataset: RawDatasetBundle,
        openml_split_data: Optional[tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, list[str], list[str], list[str]]] = None,
    ) -> Iterable[FoldSplit]:
        if isinstance(dataset, OpenMLRawDatasetBundle):
            yield self._openml_fold(dataset, openml_split_data)
            return

        X = dataset.X.to_numpy() if isinstance(dataset.X, pd.DataFrame) else np.asarray(dataset.X)
        splitter = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.seed)
        for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X), start=1):
            yield self._local_cv_fold(dataset, fold_idx, train_idx, test_idx)

    def run_single_fold(
        self,
        dataset: RawDatasetBundle,
        strategy_name: str,
        partitioner_config: Mapping[str, Any],
        model_name: str,
        model_factory: Callable[[], Any],
        fold: FoldSplit,
    ) -> dict[str, Any]:
        plan = self._build_execution_plan(partitioner_config, train_size=len(fold.y_train))
        try:
            with progress_bar(
                enabled=self.show_progress,
                desc=f"Fold pipeline ({dataset.name}/{strategy_name}/{model_name}/{fold.split_label})",
                total=5,
            ) as fold_stage:
                fold = self._ensure_validation_split(dataset, partitioner_config, fold)
                fold_stage.update(1)

                X_train_df, X_val_df, X_test_df = self._ensure_fold_dataframes(dataset, fold)
                fold_stage.update(1)

                if plan.use_direct_model:
                    return self._run_direct_model_fold(
                        dataset=dataset,
                        strategy_name=strategy_name,
                        partitioner_config=partitioner_config,
                        model_name=model_name,
                        model_factory=model_factory,
                        fold=fold,
                        plan=plan,
                        X_train_df=X_train_df,
                        X_test_df=X_test_df,
                        fold_stage=fold_stage,
                    )

                return self._run_ensemble_fold(
                    dataset=dataset,
                    strategy_name=strategy_name,
                    partitioner_config=partitioner_config,
                    model_name=model_name,
                    model_factory=model_factory,
                    fold=fold,
                    plan=plan,
                    X_train_df=X_train_df,
                    X_val_df=X_val_df,
                    X_test_df=X_test_df,
                    fold_stage=fold_stage,
                )
        except Exception as ex:
            return self._log_failed_fold(
                dataset=dataset,
                strategy_name=strategy_name,
                partitioner_config=partitioner_config,
                model_name=model_name,
                fold=fold,
                plan=plan,
                error=ex,
            )

    def _openml_fold(
        self,
        dataset: OpenMLRawDatasetBundle,
        openml_split_data: Optional[tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, list[str], list[str], list[str]]],
    ) -> FoldSplit:
        if openml_split_data is None:
            openml_split_data = dataset.load_split_data(show_progress=self.show_progress)
        X_train_full, y_train_full, X_test, y_test, _, _, _ = openml_split_data
        X_train, X_val, y_train, y_val = self._prepare_train_val_for_execution(
            X_train_full=X_train_full,
            y_train_full=y_train_full,
            problem_type=dataset.problem_type,
            random_state=self.seed + 1,
        )
        return FoldSplit(1, "split_1", X_train, X_val, X_test, y_train, y_val, y_test)

    def _local_cv_fold(
        self,
        dataset: RawDatasetBundle,
        fold_idx: int,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
    ) -> FoldSplit:
        X_train_full = dataset.X.iloc[train_idx].reset_index(drop=True)
        y_train_full = dataset.y.iloc[train_idx].reset_index(drop=True)
        X_test = dataset.X.iloc[test_idx].reset_index(drop=True)
        y_test = dataset.y.iloc[test_idx].reset_index(drop=True)
        X_train, X_val, y_train, y_val = self._prepare_train_val_for_execution(
            X_train_full=X_train_full,
            y_train_full=y_train_full,
            problem_type=dataset.problem_type,
            random_state=self.seed + fold_idx,
        )
        return FoldSplit(fold_idx, f"fold_{fold_idx}", X_train, X_val, X_test, y_train, y_val, y_test)

    def _prepare_train_val_for_execution(
        self,
        X_train_full: Any,
        y_train_full: Any,
        problem_type: str,
        random_state: int,
    ) -> tuple[Any, Any, Any, Any]:
        if len(y_train_full) < 20000:
            X_train = X_train_full.reset_index(drop=True) if isinstance(X_train_full, pd.DataFrame) else X_train_full
            y_train = y_train_full.reset_index(drop=True) if isinstance(y_train_full, pd.Series) else y_train_full
            X_val = X_train.iloc[0:0].copy() if isinstance(X_train, pd.DataFrame) else X_train[:0]
            y_val = y_train.iloc[0:0].copy() if isinstance(y_train, pd.Series) else y_train[:0]
            return X_train, X_val, y_train, y_val

        X_train, _, X_val, y_train, _, y_val = self.loader.prepare_train_val_test_balanced(
            X_train_full,
            y_train_full,
            test_size=0.2,
            val_size=0,
            min_samples=20,
            problem=problem_type,
            random_state=random_state,
        )
        if len(y_val) > 12000:
            X_val = X_val.iloc[:12000] if isinstance(X_val, pd.DataFrame) else X_val[:12000]
            y_val = y_val.iloc[:12000] if isinstance(y_val, pd.Series) else y_val[:12000]
        return X_train, X_val, y_train, y_val

    def _build_execution_plan(self, partitioner_config: Mapping[str, Any], train_size: int) -> FoldExecutionPlan:
        force_chunking = bool(partitioner_config.get("force_chunking", False))
        force_direct_model = (
            bool(partitioner_config.get("force_direct_model", False))
            or partitioner_config.get("strategy") == "full_dataset"
        )
        configured_partitions = int(partitioner_config.get("n_partitions", 1))
        effective_partitions = configured_partitions if force_chunking else max(1, int(np.ceil(train_size / 20000)))
        target_chunks = 10
        chunks_percent = min(100.0, 100.0 * target_chunks / max(1, effective_partitions))
        use_direct_model = force_direct_model or (train_size < 20000 and not force_chunking)
        return FoldExecutionPlan(
            force_chunking=force_chunking,
            force_direct_model=force_direct_model,
            configured_partitions=configured_partitions,
            effective_partitions=effective_partitions,
            chunks_percent=chunks_percent,
            use_direct_model=use_direct_model,
        )

    def _ensure_validation_split(
        self,
        dataset: RawDatasetBundle,
        partitioner_config: Mapping[str, Any],
        fold: FoldSplit,
    ) -> FoldSplit:
        if not bool(partitioner_config.get("force_chunking", False)) or len(fold.y_val) > 0:
            return fold
        X_train, X_val, y_train, y_val = self._split_small_train_val(
            X_train=fold.X_train,
            y_train=fold.y_train,
            problem_type=dataset.problem_type,
            random_state=self.seed + fold.fold_idx,
        )
        return FoldSplit(
            fold_idx=fold.fold_idx,
            split_label=fold.split_label,
            X_train=X_train,
            X_val=X_val,
            X_test=fold.X_test,
            y_train=y_train,
            y_val=y_val,
            y_test=fold.y_test,
        )

    @staticmethod
    def _split_small_train_val(
        X_train: Any,
        y_train: Any,
        problem_type: str,
        random_state: int,
    ) -> tuple[Any, Any, Any, Any]:
        if len(y_train) < 4:
            X_val = X_train.iloc[0:0].copy() if isinstance(X_train, pd.DataFrame) else X_train[:0]
            y_val = y_train.iloc[0:0].copy() if isinstance(y_train, pd.Series) else y_train[:0]
            return X_train, X_val, y_train, y_val

        y_array = np.asarray(y_train)
        stratify = None
        if problem_type == "classification":
            _, counts = np.unique(y_array, return_counts=True)
            if counts.size > 1 and np.min(counts) >= 2:
                stratify = y_array

        train_idx, val_idx = train_test_split(
            np.arange(len(y_array)),
            test_size=0.25,
            random_state=random_state,
            stratify=stratify,
        )

        def _take(value: Any, indices: np.ndarray) -> Any:
            if isinstance(value, (pd.DataFrame, pd.Series)):
                return value.iloc[indices].reset_index(drop=True)
            return value[indices]

        return (
            _take(X_train, train_idx),
            _take(X_train, val_idx),
            _take(y_train, train_idx),
            _take(y_train, val_idx),
        )

    def _ensure_fold_dataframes(self, dataset: RawDatasetBundle, fold: FoldSplit) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        return (
            self._ensure_dataframe(fold.X_train, dataset),
            self._ensure_dataframe(fold.X_val, dataset),
            self._ensure_dataframe(fold.X_test, dataset),
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

    def _run_direct_model_fold(
        self,
        dataset: RawDatasetBundle,
        strategy_name: str,
        partitioner_config: Mapping[str, Any],
        model_name: str,
        model_factory: Callable[[], Any],
        fold: FoldSplit,
        plan: FoldExecutionPlan,
        X_train_df: pd.DataFrame,
        X_test_df: pd.DataFrame,
        fold_stage: Any,
    ) -> dict[str, Any]:
        fit_started = perf_counter()
        model = model_factory()
        model.fit(X_train_df, fold.y_train)
        fit_time = perf_counter() - fit_started
        fold_stage.update(1)

        infer_started = perf_counter()
        direct_classes = (
            np.unique(np.asarray(fold.y_train))
            if dataset.problem_type == "classification"
            else None
        )
        predictions, y_proba = self._predict_direct_model(
            model,
            X_test_df,
            dataset.problem_type,
            classes=direct_classes,
        )
        infer_time = perf_counter() - infer_started
        fold_stage.update(1)

        model_metrics = calculate_metrics(
            y_true=fold.y_test,
            y_labels=predictions,
            y_proba=y_proba if dataset.problem_type == "classification" else None,
            problem_type=dataset.problem_type,
            classes=(
                direct_classes
                if dataset.problem_type == "classification"
                else None
            ),
        )
        sample_stats = self._build_train_sample_stats(
            y_train=fold.y_train,
            problem_type=dataset.problem_type,
            total_train_size=len(fold.y_train),
        )
        sample_stats["chunk_count"] = 1
        sample_stats["chunk_size_mean"] = float(len(fold.y_train))
        sample_stats["model_fit_rows_total"] = int(len(fold.y_train))
        sample_stats["active_model_count"] = 1
        model_complexity_diagnostics = summarize_ensemble_complexity(
            ({"name": "full_dataset", "model": model},),
            X_test_df,
        )

        payload = self._log_direct_model_fold(
            dataset=dataset,
            strategy_name=strategy_name,
            partitioner_config=partitioner_config,
            model_name=model_name,
            fold=fold,
            model_metrics=model_metrics,
            fit_time=fit_time,
            infer_time=infer_time,
            sample_stats=sample_stats,
            model_complexity_diagnostics=model_complexity_diagnostics,
        )
        fold_stage.update(1)
        return payload

    @staticmethod
    def _predict_direct_model(
        model: Any,
        X_test_df: pd.DataFrame,
        problem_type: str,
        *,
        classes: Optional[np.ndarray] = None,
    ) -> tuple[Any, Any]:
        if problem_type != "classification":
            return model.predict(X_test_df), None

        if not hasattr(model, "predict_proba"):
            raise ClassificationProbabilitiesRequiredError(
                scope="EnsembleFoldBenchmarkExecutor.direct_model",
                details={"model": type(model).__name__},
            )
        y_proba = np.asarray(model.predict_proba(X_test_df), dtype=float)
        model_classes = getattr(model, "classes_", None)
        global_classes = (
            np.asarray(classes)
            if classes is not None
            else np.asarray(model_classes)
        )
        if (
            y_proba.ndim == 2
            and y_proba.shape[1] > 0
            and model_classes is not None
            and len(model_classes) == y_proba.shape[1]
        ):
            aligned = np.zeros((y_proba.shape[0], global_classes.size), dtype=float)
            for model_index, label in enumerate(np.asarray(model_classes)):
                matches = np.where(global_classes == label)[0]
                if matches.size != 1:
                    raise ClassificationProbabilitiesRequiredError(
                        scope="EnsembleFoldBenchmarkExecutor.direct_model",
                        message="Direct model classes cannot be aligned.",
                        details={"model_class": str(label)},
                    )
                aligned[:, int(matches[0])] = y_proba[:, model_index]
            aligned = np.clip(aligned, 1e-15, 1.0)
            aligned /= aligned.sum(axis=1, keepdims=True)
            return global_classes[np.argmax(aligned, axis=1)], aligned
        raise ClassificationProbabilitiesRequiredError(
            scope="EnsembleFoldBenchmarkExecutor.direct_model",
            message="Direct model returned probabilities that cannot be aligned.",
            details={"model": type(model).__name__},
        )

    def _run_ensemble_fold(
        self,
        dataset: RawDatasetBundle,
        strategy_name: str,
        partitioner_config: Mapping[str, Any],
        model_name: str,
        model_factory: Callable[[], Any],
        fold: FoldSplit,
        plan: FoldExecutionPlan,
        X_train_df: pd.DataFrame,
        X_val_df: pd.DataFrame,
        X_test_df: pd.DataFrame,
        fold_stage: Any,
    ) -> dict[str, Any]:
        class_samples = None
        if dataset.problem_type == "classification":
            class_samples = self._class_representatives(X_train_df, fold.y_train, seed=self.seed + fold.fold_idx)

        tuned_partitioner_config = self._tune_partitioner_config(partitioner_config, plan)
        ensemble = SamplingEnsemble(
            problem=dataset.problem_type,
            partitioner_config=tuned_partitioner_config,
            model_factory=model_factory,
            ensemble_method=tuned_partitioner_config.get("ensemble_method", "voting"),
            show_progress=self.show_progress,
        )

        fit_started = perf_counter()
        ensemble.train_partition_models(
            X_train=X_train_df,
            y_train=fold.y_train,
            X_val=X_val_df,
            y_val=fold.y_val,
            class_samples=class_samples,
            cv_fold=fold.fold_idx,
            validation_metric=(
                primary_classification_metric(np.unique(np.asarray(fold.y_train)))
                if dataset.problem_type == "classification"
                else "rmse"
            ),
            train_all_chunks=True,
            save_models_to_disk=False,
        )
        fit_time = perf_counter() - fit_started
        fold_stage.update(1)

        infer_started = perf_counter()
        y_proba = (
            ensemble.ensemble_predict_proba_batch(
                X_test_df,
                batch_size=10000,
            )
            if dataset.problem_type == "classification"
            else None
        )
        predictions = (
            ensemble._labels_from_proba(y_proba)
            if y_proba is not None
            else ensemble.ensemble_predict_batch(X_test_df, batch_size=10000)
        )
        infer_time = perf_counter() - infer_started
        test_routing_diagnostics = ensemble.build_routing_diagnostics(X_test_df)
        fold_stage.update(1)

        model_metrics = calculate_metrics(
            y_true=fold.y_test,
            y_labels=predictions,
            y_proba=y_proba,
            problem_type=dataset.problem_type,
            classes=(
                ensemble._classification_classes()
                if dataset.problem_type == "classification"
                else None
            ),
        )
        sample_stats, chunk_sizes = self._build_ensemble_sample_stats(
            ensemble=ensemble,
            y_train=fold.y_train,
            problem_type=dataset.problem_type,
        )
        model_complexity_diagnostics = summarize_ensemble_complexity(
            ensemble.models,
            X_test_df,
        )
        payload = self._log_ensemble_fold(
            dataset=dataset,
            strategy_name=strategy_name,
            tuned_partitioner_config=tuned_partitioner_config,
            model_name=model_name,
            fold=fold,
            plan=plan,
            ensemble=ensemble,
            model_metrics=model_metrics,
            fit_time=fit_time,
            infer_time=infer_time,
            sample_stats=sample_stats,
            chunk_sizes=chunk_sizes,
            test_routing_diagnostics=test_routing_diagnostics,
            model_complexity_diagnostics=model_complexity_diagnostics,
        )
        fold_stage.update(1)
        return payload

    @staticmethod
    def _tune_partitioner_config(partitioner_config: Mapping[str, Any], plan: FoldExecutionPlan) -> dict[str, Any]:
        tuned_partitioner_config = dict(partitioner_config)
        tuned_partitioner_config["n_partitions"] = plan.effective_partitions
        tuned_partitioner_config["chunks_percent"] = plan.chunks_percent
        return tuned_partitioner_config

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

    @staticmethod
    def _build_train_sample_stats(
        y_train: Any,
        problem_type: str,
        total_train_size: int,
    ) -> Dict[str, Any]:
        y_array = np.asarray(y_train)
        if problem_type == "classification":
            return build_sample_stats(
                y_sampled=y_array,
                total_train_size=total_train_size,
            )

        y_numeric = pd.to_numeric(pd.Series(y_array), errors="coerce").to_numpy()
        finite_mask = np.isfinite(y_numeric)
        y_finite = y_numeric[finite_mask]
        stats: Dict[str, Any] = {
            "sample_size": int(y_array.shape[0]),
            "coverage_ratio": float(y_array.shape[0] / max(total_train_size, 1)),
        }
        if y_finite.size > 0:
            stats.update(
                {
                    "target_mean": float(np.mean(y_finite)),
                    "target_std": float(np.std(y_finite)),
                    "target_min": float(np.min(y_finite)),
                    "target_max": float(np.max(y_finite)),
                }
            )
        return stats

    def _build_ensemble_sample_stats(
        self,
        ensemble: SamplingEnsemble,
        y_train: Any,
        problem_type: str,
    ) -> tuple[dict[str, Any], list[int]]:
        chunk_sizes = [
            int(model_info.get("data_size", 0))
            for model_info in ensemble.models
        ]
        sample_stats = self._build_train_sample_stats(
            y_train=y_train,
            problem_type=problem_type,
            total_train_size=len(y_train),
        )
        accounting = _build_ensemble_sample_accounting(ensemble)
        sample_stats.update(
            {
                "sample_size": accounting.selected_rows,
                "coverage_ratio": float(
                    accounting.selected_rows / max(len(y_train), 1)
                ),
                "selected_rows": accounting.selected_rows,
                "selected_partition_count": (
                    accounting.selected_partition_count
                ),
                "model_fit_rows_total": accounting.model_fit_rows_total,
                "class_coverage_rows_added": (
                    accounting.class_coverage_rows_added
                ),
                "active_model_rows": accounting.active_model_rows,
                "active_model_count": accounting.active_model_count,
                "active_model_coverage_ratio": float(
                    accounting.active_model_rows / max(len(y_train), 1)
                ),
                "chunk_count": accounting.selected_partition_count,
                "chunk_size_mean": float(
                    accounting.selected_rows
                    / max(accounting.selected_partition_count, 1)
                ),
            }
        )
        if problem_type == "classification":
            source_distribution = dict(
                sample_stats.get("class_distribution", {})
            )
            selected_distribution = _selected_class_distribution(ensemble)
            sample_stats["source_class_distribution"] = source_distribution
            sample_stats["source_class_coverage_count"] = len(
                source_distribution
            )
            if selected_distribution:
                sample_stats["class_distribution"] = selected_distribution
                sample_stats["class_coverage_count"] = len(
                    selected_distribution
                )
        return sample_stats, chunk_sizes

    def _log_direct_model_fold(
        self,
        dataset: RawDatasetBundle,
        strategy_name: str,
        partitioner_config: Mapping[str, Any],
        model_name: str,
        fold: FoldSplit,
        model_metrics: Mapping[str, Any],
        fit_time: float,
        infer_time: float,
        sample_stats: Mapping[str, Any],
        model_complexity_diagnostics: Mapping[str, Any],
    ) -> dict[str, Any]:
        fold_value = self._fold_value(fold)
        return self.logger.log_strategy_run(
            dataset_name=dataset.name,
            strategy_name=f"{strategy_name}__{model_name}__{fold.split_label}",
            strategy_params={
                **dict(partitioner_config),
                "model": model_name,
                "cv_fold": fold_value,
                "split_label": fold.split_label,
                "chunking_skipped": True,
            },
            model_metrics=dict(model_metrics),
            timings={"fit": fit_time, "sample": 0.0, "inference": infer_time},
            sample_stats=dict(sample_stats),
            extra={
                **self._base_fold_extra(
                    dataset,
                    strategy_name,
                    model_name,
                    fold,
                    partitioner_config,
                ),
                "contracts": self._build_fold_contract_payload(
                    dataset=dataset,
                    fold=fold,
                    model_metrics=model_metrics,
                    timings={"fit": fit_time, "sample": 0.0, "inference": infer_time},
                    sample_stats=sample_stats,
                ),
                "effective_partitions": 1,
                "chunks_percent": 100.0,
                "n_chunks": 1,
                "chunk_sizes": [int(len(fold.y_train))],
                "partition_metrics": [],
                "chunking_skipped": True,
                "chunking_skip_reason": "train_size_below_20000",
                "model_complexity_diagnostics": dict(
                    model_complexity_diagnostics
                ),
            },
        )

    def _log_ensemble_fold(
        self,
        dataset: RawDatasetBundle,
        strategy_name: str,
        tuned_partitioner_config: Mapping[str, Any],
        model_name: str,
        fold: FoldSplit,
        plan: FoldExecutionPlan,
        ensemble: SamplingEnsemble,
        model_metrics: Mapping[str, Any],
        fit_time: float,
        infer_time: float,
        sample_stats: Mapping[str, Any],
        chunk_sizes: Sequence[int],
        test_routing_diagnostics: Mapping[str, Any],
        model_complexity_diagnostics: Mapping[str, Any],
    ) -> dict[str, Any]:
        fold_value = self._fold_value(fold)
        return self.logger.log_strategy_run(
            dataset_name=dataset.name,
            strategy_name=f"{strategy_name}__{model_name}__{fold.split_label}",
            strategy_params={
                **dict(tuned_partitioner_config),
                "model": model_name,
                "cv_fold": fold_value,
                "split_label": fold.split_label,
            },
            model_metrics=dict(model_metrics),
            timings={"fit": fit_time, "sample": 0.0, "inference": infer_time},
            sample_stats=dict(sample_stats),
            extra={
                **self._base_fold_extra(
                    dataset,
                    strategy_name,
                    model_name,
                    fold,
                    tuned_partitioner_config,
                ),
                "contracts": self._build_fold_contract_payload(
                    dataset=dataset,
                    fold=fold,
                    model_metrics=model_metrics,
                    timings={"fit": fit_time, "sample": 0.0, "inference": infer_time},
                    sample_stats=sample_stats,
                ),
                "effective_partitions": plan.effective_partitions,
                "chunks_percent": plan.chunks_percent,
                "n_chunks": len(ensemble.models),
                "chunk_sizes": list(chunk_sizes),
                "partition_metrics": ensemble.partition_metrics,
                "partition_diagnostics": getattr(ensemble, "partition_diagnostics_", {}),
                "validation_diagnostics": getattr(ensemble, "validation_diagnostics_", {}),
                "test_routing_diagnostics": dict(test_routing_diagnostics),
                "sampler_diagnostics": getattr(ensemble.partitioner, "diagnostics_", {}),
                "budget_policy": getattr(ensemble, "budget_policy_", {}),
                "runtime_diagnostics": getattr(ensemble, "runtime_diagnostics_", {}),
                "model_complexity_diagnostics": dict(
                    model_complexity_diagnostics
                ),
                "runtime_contract": (
                    ensemble.runtime_contract_.to_dict()
                    if getattr(ensemble, "runtime_contract_", None) is not None
                    else {}
                ),
                "partition_size_contract": (
                    ensemble.partition_size_diagnostics_contract_.to_dict()
                    if getattr(
                        ensemble,
                        "partition_size_diagnostics_contract_",
                        None,
                    ) is not None
                    else {}
                ),
            },
        )

    def _log_failed_fold(
        self,
        dataset: RawDatasetBundle,
        strategy_name: str,
        partitioner_config: Mapping[str, Any],
        model_name: str,
        fold: FoldSplit,
        plan: FoldExecutionPlan,
        error: Exception,
    ) -> dict[str, Any]:
        fold_value = self._fold_value(fold)
        return self.logger.log_strategy_run(
            dataset_name=dataset.name,
            strategy_name=f"{strategy_name}__{model_name}__{fold.split_label}",
            strategy_params={
                **dict(partitioner_config),
                "model": model_name,
                "cv_fold": fold_value,
                "split_label": fold.split_label,
            },
            model_metrics={},
            timings={"fit": 0.0, "sample": 0.0, "inference": 0.0},
            sample_stats=build_sample_stats(
                y_sampled=np.array([], dtype=float),
                total_train_size=max(len(fold.y_train), 1),
            ),
            extra={
                **self._base_fold_extra(
                    dataset,
                    strategy_name,
                    model_name,
                    fold,
                    partitioner_config,
                ),
                "contracts": {
                    "dataset": dataset_to_contract(dataset).to_dict(),
                    "fold": fold_to_contract(fold).to_dict(),
                    "evaluation": evaluation_to_contract(
                        metrics={},
                        timings={"fit": 0.0, "sample": 0.0, "inference": 0.0},
                        sample_stats={},
                    ).to_dict(),
                },
                "effective_partitions": plan.effective_partitions,
                "chunks_percent": plan.chunks_percent,
                "error": str(error),
                "error_code": getattr(error, "code", type(error).__name__),
                "error_details": (
                    error.to_dict()
                    if hasattr(error, "to_dict")
                    else {}
                ),
            },
        )

    def _base_fold_extra(
        self,
        dataset: RawDatasetBundle,
        strategy_name: str,
        model_name: str,
        fold: FoldSplit,
        partitioner_config: Mapping[str, Any],
    ) -> dict[str, Any]:
        task_id = getattr(dataset, "task_id", None)
        classification_classes = (
            np.unique(np.asarray(fold.y_train))
            if dataset.problem_type == "classification"
            else np.asarray([])
        )
        leaf_run = self._leaf_run_key(
            dataset=dataset,
            strategy_name=strategy_name,
            partitioner_config=partitioner_config,
            model_name=model_name,
            split_label=fold.split_label,
        )
        return {
            "problem_type": dataset.problem_type,
            "n_classes": (
                int(classification_classes.size)
                if dataset.problem_type == "classification"
                else None
            ),
            "primary_metric": (
                primary_classification_metric(classification_classes)
                if dataset.problem_type == "classification"
                else "rmse"
            ),
            "strategy": strategy_name,
            "model": model_name,
            "cv_fold": self._fold_value(fold),
            "split_label": fold.split_label,
            "source_path": dataset.source_path,
            "n_train": int(len(fold.X_train)),
            "n_val": int(len(fold.X_val)),
            "n_test": int(len(fold.X_test)),
            "task_id": task_id,
            "task_name": getattr(dataset, "task_name", None),
            "suite_id": getattr(dataset, "suite_id", None),
            "dataset_id": getattr(dataset, "dataset_id", None),
            "openml_repeat": 0 if task_id is not None else None,
            "openml_fold": 0 if task_id is not None else None,
            "openml_sample": 0 if task_id is not None else None,
            "leaf_run_key": leaf_run.key,
            "leaf_run": leaf_run.to_dict(),
        }

    @staticmethod
    def _fold_value(fold: FoldSplit) -> Optional[int]:
        return fold.fold_idx if fold.split_label.startswith("fold_") else None

    @staticmethod
    def _build_fold_contract_payload(
        dataset: RawDatasetBundle,
        fold: FoldSplit,
        model_metrics: Mapping[str, Any],
        timings: Mapping[str, float],
        sample_stats: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "dataset": dataset_to_contract(dataset).to_dict(),
            "fold": fold_to_contract(fold).to_dict(),
            "evaluation": evaluation_to_contract(
                metrics=model_metrics,
                timings=timings,
                sample_stats=sample_stats,
            ).to_dict(),
        }


class EnsembleChunkBenchmarkRunner:
    """Runner for chunk-based SamplingEnsemble benchmarks on raw AMLB datasets."""

    def __init__(
        self,
        logger: Optional[BenchmarkLogger] = None,
        cv_folds: int = 3,
        seed: int = 42,
        show_progress: bool = True,
        on_record: Optional[Callable[[Dict[str, Any]], None]] = None,
        resume_plan: ResumePlan | None = None,
    ) -> None:
        self.logger = logger or BenchmarkLogger()
        self.cv_folds = cv_folds
        self.seed = seed
        self.show_progress = show_progress
        self.loader = AMLBDatasetLoader()
        self.on_record = on_record
        self.resume_plan = resume_plan
        self.fold_executor = EnsembleFoldBenchmarkExecutor(
            logger=self.logger,
            loader=self.loader,
            cv_folds=self.cv_folds,
            seed=self.seed,
            show_progress=self.show_progress,
            resume_plan=self.resume_plan,
        )

    def run_dataset(
        self,
        dataset: RawDatasetBundle,
        strategy_configs: StrategyGridContract | Mapping[str, Mapping[str, Any]],
        model_pool: Mapping[str, Callable[[], Any]],
    ) -> List[Dict[str, Any]]:
        strategy_grid = (
            strategy_configs
            if isinstance(strategy_configs, StrategyGridContract)
            else normalize_strategy_grid(strategy_configs)
        )
        if not self._has_pending_leaf_runs(
            dataset,
            strategy_grid,
            model_pool,
        ):
            return []
        openml_split_data = self._load_openml_split(dataset)
        try:
            return self._run_model_strategy_grid(
                dataset=dataset,
                strategy_configs=strategy_grid,
                model_pool=model_pool,
                openml_split_data=openml_split_data,
            )
        finally:
            self._release_openml_split(openml_split_data)

    def run_scenario_grid(
        self,
        dataset: RawDatasetBundle,
        scenario_grid: ModelStrategyScenarioGridContract,
        model_pool: Mapping[str, Callable[[], Any]],
    ) -> List[Dict[str, Any]]:
        """Run explicitly bound model-strategy scenarios for one dataset."""
        self._validate_scenario_models(scenario_grid, model_pool)
        if not self._scenario_grid_has_pending_leaf_runs(
            dataset,
            scenario_grid,
        ):
            return []

        openml_split_data = self._load_openml_split(dataset)
        records: list[dict[str, Any]] = []
        try:
            scenarios = tqdm(
                scenario_grid.scenarios,
                total=len(scenario_grid.scenarios),
                disable=not self.show_progress,
                desc=f"Scenarios ({dataset.name})",
                leave=False,
            )
            for scenario in scenarios:
                fold_records = self.fold_executor.run_strategy_folds(
                    dataset=dataset,
                    strategy_name=scenario.name,
                    partitioner_config=scenario.strategy.materialize(),
                    model_name=scenario.model_name,
                    model_factory=model_pool[scenario.model_name],
                    openml_split_data=openml_split_data,
                )
                for record in fold_records:
                    self._record_run(records, record)
            return records
        finally:
            self._release_openml_split(openml_split_data)

    @staticmethod
    def _validate_scenario_models(
        scenario_grid: ModelStrategyScenarioGridContract,
        model_pool: Mapping[str, Callable[[], Any]],
    ) -> None:
        missing = sorted(set(scenario_grid.model_names) - set(model_pool))
        if missing:
            raise InvalidExperimentConfigError(
                scope="benchmark.scenario_grid",
                code="missing_scenario_models",
                message=(
                    "Scenario grid references models absent from model_pool."
                ),
                details={"missing_models": missing},
            )

    def _scenario_grid_has_pending_leaf_runs(
        self,
        dataset: RawDatasetBundle,
        scenario_grid: ModelStrategyScenarioGridContract,
    ) -> bool:
        if self.resume_plan is None:
            return True
        pending = False
        split_labels = (
            ("split_1",)
            if isinstance(dataset, OpenMLRawDatasetBundle)
            else tuple(
                f"fold_{fold_idx}"
                for fold_idx in range(1, self.cv_folds + 1)
            )
        )
        for scenario in scenario_grid.scenarios:
            config = scenario.strategy.materialize()
            for split_label in split_labels:
                leaf_run = leaf_run_key_from_components(
                    dataset=dataset.name,
                    split=split_label,
                    model=scenario.model_name,
                    strategy=scenario.name,
                    strategy_config=config,
                    seed=self.seed,
                )
                if self.resume_plan.should_execute(leaf_run):
                    pending = True
                else:
                    self.fold_executor.skipped_leaf_keys.add(leaf_run.key)
        return pending

    def _has_pending_leaf_runs(
        self,
        dataset: RawDatasetBundle,
        strategy_grid: StrategyGridContract,
        model_pool: Mapping[str, Callable[[], Any]],
    ) -> bool:
        if self.resume_plan is None:
            return True
        pending = False
        materialized = materialize_strategy_grid(strategy_grid)
        split_labels = (
            ("split_1",)
            if isinstance(dataset, OpenMLRawDatasetBundle)
            else tuple(
                f"fold_{fold_idx}"
                for fold_idx in range(1, self.cv_folds + 1)
            )
        )
        for model_name in model_pool:
            for strategy_name, partitioner_config in materialized.items():
                for split_label in split_labels:
                    leaf_run = leaf_run_key_from_components(
                        dataset=dataset.name,
                        split=split_label,
                        model=model_name,
                        strategy=strategy_name,
                        strategy_config=partitioner_config,
                        seed=self.seed,
                    )
                    if self.resume_plan.should_execute(leaf_run):
                        pending = True
                    else:
                        self.fold_executor.skipped_leaf_keys.add(
                            leaf_run.key
                        )
        return pending

    def resume_diagnostics(self) -> dict[str, Any]:
        return {
            "enabled": self.resume_plan is not None,
            "skipped_leaf_runs": len(
                self.fold_executor.skipped_leaf_keys
            ),
        }

    def _load_openml_split(
        self,
        dataset: RawDatasetBundle,
    ) -> Optional[tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, list[str], list[str], list[str]]]:
        if not isinstance(dataset, OpenMLRawDatasetBundle):
            return None
        with progress_bar(
            enabled=self.show_progress,
            desc=f"Load OpenML split ({dataset.name})",
            total=1,
        ) as stage:
            openml_split_data = dataset.load_split_data(show_progress=self.show_progress)
            stage.update(1)
        return openml_split_data

    def _run_model_strategy_grid(
        self,
        dataset: RawDatasetBundle,
        strategy_configs: StrategyGridContract | Mapping[str, Mapping[str, Any]],
        model_pool: Mapping[str, Callable[[], Any]],
        openml_split_data: Optional[tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, list[str], list[str], list[str]]],
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for model_name, model_factory in self._iter_models(dataset, model_pool):
            self._run_strategies_for_model(
                records=records,
                dataset=dataset,
                strategy_configs=strategy_configs,
                model_name=model_name,
                model_factory=model_factory,
                openml_split_data=openml_split_data,
            )
        return records

    def _iter_models(
        self,
        dataset: RawDatasetBundle,
        model_pool: Mapping[str, Callable[[], Any]],
    ) -> Iterable[tuple[str, Callable[[], Any]]]:
        return tqdm(
            model_pool.items(),
            total=len(model_pool),
            disable=not self.show_progress,
            desc=f"Models ({dataset.name})",
            leave=False,
        )

    def _run_strategies_for_model(
        self,
        records: list[dict[str, Any]],
        dataset: RawDatasetBundle,
        strategy_configs: StrategyGridContract | Mapping[str, Mapping[str, Any]],
        model_name: str,
        model_factory: Callable[[], Any],
        openml_split_data: Optional[tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, list[str], list[str], list[str]]],
    ) -> None:
        for strategy_name, partitioner_config in self._iter_strategies(dataset, model_name, strategy_configs):
            fold_records = self.fold_executor.run_strategy_folds(
                dataset=dataset,
                strategy_name=strategy_name,
                partitioner_config=partitioner_config,
                model_name=model_name,
                model_factory=model_factory,
                openml_split_data=openml_split_data,
            )
            for record in fold_records:
                self._record_run(records, record)

    def _iter_strategies(
        self,
        dataset: RawDatasetBundle,
        model_name: str,
        strategy_configs: StrategyGridContract | Mapping[str, Mapping[str, Any]],
    ) -> Iterable[tuple[str, Mapping[str, Any]]]:
        materialized = materialize_strategy_grid(strategy_configs)
        return tqdm(
            materialized.items(),
            total=len(materialized),
            disable=not self.show_progress,
            desc=f"Strategies ({dataset.name}/{model_name})",
            leave=False,
        )

    def _record_run(self, records: list[dict[str, Any]], record: dict[str, Any]) -> None:
        if self.on_record is not None:
            self.on_record(record)
        records.append(record)

    @staticmethod
    def _release_openml_split(openml_split_data: Optional[Any]) -> None:
        if openml_split_data is not None:
            del openml_split_data
            gc.collect()

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
