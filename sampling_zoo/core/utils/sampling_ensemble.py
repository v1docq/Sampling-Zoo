# model_integration.py
import pickle
import os
from scipy.stats import mode
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Callable
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sampling_zoo.core.api.api_main import SamplingStrategyFactory
from sampling_zoo.core.metrics.eval_metrics import calculate_metrics, get_metric_comparator
from sampling_zoo.core.experiment.contracts import PartitionTrainingRequest, PartitionTrainingResult
from sampling_zoo.core.experiment.morphisms import (
    chunk_models_to_contracts,
    partitions_to_contract,
    routing_to_contract,
)
from sampling_zoo.core.utils.ensemble_routing import RoutedWeightedRouter
from sampling_zoo.core.utils.progress import progress_bar, progress_iter, progress_write
from sampling_zoo.core.utils.routed_em_refiner import RoutedEMModelRefiner

try:
    from lightgbm import LGBMRegressor, LGBMClassifier
except Exception:  # pragma: no cover - optional dependency
    LGBMRegressor = None
    LGBMClassifier = None


class SamplingEnsemble:
    """
    Интеграция Sampling-Zoo с ML моделями для работы с большими датасетами
    через интеллектуальное семплирование и ансамблирование
    """

    def __init__(self,
                 problem: str,
                 partitioner_config: Dict[str, Any] = None,
                 model_class: Optional[Callable] = None,
                 model_params: Dict[str, Any] = None,
                 model_factory: Optional[Callable[[], Any]] = None,
                 ensemble_method: str = 'voting',
                 show_progress: bool = True):

        self.problem = problem
        self.partitioner_config = partitioner_config or {
            'strategy': 'feature_clustering',
            'n_clusters': 3,
            'method': 'kmeans'
        }

        # Автоматический выбор модели если не указана
        if model_factory is None and model_class is None:
            if problem == 'classification':
                model_class = LGBMClassifier or RandomForestClassifier
            elif problem == 'regression':
                model_class = LGBMRegressor or RandomForestRegressor
            else:
                raise ValueError("Problem type must be 'classification' or 'regression'")

        self.model_class = model_class
        self.model_params = model_params or {}
        self.model_factory = model_factory
        self.ensemble_method = ensemble_method
        self.show_progress = show_progress
        self.router = RoutedWeightedRouter(
            problem=self.problem,
            config=self.partitioner_config,
            show_progress=self.show_progress,
        )
        self.router_mode = self.router.router_mode
        self.bs_size = 1000
        self.partitions = None
        self.partitioner = None
        self.models = []
        self.partition_metrics = {}
        self.partition_diagnostics_ = {}
        self.validation_diagnostics_ = {}
        self.test_routing_diagnostics_ = {}
        self.partition_training_contract_ = None

    def _log(self, message: str) -> None:
        progress_write(message, enabled=self.show_progress)

    def prepare_data_partitions(self,
                                features: pd.DataFrame,
                                target: pd.Series,
                                random_state: int = 42) -> Dict[str, pd.DataFrame]:
        """
        Build data partitions with the configured Sampling-Zoo strategy.
        """
        try:
            strategy_name = self._strategy_name()
            with progress_bar(
                enabled=self.show_progress,
                desc=f"Prepare chunks ({strategy_name})",
                total=4,
            ) as stage:
                strategy_kwargs = self._build_partitioner_kwargs(strategy_name, random_state)
                stage.update(1)

                self.partitioner = self._create_partitioner(strategy_name, strategy_kwargs)
                stage.update(1)

                self.partitions = self._fit_and_collect_partitions(self.partitioner, strategy_name, features, target)
                stage.update(1)

                self.partitions = self._apply_budget_policy_to_partitions(
                    partitions=self.partitions,
                    total_rows=len(features),
                    random_state=random_state,
                )
                stage.update(1)

            self.partition_diagnostics_ = self._build_partition_target_diagnostics(self.partitions, target)
            self._log_partition_summary(self.partitions)
            return self.partitions

        except ImportError as exc:
            raise ImportError(
                "Sampling-Zoo is not installed. Install it from https://github.com/v1docq/Sampling-Zoo"
            ) from exc

    def _strategy_name(self) -> str:
        return self.partitioner_config['strategy']

    @staticmethod
    def _reserved_partitioner_config_keys() -> set:
        return {
            'strategy',
            'model',
            'problem',
            'ensemble_method',
            'load_filename',
            'save_filename',
            'budget_ratio',
            'experiment_chunk_fraction',
            'force_chunking',
            'force_direct_model',
            'show_progress',
            'router',
            'router_n_estimators',
            'router_max_depth',
            'gating_hidden_dim',
            'gating_epochs',
            'gating_lr',
            'gating_kl_weight',
            'gating_balance_weight',
            'gating_weight_decay',
            'gating_batch_size',
            'gating_device',
            'routing_refinement',
            'em_max_iterations',
            'em_min_improvement',
            'em_assignment_policy',
            'em_min_partition_size',
            'em_refit_router',
            'em_keep_best',
        }

    @staticmethod
    def _normalize_router_mode(router_mode: Any) -> str:
        normalized = str(router_mode or 'spectral').strip().lower()
        if normalized not in {'spectral', 'learned_head', 'constrained_gating'}:
            raise ValueError("router must be one of: spectral, learned_head, constrained_gating")
        return normalized

    def _build_partitioner_kwargs(self, strategy_name: str, random_state: int) -> Dict[str, Any]:
        strategy_kwargs = {
            key: value
            for key, value in self.partitioner_config.items()
            if key not in self._reserved_partitioner_config_keys()
        }
        strategy_kwargs.setdefault('n_partitions', self.partitioner_config.get('n_partitions', 5))
        strategy_kwargs.setdefault('random_state', random_state)
        strategy_kwargs = self._filter_partitioner_kwargs(strategy_name, strategy_kwargs)
        return self._with_supervised_partitioner_kwargs(strategy_name, strategy_kwargs, random_state)

    def _filter_partitioner_kwargs(self, strategy_name: str, strategy_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        allowed_by_strategy = {
            'feature_clustering': {'n_partitions', 'method', 'feature_engineering', 'random_state'},
            'random': {'n_partitions', 'random_state', 'chunks_percent'},
        }
        allowed_keys = allowed_by_strategy.get(strategy_name)
        if allowed_keys is not None:
            return {key: value for key, value in strategy_kwargs.items() if key in allowed_keys}
        if strategy_name == 'rmt_contraction':
            strategy_kwargs.setdefault('show_progress', self.show_progress)
        return strategy_kwargs

    def _with_supervised_partitioner_kwargs(
        self,
        strategy_name: str,
        strategy_kwargs: Dict[str, Any],
        random_state: int,
    ) -> Dict[str, Any]:
        if strategy_name not in ['difficulty', 'uncertainty']:
            return strategy_kwargs
        strategy_kwargs.update({
            'problem': self.problem,
            'model': self._create_support_model(random_state),
            'chunks_percent': self.partitioner_config.get('chunks_percent', 100),
        })
        return strategy_kwargs

    def _create_support_model(self, random_state: int) -> Callable:
        if self.problem == 'classification':
            return (
                LGBMClassifier(n_estimators=50, n_jobs=-1, verbosity=-1) if LGBMClassifier is not None
                else RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=random_state)
            )
        return (
            LGBMRegressor(n_estimators=50, n_jobs=-1, verbosity=-1) if LGBMRegressor is not None
            else RandomForestRegressor(n_estimators=50, n_jobs=-1, random_state=random_state)
        )

    @staticmethod
    def _create_partitioner(strategy_name: str, strategy_kwargs: Dict[str, Any]) -> Any:
        return SamplingStrategyFactory().create_strategy(
            strategy_type=strategy_name,
            **strategy_kwargs,
        )

    def _fit_and_collect_partitions(
        self,
        partitioner: Any,
        strategy_name: str,
        features: pd.DataFrame,
        target: pd.Series,
    ) -> Dict[str, Any]:
        if strategy_name in ['difficulty', 'uncertainty']:
            return self._fit_supervised_partitioner(partitioner, features, target)
        if strategy_name.__contains__('stratified'):
            return self._fit_stratified_partitioner(partitioner, features, target)
        return self._fit_standard_partitioner(partitioner, features, target)

    @staticmethod
    def _fit_supervised_partitioner(partitioner: Any, features: pd.DataFrame, target: pd.Series) -> Dict[str, Any]:
        partitioner.fit(features, target=target)
        return partitioner.get_partitions(features, target)

    @staticmethod
    def _fit_standard_partitioner(partitioner: Any, features: pd.DataFrame, target: pd.Series) -> Dict[str, Any]:
        partitioner.fit(features)
        return partitioner.get_partitions(features, target)

    @staticmethod
    def _fit_stratified_partitioner(partitioner: Any, features: pd.DataFrame, target: pd.Series) -> Dict[str, Any]:
        features['target'] = target
        partitioner.fit(data=features, target=features.columns.to_list(), data_target=features['target'])
        partitions = partitioner.get_partitions(features, target=features['target'])
        for chunk in partitions:
            del partitions[chunk]['feature']['target']
        return partitions

    def _log_partition_summary(self, partitions: Dict[str, Any]) -> None:
        self._log(f"Created {len(partitions)} data partitions:")
        sample_size = self._first_partition_size(partitions)
        if sample_size is not None:
            self._log(f"Samples in first partition: {sample_size}")

    def _first_partition_size(self, partitions: Dict[str, Any]) -> Optional[int]:
        if not partitions:
            return None
        sample_key = next(iter(partitions))
        return self._partition_size(partitions[sample_key])

    @staticmethod
    def _partition_size(partition_data: Any) -> int:
        if isinstance(partition_data, dict) and 'feature' in partition_data:
            return len(partition_data['feature'])
        return len(partition_data)

    @staticmethod
    def _take_local_rows(value: Any, local_indices: np.ndarray) -> Any:
        if isinstance(value, (pd.DataFrame, pd.Series)):
            return value.iloc[local_indices].reset_index(drop=True)
        return np.asarray(value)[local_indices]

    def _slice_partition(self, partition_data: Any, local_indices: np.ndarray) -> Any:
        if isinstance(partition_data, dict):
            return {
                key: self._take_local_rows(value, local_indices)
                for key, value in partition_data.items()
            }
        return self._take_local_rows(partition_data, local_indices)

    def _apply_budget_policy_to_partitions(
        self,
        partitions: Dict[str, Any],
        total_rows: int,
        random_state: int,
    ) -> Dict[str, Any]:
        budget_ratio = self.partitioner_config.get('budget_ratio')
        if budget_ratio is None:
            self.budget_policy_ = {'applied': False}
            return partitions

        budget_ratio = float(budget_ratio)
        if not (0 < budget_ratio <= 1):
            raise ValueError("budget_ratio must be in (0, 1]")

        sizes = {name: self._partition_size(chunk) for name, chunk in partitions.items()}
        sizes = {name: size for name, size in sizes.items() if size > 0}
        if not sizes:
            self.budget_policy_ = {'applied': False, 'reason': 'empty_partitions'}
            return partitions

        budget_size = max(1, min(total_rows, int(round(total_rows * budget_ratio))))
        current_size = int(sum(sizes.values()))
        if current_size <= budget_size:
            self.budget_policy_ = {
                'applied': False,
                'budget_ratio': budget_ratio,
                'budget_size': budget_size,
                'current_size': current_size,
            }
            return partitions

        ordered_names = sorted(sizes, key=lambda name: sizes[name], reverse=True)
        if budget_size < len(ordered_names):
            ordered_names = ordered_names[:budget_size]

        ordered_total = sum(sizes[name] for name in ordered_names)
        counts = {
            name: max(1, min(sizes[name], int(np.floor(budget_size * sizes[name] / max(ordered_total, 1)))))
            for name in ordered_names
        }

        while sum(counts.values()) > budget_size:
            candidates = [name for name, count in counts.items() if count > 1]
            if not candidates:
                break
            counts[max(candidates, key=lambda item: counts[item])] -= 1

        while sum(counts.values()) < budget_size:
            candidates = [name for name in ordered_names if counts[name] < sizes[name]]
            if not candidates:
                break
            counts[max(candidates, key=lambda item: sizes[item] - counts[item])] += 1

        rng = np.random.default_rng(random_state)
        budgeted: Dict[str, Any] = {}
        for name in ordered_names:
            local_indices = np.sort(rng.choice(np.arange(sizes[name]), size=counts[name], replace=False))
            budgeted[name] = self._slice_partition(partitions[name], local_indices)

        self.budget_policy_ = {
            'applied': True,
            'budget_ratio': budget_ratio,
            'budget_size': budget_size,
            'current_size': current_size,
            'selected_size': int(sum(counts.values())),
            'partition_sizes': {name: int(count) for name, count in counts.items()},
        }
        return budgeted

    def _build_partition_target_diagnostics(self, partitions: Dict[str, Any], target: pd.Series) -> Dict[str, Any]:
        global_values = self._numeric_target_values(target)
        global_summary = self._target_summary(global_values)
        global_quantiles = global_summary.get('quantiles', {})
        chunk_diagnostics: Dict[str, Any] = {}
        sizes: List[int] = []
        mean_drifts: List[float] = []
        standardized_mean_drifts: List[float] = []
        quantile_drifts: List[float] = []

        for name, partition_data in partitions.items():
            chunk_values = self._partition_target_values(partition_data)
            chunk_summary = self._target_summary(chunk_values)
            size = int(chunk_summary.get('count', 0))
            sizes.append(size)

            drift = self._target_drift_summary(chunk_summary, global_summary, global_quantiles)
            if drift.get('mean_abs_drift') is not None:
                mean_drifts.append(float(drift['mean_abs_drift']))
            if drift.get('mean_std_units') is not None:
                standardized_mean_drifts.append(float(drift['mean_std_units']))
            if drift.get('quantile_l1_drift') is not None:
                quantile_drifts.append(float(drift['quantile_l1_drift']))

            chunk_diagnostics[name] = {
                **chunk_summary,
                **drift,
            }

        return {
            'global_target': global_summary,
            'chunk_size_imbalance': self._chunk_size_imbalance(sizes),
            'target_drift_summary': {
                'mean_abs_drift_avg': float(np.mean(mean_drifts)) if mean_drifts else None,
                'mean_abs_drift_max': float(np.max(mean_drifts)) if mean_drifts else None,
                'mean_std_units_avg': float(np.mean(standardized_mean_drifts)) if standardized_mean_drifts else None,
                'mean_std_units_max': float(np.max(standardized_mean_drifts)) if standardized_mean_drifts else None,
                'quantile_l1_drift_avg': float(np.mean(quantile_drifts)) if quantile_drifts else None,
                'quantile_l1_drift_max': float(np.max(quantile_drifts)) if quantile_drifts else None,
            },
            'chunks': chunk_diagnostics,
        }

    @staticmethod
    def _numeric_target_values(target: Any) -> np.ndarray:
        values = pd.to_numeric(pd.Series(target), errors='coerce').to_numpy(dtype=float)
        return values[np.isfinite(values)]

    @staticmethod
    def _partition_target_values(partition_data: Any) -> np.ndarray:
        if isinstance(partition_data, dict) and 'target' in partition_data:
            return SamplingEnsemble._numeric_target_values(partition_data['target'])
        return np.asarray([], dtype=float)

    @staticmethod
    def _target_summary(values: np.ndarray) -> Dict[str, Any]:
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        summary: Dict[str, Any] = {
            'count': int(values.size),
        }
        if values.size == 0:
            summary.update({
                'mean': None,
                'std': None,
                'min': None,
                'max': None,
                'quantiles': {},
            })
            return summary

        quantile_levels = (0.1, 0.25, 0.5, 0.75, 0.9)
        quantile_values = np.quantile(values, quantile_levels)
        summary.update({
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'quantiles': {
                f"q{int(level * 100):02d}": float(value)
                for level, value in zip(quantile_levels, quantile_values)
            },
        })
        return summary

    @staticmethod
    def _target_drift_summary(
        chunk_summary: Dict[str, Any],
        global_summary: Dict[str, Any],
        global_quantiles: Dict[str, float],
    ) -> Dict[str, Any]:
        chunk_mean = chunk_summary.get('mean')
        global_mean = global_summary.get('mean')
        global_std = global_summary.get('std')
        if chunk_mean is None or global_mean is None:
            return {
                'mean_abs_drift': None,
                'mean_std_units': None,
                'quantile_l1_drift': None,
            }

        mean_abs_drift = abs(float(chunk_mean) - float(global_mean))
        mean_std_units = (
            mean_abs_drift / max(float(global_std), 1e-12)
            if global_std is not None and np.isfinite(float(global_std))
            else None
        )
        chunk_quantiles = chunk_summary.get('quantiles', {}) or {}
        common_keys = [key for key in global_quantiles if key in chunk_quantiles]
        quantile_l1_drift = (
            float(np.mean([abs(float(chunk_quantiles[key]) - float(global_quantiles[key])) for key in common_keys]))
            if common_keys
            else None
        )
        return {
            'mean_abs_drift': float(mean_abs_drift),
            'mean_std_units': float(mean_std_units) if mean_std_units is not None else None,
            'quantile_l1_drift': quantile_l1_drift,
        }

    @staticmethod
    def _chunk_size_imbalance(sizes: List[int]) -> Dict[str, Any]:
        if not sizes:
            return {
                'min': 0,
                'max': 0,
                'mean': 0.0,
                'std': 0.0,
                'max_to_min_ratio': None,
                'coefficient_of_variation': None,
            }
        arr = np.asarray(sizes, dtype=float)
        mean = float(np.mean(arr))
        min_size = int(np.min(arr))
        return {
            'min': min_size,
            'max': int(np.max(arr)),
            'mean': mean,
            'std': float(np.std(arr)),
            'max_to_min_ratio': float(np.max(arr) / min_size) if min_size > 0 else None,
            'coefficient_of_variation': float(np.std(arr) / mean) if mean > 0 else None,
        }

    def _create_model_instance(self):
        """Создает экземпляр модели с заданными параметрами"""
        if self.model_factory is not None:
            return self.model_factory()
        if self.model_class is None:
            raise ValueError("Model class is not configured.")
        return self.model_class(**self.model_params)

    def _run_inference(self, fitted_model: Callable, test_data: pd.DataFrame,
                       calculation_mode: str = 'batch', batch_size: int = None):
        if calculation_mode == 'batch':
            predict_labels, predict_proba = [], []
            batch_size = batch_size if batch_size is not None else self.bs_size
            batch_data = [test_data.iloc[i:i + self.bs_size] for i in list(range(0, len(test_data), batch_size))]
            for batch in progress_iter(
                batch_data,
                enabled=self.show_progress,
                total=len(batch_data),
                desc="Model inference batches",
            ):
                if self.problem == 'regression':
                    labels = fitted_model.predict(batch)
                    predict_labels.append(labels)
                    predict_proba.append(labels)
                else:
                    proba = fitted_model.predict_proba(batch)
                    classes = getattr(fitted_model, "classes_", None)
                    if classes is not None:
                        labels = classes[np.argmax(proba, axis=1)]
                    else:
                        labels = np.argmax(proba, axis=1)
                    predict_labels.append(labels)
                    predict_proba.append(proba)
            return np.concatenate(predict_labels), np.concatenate(predict_proba)
        elif calculation_mode == 'non-batch':
            if self.problem == 'regression':
                labels = fitted_model.predict(test_data)
                proba = labels
            else:
                proba = fitted_model.predict_proba(test_data)
                classes = getattr(fitted_model, "classes_", None)
                if classes is not None:
                    labels = classes[np.argmax(proba, axis=1)]
                else:
                    labels = np.argmax(proba, axis=1)
            return labels, proba
        else:
            raise ValueError("Calculation mode must be 'batch' or 'non-batch'")

    @staticmethod
    def ensure_all_classes_in_chunk(chunk, class_representatives):
        """
        Добавляет в чанк по одному примеру отсутствующих классов.
        """
        present_classes = set(np.unique(chunk['target']))
        all_classes = set(class_representatives.keys())

        missing_classes = all_classes - present_classes

        if not missing_classes:
            return chunk

        X_extra = []
        y_extra = []

        for cls in missing_classes:
            x_rep, y_rep = class_representatives[cls]
            X_extra.append(x_rep)
            y_extra.append(y_rep)

        if isinstance(chunk['feature'], pd.DataFrame):
            extra_df = pd.DataFrame(X_extra, columns=chunk['feature'].columns)
            for column, dtype in chunk['feature'].dtypes.items():
                try:
                    extra_df[column] = extra_df[column].astype(dtype)
                except Exception:
                    continue
            chunk['feature'] = pd.concat([chunk['feature'], extra_df], ignore_index=True)
        else:
            X_extra_arr = np.stack(X_extra)
            chunk['feature'] = np.vstack([chunk['feature'], X_extra_arr])

        if isinstance(chunk['target'], pd.Series):
            extra_target = pd.Series(y_extra, name=chunk['target'].name)
            chunk['target'] = pd.concat([chunk['target'], extra_target], ignore_index=True)
        else:
            y_extra_arr = np.array(y_extra)
            chunk['target'] = np.hstack([chunk['target'], y_extra_arr])

        return chunk

    def train_partition_models(
            self,
            X_train,
            y_train,
            X_val,
            y_val,
            class_samples,
            cv_fold,
            validation_metric: str = None,
            train_all_chunks: bool = False,
            save_models_to_disk: bool = True,
    ):
        """
        Train one model per prepared data partition.
        """
        partitions = self._load_or_prepare_partitions(
            X_train=X_train,
            y_train=y_train,
            cv_fold=cv_fold,
            save_models_to_disk=save_models_to_disk,
        )
        validation_metric = self._normalize_validation_metric(validation_metric)
        metric_is_better = get_metric_comparator(validation_metric)
        training_request = self._build_partition_training_request(partitions, validation_metric)

        self._train_partition_loop(
            partitions=partitions,
            X_val=X_val,
            y_val=y_val,
            class_samples=class_samples,
            cv_fold=cv_fold,
            validation_metric=validation_metric,
            metric_is_better=metric_is_better,
            train_all_chunks=train_all_chunks,
            save_models_to_disk=save_models_to_disk,
        )
        self._finalize_partition_training(
            partitions=partitions,
            X_val=X_val,
            y_val=y_val,
            metric_is_better=metric_is_better,
            validation_metric=validation_metric,
        )
        self.partition_training_contract_ = self._build_partition_training_result(
            request=training_request,
            partitions=partitions,
        )

    def _build_partition_training_request(
        self,
        partitions: Dict[str, Any],
        validation_metric: str,
    ) -> PartitionTrainingRequest:
        return PartitionTrainingRequest(
            problem=self.problem,
            ensemble_method=self.ensemble_method,
            validation_metric=validation_metric,
            n_partitions=len(partitions),
            routing_refinement=str(self.partitioner_config.get('routing_refinement', 'none')),
        )

    def _build_partition_training_result(
        self,
        request: PartitionTrainingRequest,
        partitions: Dict[str, Any],
    ) -> PartitionTrainingResult:
        return PartitionTrainingResult(
            request=request,
            partitions=partitions_to_contract(partitions, self.partition_diagnostics_),
            chunk_models=chunk_models_to_contracts(self.models),
            routing=routing_to_contract(self.router),
            validation_diagnostics=dict(self.validation_diagnostics_),
        )

    def _load_or_prepare_partitions(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        cv_fold: int,
        save_models_to_disk: bool,
    ) -> Dict[str, Any]:
        self._ensure_dump_dir_if_needed(save_models_to_disk)
        if 'load_filename' in self.partitioner_config:
            return self._load_partitions_from_disk(cv_fold)

        partitions = self.prepare_data_partitions(X_train, y_train)
        self._save_partitions_if_requested(partitions, cv_fold)
        return partitions

    def _ensure_dump_dir_if_needed(self, save_models_to_disk: bool) -> None:
        if save_models_to_disk or 'load_filename' in self.partitioner_config or 'save_filename' in self.partitioner_config:
            os.makedirs("dumps", exist_ok=True)

    def _load_partitions_from_disk(self, cv_fold: int) -> Dict[str, Any]:
        with open(f"dumps/{self.partitioner_config['load_filename']}_{cv_fold}.pkl", "rb") as handle:
            return pickle.load(handle)

    def _save_partitions_if_requested(self, partitions: Dict[str, Any], cv_fold: int) -> None:
        if 'save_filename' not in self.partitioner_config:
            return
        with open(f"dumps/{self.partitioner_config['save_filename']}_{cv_fold}.pkl", "wb") as handle:
            pickle.dump(partitions, handle)

    def _normalize_validation_metric(self, validation_metric: Optional[str]) -> str:
        if validation_metric is None:
            return 'f1_weighted' if self.problem == 'classification' else 'rmse'
        if validation_metric == 'f1':
            return 'f1_weighted'
        return validation_metric

    def _train_partition_loop(
        self,
        partitions: Dict[str, Any],
        X_val: pd.DataFrame,
        y_val: pd.Series,
        class_samples: Any,
        cv_fold: int,
        validation_metric: str,
        metric_is_better: Callable,
        train_all_chunks: bool,
        save_models_to_disk: bool,
    ) -> None:
        validation_results = []
        best_validation_result = None
        best_validation_result_not_updated = 0

        partition_iter = progress_iter(
            partitions.items(),
            enabled=self.show_progress,
            total=len(partitions),
            desc="Train chunk models",
        )
        for partition_name, partition_data in partition_iter:
            partition_iter.set_postfix_str(str(partition_name))
            try:
                current_validation_result = self._train_partition_and_score_ensemble(
                    partition_name=partition_name,
                    partition_data=partition_data,
                    X_val=X_val,
                    y_val=y_val,
                    class_samples=class_samples,
                    cv_fold=cv_fold,
                    validation_metric=validation_metric,
                    save_models_to_disk=save_models_to_disk,
                )
                validation_results.append(current_validation_result)
                best_validation_result, best_validation_result_not_updated = self._update_best_validation_result(
                    current_validation_result=current_validation_result,
                    best_validation_result=best_validation_result,
                    best_validation_result_not_updated=best_validation_result_not_updated,
                    metric_is_better=metric_is_better,
                )
                if self._should_stop_partition_training(
                    validation_results=validation_results,
                    current_validation_result=current_validation_result,
                    best_validation_result_not_updated=best_validation_result_not_updated,
                    metric_is_better=metric_is_better,
                    train_all_chunks=train_all_chunks,
                ):
                    break
            except Exception as exc:
                self._log(f"Error while training chunk {partition_name}: {str(exc)}")
                continue

    def _train_partition_and_score_ensemble(
        self,
        partition_name: str,
        partition_data: Dict[str, Any],
        X_val: pd.DataFrame,
        y_val: pd.Series,
        class_samples: Any,
        cv_fold: int,
        validation_metric: str,
        save_models_to_disk: bool,
    ) -> float:
        model_info = self._train_single_partition_model(
            partition_name=partition_name,
            partition_data=partition_data,
            X_val=X_val,
            y_val=y_val,
            class_samples=class_samples,
            cv_fold=cv_fold,
            save_models_to_disk=save_models_to_disk,
        )
        ensemble_metrics = self._evaluate_current_ensemble(X_val, y_val)
        current_validation_result = ensemble_metrics[validation_metric]
        self._log(f"Ensemble metrics after {partition_name}: {ensemble_metrics}")
        self._log(f"Current validation metric: {current_validation_result}")
        return current_validation_result

    def _train_single_partition_model(
        self,
        partition_name: str,
        partition_data: Dict[str, Any],
        X_val: pd.DataFrame,
        y_val: pd.Series,
        class_samples: Any,
        cv_fold: int,
        save_models_to_disk: bool,
    ) -> Dict[str, Any]:
        self._log(f"Training model for chunk {partition_name}...")
        partition_data = self._ensure_partition_class_coverage(partition_data, class_samples)
        model = self._create_model_instance()
        model.fit(partition_data['feature'], partition_data['target'])
        self._save_partition_model_if_requested(model, partition_name, cv_fold, save_models_to_disk)

        predict_labels, predict_proba = self._run_inference(model, X_val, calculation_mode='non-batch')
        metrics = calculate_metrics(
            y_true=y_val,
            problem_type=self.problem,
            y_labels=predict_labels,
            y_proba=predict_proba if self.problem == "classification" else None,
        )
        model_info = self._build_partition_model_info(partition_name, model, partition_data, metrics, predict_labels)
        self._register_partition_model(partition_name, model_info, metrics)
        self._log_partition_model_result(partition_name, model_info, metrics)
        return model_info

    def _ensure_partition_class_coverage(self, partition_data: Dict[str, Any], class_samples: Any) -> Dict[str, Any]:
        if self.problem == 'classification' and class_samples:
            return self.ensure_all_classes_in_chunk(partition_data, class_samples)
        return partition_data

    @staticmethod
    def _build_partition_model_info(
        partition_name: str,
        model: Callable,
        partition_data: Dict[str, Any],
        metrics: Dict[str, Any],
        predict_labels: np.ndarray,
    ) -> Dict[str, Any]:
        return {
            'name': partition_name,
            'model': model,
            'data_size': len(partition_data['feature']),
            'metrics': metrics,
            'val_predictions': predict_labels,
        }

    def _register_partition_model(self, partition_name: str, model_info: Dict[str, Any], metrics: Dict[str, Any]) -> None:
        self.models.append(model_info)
        self.partition_metrics[partition_name] = metrics

    def _log_partition_model_result(self, partition_name: str, model_info: Dict[str, Any], metrics: Dict[str, Any]) -> None:
        self._log(f"Model {partition_name} trained. Data size: {model_info['data_size']}")
        self._log(f"Model {partition_name} metrics: {metrics}")

    @staticmethod
    def _save_partition_model_if_requested(
        model: Callable,
        partition_name: str,
        cv_fold: int,
        save_models_to_disk: bool,
    ) -> None:
        if not save_models_to_disk:
            return
        with open(f"dumps/{partition_name}_{cv_fold}.pkl", "wb") as handle:
            pickle.dump(model, handle)

    def _evaluate_current_ensemble(self, X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Any]:
        predictions = self.ensemble_predict(X_val, stage='validation')
        return calculate_metrics(
            y_true=y_val,
            y_labels=predictions,
            y_proba=None,
            problem_type=self.problem,
        )

    @staticmethod
    def _update_best_validation_result(
        current_validation_result: float,
        best_validation_result: Optional[float],
        best_validation_result_not_updated: int,
        metric_is_better: Callable,
    ) -> tuple[float, int]:
        if best_validation_result is None or metric_is_better(current_validation_result, best_validation_result):
            return current_validation_result, 0
        return best_validation_result, best_validation_result_not_updated + 1

    def _should_stop_partition_training(
        self,
        validation_results: List[float],
        current_validation_result: float,
        best_validation_result_not_updated: int,
        metric_is_better: Callable,
        train_all_chunks: bool,
    ) -> bool:
        if train_all_chunks:
            return False
        if len(validation_results) > 10 and metric_is_better(np.mean(validation_results[:-10]), current_validation_result):
            del self.models[-1]
        return best_validation_result_not_updated >= 10

    def _finalize_partition_training(
        self,
        partitions: Dict[str, Any],
        X_val: pd.DataFrame,
        y_val: pd.Series,
        metric_is_better: Callable,
        validation_metric: str,
    ) -> None:
        if not self.models:
            return

        if self._uses_moe_routing():
            self._finalize_moe_partition_training(
                partitions=partitions,
                X_val=X_val,
                y_val=y_val,
                validation_metric=validation_metric,
            )
            return

        full_metrics = self._evaluate_current_ensemble(X_val, y_val)
        self._log(f"Ensemble metrics before pruning: {full_metrics}")

        _selected, best_score = self.select_best_models_forward(
            X_val=X_val,
            y_val=y_val,
            metric_is_better=metric_is_better,
            validation_metric=validation_metric,
        )

        reduced_metrics = self._evaluate_current_ensemble(X_val, y_val)
        self._log(f"Ensemble metrics after pruning: {reduced_metrics}")
        self._log(f"Best validation metric after pruning ({validation_metric}): {best_score}")
        self.validation_diagnostics_ = self._build_validation_diagnostics(
            X_val=X_val,
            y_val=y_val,
            full_metrics=full_metrics,
            reduced_metrics=reduced_metrics,
            best_score=best_score,
            validation_metric=validation_metric,
            selection_policy='forward_pruning',
        )

    def _uses_moe_routing(self) -> bool:
        return (
            self.ensemble_method == 'routed_weighted'
            and self.router.can_route(self.partitioner)
        )

    def _finalize_moe_partition_training(
        self,
        partitions: Dict[str, Any],
        X_val: pd.DataFrame,
        y_val: pd.Series,
        validation_metric: str,
    ) -> None:
        full_metrics = self._evaluate_current_ensemble(X_val, y_val)
        self._log(f"Routed MoE metrics before local calibration: {full_metrics}")

        local_metrics = self._build_local_validation_metrics(
            X_val,
            y_val,
            list(self.models),
            use_base_router=True,
        )
        self._attach_local_validation_metrics(self.models, local_metrics)
        self.router.fit(
            X_val=X_val,
            y_val=y_val,
            active_models=list(self.models),
            partitioner=self.partitioner,
        )
        if self.router.weights_are_final() or self.router.router_mode == 'learned_head':
            local_metrics = self._build_local_validation_metrics(X_val, y_val, list(self.models))
            self._attach_local_validation_metrics(self.models, local_metrics)

        routed_metrics = self._evaluate_current_ensemble(X_val, y_val)
        routing_refinement = self._run_routing_refinement(
            partitions=partitions,
            X_val=X_val,
            y_val=y_val,
            validation_metric=validation_metric,
        )
        if routing_refinement.get('status') == 'completed':
            routed_metrics = self._evaluate_current_ensemble(X_val, y_val)
        best_score = routed_metrics.get(validation_metric)
        self._log(f"Routed MoE metrics after local calibration: {routed_metrics}")
        self._log(f"Routed MoE validation metric ({validation_metric}): {best_score}")
        self.validation_diagnostics_ = self._build_validation_diagnostics(
            X_val=X_val,
            y_val=y_val,
            full_metrics=full_metrics,
            reduced_metrics=routed_metrics,
            best_score=best_score,
            validation_metric=validation_metric,
            selection_policy='moe_keep_routed_experts',
            routing_refinement_diagnostics=routing_refinement,
        )

    def _run_routing_refinement(
        self,
        partitions: Dict[str, Any],
        X_val: pd.DataFrame,
        y_val: pd.Series,
        validation_metric: str,
    ) -> Dict[str, Any]:
        refiner = RoutedEMModelRefiner(
            ensemble=self,
            config=self.partitioner_config,
            show_progress=self.show_progress,
        )
        return refiner.refine(
            partitions=partitions,
            X_val=X_val,
            y_val=y_val,
            validation_metric=validation_metric,
        )

    @staticmethod
    def _attach_local_validation_metrics(
        active_models: List[Dict[str, Any]],
        local_metrics: Dict[str, Any],
    ) -> None:
        for model_info in active_models:
            name = str(model_info.get('name'))
            chunk_metrics = local_metrics.get(name, {})
            model_info['local_metrics'] = chunk_metrics.get('metrics', {})
            model_info['local_assigned_count'] = int(chunk_metrics.get('assigned_count', 0) or 0)
            model_info['local_mean_routing_probability'] = chunk_metrics.get('mean_routing_probability')

    def select_best_models_forward(
            self,
            X_val: pd.DataFrame,
            y_val: np.ndarray,
            metric_is_better: Callable,
            validation_metric: str
    ):
        """
        Forward selection моделей в ансамбле.
        Оставляет в self.models только лучший набор.
        """
        validation_metric = validation_metric or self.validation_metric

        n_models = len(self.models)
        remaining = list(range(n_models))
        selected = []

        best_score = None

        def evaluate(indices):
            selected_models = [self.models[i] for i in indices]
            if not selected_models:
                return None
            preds = self.ensemble_predict(X_val, stage='validation', models=selected_models)

            return calculate_metrics(
                y_true=y_val,
                y_labels=preds,
                y_proba=None,
                problem_type=self.problem,
            )[validation_metric]

        with progress_bar(
            enabled=self.show_progress,
            desc="Forward model selection",
            total=n_models,
        ) as selection_bar:
            while remaining:
                best_candidate = None
                best_candidate_score = best_score

                for i in remaining:
                    candidate = selected + [i]
                    score = evaluate(candidate)

                    if best_candidate_score is None or metric_is_better(score, best_candidate_score):
                        best_candidate = i
                        best_candidate_score = score

                if best_candidate is None:
                    break

                selected.append(best_candidate)
                remaining.remove(best_candidate)
                best_score = best_candidate_score
                selection_bar.update(1)
                self._log(f"Forward selection: models={len(selected)} {validation_metric}={best_score}")
        self.models = [self.models[i] for i in selected]

        return selected, best_score

    def _validation_weights(self, active_models: List[Dict[str, Any]]) -> np.ndarray:
        """
        Converts validation metrics into non-negative model priors.
        For regression lower RMSE/MAE is better; for classification higher F1/accuracy is better.
        """
        return self.router.validation_prior_weights(
            active_models,
            ensemble_method=self.ensemble_method,
        )

    def _build_validation_diagnostics(
        self,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        full_metrics: Dict[str, Any],
        reduced_metrics: Dict[str, Any],
        best_score: Any,
        validation_metric: str,
        selection_policy: str,
        routing_refinement_diagnostics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        active_models = list(self.models)
        return {
            'validation_metric': validation_metric,
            'best_validation_metric': self._safe_float(best_score),
            'selection_policy': selection_policy,
            'active_model_names': [str(model_info.get('name')) for model_info in active_models],
            'full_ensemble_metrics_before_pruning': dict(full_metrics),
            'ensemble_metrics_after_pruning': dict(reduced_metrics),
            'routing': self.build_routing_diagnostics(X_val, active_models=active_models),
            'local_partition_metrics': self._build_local_validation_metrics(X_val, y_val, active_models),
            'router': dict(self.router.diagnostics_),
            'router_head': dict(self.router.diagnostics_),
            'routing_refinement': routing_refinement_diagnostics or {'mode': 'none', 'status': 'disabled'},
        }

    @staticmethod
    def _safe_float(value: Any) -> Any:
        if value is None:
            return None
        try:
            value = float(value)
        except (TypeError, ValueError):
            return value
        return value if np.isfinite(value) else None

    def _build_local_validation_metrics(
        self,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        active_models: List[Dict[str, Any]],
        use_base_router: bool = False,
    ) -> Dict[str, Any]:
        if not active_models or len(X_val) == 0:
            return {}
        return self.router.build_local_validation_metrics(
            X_val=X_val,
            y_val=y_val,
            active_models=active_models,
            partitioner=self.partitioner,
            use_base_router=use_base_router,
        )

    def _routing_weights(self, features: pd.DataFrame, active_models: List[Dict[str, Any]]) -> np.ndarray:
        return self.router.weights(
            features=features,
            active_models=active_models,
            partitioner=self.partitioner,
        )

    def _base_routing_weights(self, features: pd.DataFrame, active_models: List[Dict[str, Any]]) -> np.ndarray:
        return self.router.base_weights(
            features=features,
            active_models=active_models,
            partitioner=self.partitioner,
        )

    def build_routing_diagnostics(
        self,
        features: pd.DataFrame,
        active_models: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        active_models = active_models if active_models is not None else list(self.models)
        return self.router.diagnostics(
            features=features,
            active_models=active_models,
            partitioner=self.partitioner,
        )

    def ensemble_predict(self, features: pd.DataFrame, stage: str = 'inference', models: Optional[List[Dict[str, Any]]] = None) -> np.ndarray:
        """
        Ансамблирование предсказаний всех моделей
        """
        active_models = models if models is not None else self.models

        if not active_models:
            raise ValueError("Модели не обучены. Сначала вызовите train_partition_models()")

        predictions = []

        for model_info in active_models:
            if stage == 'validation':
                pred = model_info['val_predictions']
            elif stage == 'inference':
                pred = model_info['model'].predict(features)
            predictions.append(pred)

        # Различные стратегии ансамблирования
        if self.ensemble_method == 'voting':
            # Для классификации - мажоритарное голосование
            if self.problem == 'classification':
                stacked_preds = np.column_stack(predictions)
                final_pred, _ = mode(stacked_preds, axis=1)
                return final_pred.ravel()

            # Для регрессии - усреднение
            elif self.problem == 'regression':
                return np.mean(predictions, axis=0)

        elif self.ensemble_method == 'weighted':
            # Взвешенное голосование на основе качества моделей
            weights = self._validation_weights(active_models)

            if self.problem == 'classification':
                # Для классификации: взвешенное голосование по вероятностям
                proba_predictions = []
                for model_info in active_models:
                    # Получаем вероятности если доступно
                    try:
                        proba = model_info['model'].predict_proba(features)
                        proba_predictions.append(proba)
                    except:
                        # Fallback to hard voting
                        proba_predictions.append(pd.get_dummies(model_info['model'].predict(features)))

                weighted_proba = np.average(proba_predictions, axis=0, weights=weights)
                return np.argmax(weighted_proba, axis=1)

            elif self.problem == 'regression':
                return np.average(predictions, axis=0, weights=weights)

        elif self.ensemble_method == 'routed_weighted':
            validation_weights = self._validation_weights(active_models)
            routing_weights = self._routing_weights(features, active_models)
            if self.router.weights_are_final():
                combined_weights = routing_weights
            else:
                combined_weights = routing_weights * validation_weights.reshape(1, -1)
            row_sums = combined_weights.sum(axis=1, keepdims=True)
            combined_weights = np.where(row_sums > 0, combined_weights / row_sums, 1.0 / len(active_models))

            if self.problem == 'regression':
                stacked_preds = np.column_stack(predictions)
                return np.sum(stacked_preds * combined_weights, axis=1)

            proba_predictions = []
            classes_reference = None
            for model_info in active_models:
                model = model_info['model']
                if not hasattr(model, 'predict_proba'):
                    # Fallback to hard labels encoded as one-hot over observed predictions.
                    labels = model.predict(features)
                    if classes_reference is None:
                        classes_reference = np.unique(labels)
                    one_hot = np.zeros((len(labels), len(classes_reference)))
                    for class_idx, cls in enumerate(classes_reference):
                        one_hot[:, class_idx] = (labels == cls).astype(float)
                    proba_predictions.append(one_hot)
                    continue
                proba = model.predict_proba(features)
                proba_predictions.append(proba)
                if classes_reference is None and hasattr(model, 'classes_'):
                    classes_reference = np.asarray(model.classes_)

            weighted_proba = np.zeros_like(proba_predictions[0], dtype=float)
            for model_idx, proba in enumerate(proba_predictions):
                weighted_proba += proba * combined_weights[:, model_idx:model_idx + 1]
            if classes_reference is not None and len(classes_reference) == weighted_proba.shape[1]:
                return classes_reference[np.argmax(weighted_proba, axis=1)]
            return np.argmax(weighted_proba, axis=1)

        else:
            raise ValueError(f"Неизвестный метод ансамблирования: {self.ensemble_method}")

    def ensemble_predict_batch(
            self,
            features: pd.DataFrame,
            stage: str = 'inference',
            models: Optional[List[Dict[str, Any]]] = None,
            batch_size: Optional[int] = None,
    ) -> np.ndarray:
        batch_size = batch_size or self.bs_size
        n_samples = len(features)
        batches = []
        total_batches = (n_samples + batch_size - 1) // batch_size
        batch_iter = progress_iter(
            range(total_batches),
            enabled=self.show_progress,
            total=total_batches,
            desc="Ensemble inference batches",
        )
        for batch_idx in batch_iter:
            start = batch_idx * batch_size
            end = min(start + batch_size, n_samples)
            batch = features.iloc[start:end] if isinstance(features, pd.DataFrame) else features[start:end]
            batches.append(self.ensemble_predict(batch, stage=stage, models=models))
        return np.concatenate(batches)

class SingleModelImplementation(SamplingEnsemble):
    """
    Класс для обучения одной модели на поддатасете, созданном из партиций
    """

    def __init__(self,
                 problem: str,
                 partitioner_config: Dict[str, Any] = None,
                 model_class: Optional[Callable] = None,
                 model_params: Dict[str, Any] = None):

        super().__init__(problem, partitioner_config, model_class, model_params)
        self.model = None

    def train_model(self, X_train, y_train, X_val, y_val, class_samples=None,
                   data_percent: float = 0.3, validation_metric: str = None):
        """
        Обучает одну модель на подвыборке из партиций

        Args:
            X_train: обучающие признаки
            y_train: обучающие метки
            X_val: валидационные признаки
            y_val: валидационные метки
            class_samples: представители классов (для классификации)
            data_percent: процент данных для выборки из каждой партиции
            validation_metric: метрика для валидации
        """
        partitions = self.prepare_data_partitions(X_train, y_train)

        if validation_metric is None:
            validation_metric = 'f1_weighted' if self.problem == 'classification' else 'rmse'
        elif validation_metric == 'f1':
            validation_metric = 'f1_weighted'

        metric_is_better = get_metric_comparator(validation_metric)

        all_features = []
        all_targets = []

        rng = np.random.default_rng(42)  # для воспроизводимости

        for partition_name, partition_data in partitions.items():
            X = partition_data['feature']
            y = partition_data['target']

            n = len(X)
            sample_size = int(n * data_percent)

            indices = rng.choice(n, size=sample_size, replace=False)

            all_features.append(X[indices])
            all_targets.append(y[indices])

        data = np.concatenate(all_features, axis=0)
        target = np.concatenate(all_targets, axis=0)

        # Создаем и обучаем модель
        self.model = self._create_model_instance()
        self.model.fit(data, target)

        # Валидация
        predict_labels, predict_proba = self._run_inference(self.model, X_val, calculation_mode='non-batch')
        metrics = calculate_metrics(y_true=y_val,
                                    problem_type=self.problem,
                                    y_labels=predict_labels,
                                    y_proba=None
                                    )
        self._log(f"Validation metrics: {metrics}")
        validation_result = metrics[validation_metric]
        self._log(f"Validation metric ({validation_metric}): {validation_result}")

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Предсказание на новых данных"""
        if not self.model:
            raise ValueError("Модель не обучена. Сначала вызовите train_model()")

        return self.model.predict(features)
