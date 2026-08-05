# model_integration.py
import pickle
import os
from time import perf_counter
import pandas as pd
import numpy as np
from scipy.stats import mode
from typing import List, Dict, Any, Optional, Callable
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sampling_zoo.core.api.api_main import SamplingStrategyFactory
from sampling_zoo.core.metrics.eval_metrics import calculate_metrics, get_metric_comparator
from sampling_zoo.core.experiment.contracts import (
    PartitionModelMode,
    PartitionSizeDiagnosticsContract,
    PartitionTrainingRequest,
    PartitionTrainingResult,
    RuntimeDiagnosticsContract,
)
from sampling_zoo.core.experiment.errors import (
    ClassificationProbabilitiesRequiredError,
)
from sampling_zoo.core.experiment.morphisms import (
    chunk_models_to_contracts,
    partitions_to_contract,
    routing_to_contract,
)
from sampling_zoo.core.utils.ensemble_routing import RoutedWeightedRouter
from sampling_zoo.core.utils.partition_training import (
    build_model_training_partitions,
    normalize_partition_model_mode,
)
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
        self.partition_model_mode = normalize_partition_model_mode(
            self.partitioner_config.get('partition_model_mode')
        )
        if (
            self.partition_model_mode is PartitionModelMode.CONCATENATED
            and self.ensemble_method == 'routed_weighted'
        ):
            raise ValueError(
                "partition_model_mode='concatenated' is incompatible with "
                "ensemble_method='routed_weighted'; use voting for the "
                "single-model control"
            )
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
        self.runtime_diagnostics_ = {}
        self.runtime_contract_ = None
        self.partition_size_diagnostics_contract_ = None
        self.classes_ = None
        self.class_coverage_repairs_ = {}

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
            partition_started = perf_counter()
            timings: Dict[str, float] = {}
            with progress_bar(
                enabled=self.show_progress,
                desc=f"Prepare chunks ({strategy_name})",
                total=4,
            ) as stage:
                started = perf_counter()
                strategy_kwargs = self._build_partitioner_kwargs(strategy_name, random_state)
                timings['config_resolution'] = perf_counter() - started
                stage.update(1)

                started = perf_counter()
                self.partitioner = self._create_partitioner(strategy_name, strategy_kwargs)
                timings['partitioner_initialization'] = perf_counter() - started
                stage.update(1)

                started = perf_counter()
                self.partitions = self._fit_and_collect_partitions(self.partitioner, strategy_name, features, target)
                timings['partitioner_fit_and_collect'] = perf_counter() - started
                stage.update(1)

                started = perf_counter()
                self.partitions = self._apply_budget_policy_to_partitions(
                    partitions=self.partitions,
                    total_rows=len(features),
                    random_state=random_state,
                )
                timings['budget_application'] = perf_counter() - started
                stage.update(1)

            started = perf_counter()
            self.partition_diagnostics_ = self._build_partition_target_diagnostics(self.partitions, target)
            self.partition_size_diagnostics_contract_ = (
                self._build_partition_size_diagnostics_contract(self.partitions)
            )
            timings['partition_diagnostics'] = perf_counter() - started
            timings['total'] = perf_counter() - partition_started
            self.runtime_diagnostics_['partitioning'] = timings
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
            'experiment_scenario',
            'force_chunking',
            'force_direct_model',
            'partition_model_mode',
            'scenario_family',
            'scenario_group',
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
            'downstream_proxy_n_estimators',
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
            budget_ratio = self.partitioner_config.get('budget_ratio')
            if budget_ratio is not None:
                strategy_kwargs.setdefault('sampling_budget_ratio', float(budget_ratio))
            selection_metric = str(
                strategy_kwargs.get('cluster_selection_metric', '')
            ).strip().lower()
            if selection_metric in {
                'budget_aware_validation_proxy',
                'downstream_proxy',
            }:
                strategy_kwargs.setdefault('budget_feasibility_mode', 'hard')
                strategy_kwargs.setdefault('include_single_partition_candidate', True)
                strategy_kwargs.setdefault('min_sampled_rows_per_partition', 32)
                strategy_kwargs.setdefault('budget_max_imbalance_ratio', 5.0)
                strategy_kwargs.setdefault('budget_min_partition_fraction', 0.05)
            if selection_metric == 'downstream_proxy':
                strategy_kwargs['downstream_proxy_model_factory'] = (
                    self._build_downstream_proxy_model_factory()
                )
        return strategy_kwargs

    def _build_downstream_proxy_model_factory(self) -> Callable[[], Any]:
        max_estimators = int(
            self.partitioner_config.get('downstream_proxy_n_estimators', 32)
        )
        if max_estimators < 1:
            raise ValueError("downstream_proxy_n_estimators must be positive")

        def _factory() -> Any:
            model = self._create_model_instance()
            if not hasattr(model, 'get_params') or not hasattr(model, 'set_params'):
                raise ValueError(
                    "downstream_proxy requires a sklearn-compatible model with get_params/set_params"
                )
            params = model.get_params(deep=False)
            updates: Dict[str, Any] = {}
            if 'n_estimators' in params:
                current = params.get('n_estimators')
                current = max_estimators if current is None else int(current)
                updates['n_estimators'] = min(current, max_estimators)
            if 'max_iter' in params:
                current = params.get('max_iter')
                current = max_estimators if current is None else int(current)
                updates['max_iter'] = min(current, max_estimators)
            if 'n_jobs' in params:
                updates['n_jobs'] = 1
            if updates:
                model.set_params(**updates)
            return model

        return _factory

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
        if strategy_name == 'rmt_contraction':
            return self._fit_target_aware_partitioner(partitioner, features, target)
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
    def _fit_target_aware_partitioner(
        partitioner: Any,
        features: pd.DataFrame,
        target: pd.Series,
    ) -> Dict[str, Any]:
        partitioner.fit(features, target=target)
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

        sampler_budget_plan = getattr(
            self.partitioner,
            'partition_budget_plan_',
            None,
        )
        if sampler_budget_plan is not None:
            self.budget_policy_ = {
                **sampler_budget_plan.to_dict(),
                'applied': True,
                'applied_by': 'partitioner',
            }
            return partitions

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

    def _build_partition_size_diagnostics_contract(
        self,
        partitions: Dict[str, Any],
    ) -> PartitionSizeDiagnosticsContract:
        post_sizes = {
            str(name): int(self._partition_size(partition))
            for name, partition in partitions.items()
        }
        sampler_plan = getattr(self.partitioner, 'partition_budget_plan_', None)
        if sampler_plan is not None:
            pre_sizes = sampler_plan.source_size_map
            requested_budget_size = int(sampler_plan.requested_budget_size)
        else:
            policy = getattr(self, 'budget_policy_', {}) or {}
            pre_sizes = {
                str(name): int(size)
                for name, size in policy.get('source_sizes', post_sizes).items()
            }
            requested_budget_size = policy.get('budget_size')
        selected_rows = int(sum(post_sizes.values()))
        return PartitionSizeDiagnosticsContract(
            pre_budget_sizes=pre_sizes,
            post_budget_sizes=post_sizes,
            requested_budget_size=(
                None
                if requested_budget_size is None
                else int(requested_budget_size)
            ),
            selected_rows=selected_rows,
            unique_selected_rows=selected_rows,
            duplicate_rows=0,
        )

    def _build_partition_target_diagnostics(self, partitions: Dict[str, Any], target: pd.Series) -> Dict[str, Any]:
        if self.problem == 'classification':
            return self._build_classification_partition_diagnostics(
                partitions,
                target,
            )
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

    def _build_classification_partition_diagnostics(
        self,
        partitions: Dict[str, Any],
        target: pd.Series,
    ) -> Dict[str, Any]:
        global_values = np.asarray(target).reshape(-1)
        classes, global_counts = np.unique(global_values, return_counts=True)
        global_distribution = global_counts / max(global_counts.sum(), 1)
        chunks: Dict[str, Any] = {}
        sizes: List[int] = []
        missing_chunk_count = 0
        single_class_count = 0
        drifts: List[float] = []

        for name, partition_data in partitions.items():
            chunk_values = self._partition_raw_target_values(partition_data)
            class_counts = np.asarray(
                [np.sum(chunk_values == label) for label in classes],
                dtype=int,
            )
            size = int(class_counts.sum())
            sizes.append(size)
            present_mask = class_counts > 0
            missing = [
                str(label)
                for label, present in zip(classes, present_mask)
                if not present
            ]
            if missing:
                missing_chunk_count += 1
            single_class = int(np.sum(present_mask)) <= 1
            if single_class:
                single_class_count += 1
            chunk_distribution = class_counts / max(size, 1)
            drift = float(
                0.5 * np.sum(np.abs(chunk_distribution - global_distribution))
            )
            drifts.append(drift)
            positive_counts = class_counts[present_mask]
            chunks[str(name)] = {
                'count': size,
                'class_counts': {
                    str(label): int(count)
                    for label, count in zip(classes, class_counts)
                },
                'present_classes': [
                    str(label)
                    for label, present in zip(classes, present_mask)
                    if present
                ],
                'missing_classes': missing,
                'single_class_chunk': bool(single_class),
                'min_present_class_count': (
                    int(np.min(positive_counts))
                    if positive_counts.size
                    else 0
                ),
                'class_distribution_drift': drift,
            }

        sampler_diagnostics = getattr(self.partitioner, 'diagnostics_', {}) or {}
        return {
            'global_target': {
                'count': int(global_values.size),
                'n_classes': int(classes.size),
                'classes': [str(label) for label in classes],
                'class_counts': {
                    str(label): int(count)
                    for label, count in zip(classes, global_counts)
                },
            },
            'chunk_size_imbalance': self._chunk_size_imbalance(sizes),
            'class_balance_summary': {
                'chunks_with_missing_classes': int(missing_chunk_count),
                'single_class_chunks': int(single_class_count),
                'class_distribution_drift_avg': (
                    float(np.mean(drifts)) if drifts else None
                ),
                'class_distribution_drift_max': (
                    float(np.max(drifts)) if drifts else None
                ),
            },
            'sampler_class_coverage': dict(
                sampler_diagnostics.get('class_coverage_by_partition', {})
            ),
            'chunks': chunks,
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
    def _partition_raw_target_values(partition_data: Any) -> np.ndarray:
        if isinstance(partition_data, dict) and 'target' in partition_data:
            return np.asarray(partition_data['target']).reshape(-1)
        return np.asarray([], dtype=object)

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

    def _ensure_classification_classes(self, y_train: Any, y_val: Any = None) -> None:
        if self.problem != 'classification':
            return
        values = [pd.Series(y_train)]
        if y_val is not None:
            values.append(pd.Series(y_val))
        combined = pd.concat(values, ignore_index=True).dropna()
        self.classes_ = np.asarray(np.unique(combined.to_numpy()))

    def _classification_classes(self) -> np.ndarray:
        if self.classes_ is not None:
            return np.asarray(self.classes_)
        observed = []
        for model_info in self.models:
            model_classes = model_info.get('classes')
            if model_classes is not None:
                observed.extend(np.asarray(model_classes).tolist())
        if observed:
            self.classes_ = np.asarray(np.unique(observed))
            return np.asarray(self.classes_)
        raise ClassificationProbabilitiesRequiredError(
            scope="SamplingEnsemble.classes",
            message="Global classification labels are not initialized.",
        )

    def _labels_from_proba(self, probabilities: np.ndarray) -> np.ndarray:
        classes = self._classification_classes()
        return classes[np.argmax(probabilities, axis=1)]

    def _predict_model_proba(
        self,
        model: Callable,
        features: pd.DataFrame,
        *,
        model_name: str = "model",
    ) -> np.ndarray:
        if not hasattr(model, 'predict_proba'):
            raise ClassificationProbabilitiesRequiredError(
                scope=f"SamplingEnsemble.{model_name}",
                details={"model": type(model).__name__},
            )
        probabilities = np.asarray(model.predict_proba(features), dtype=float)
        return self._align_model_proba(
            probabilities,
            getattr(model, 'classes_', None),
            model_name=model_name,
        )

    def _align_model_proba(
        self,
        probabilities: np.ndarray,
        model_classes: Any,
        *,
        model_name: str = "model",
    ) -> np.ndarray:
        global_classes = self._classification_classes()
        if probabilities.ndim != 2 or probabilities.shape[1] == 0:
            raise ClassificationProbabilitiesRequiredError(
                scope=f"SamplingEnsemble.{model_name}",
                message="predict_proba returned an invalid probability matrix.",
                details={"shape": tuple(probabilities.shape)},
            )
        if model_classes is None:
            if probabilities.shape[1] != global_classes.size:
                raise ClassificationProbabilitiesRequiredError(
                    scope=f"SamplingEnsemble.{model_name}",
                    message="Probability columns cannot be aligned without classes_.",
                    details={
                        "probability_columns": int(probabilities.shape[1]),
                        "global_classes": int(global_classes.size),
                    },
                )
            model_classes = global_classes

        model_classes = np.asarray(model_classes)
        if model_classes.size != probabilities.shape[1]:
            raise ClassificationProbabilitiesRequiredError(
                scope=f"SamplingEnsemble.{model_name}",
                message="classes_ does not align with predict_proba columns.",
                details={
                    "probability_columns": int(probabilities.shape[1]),
                    "model_classes": int(model_classes.size),
                },
            )

        aligned = np.zeros(
            (probabilities.shape[0], global_classes.size),
            dtype=float,
        )
        for model_index, label in enumerate(model_classes):
            matches = np.where(global_classes == label)[0]
            if matches.size != 1:
                raise ClassificationProbabilitiesRequiredError(
                    scope=f"SamplingEnsemble.{model_name}",
                    message="Model exposes a class absent from the global class set.",
                    details={"model_class": str(label)},
                )
            aligned[:, int(matches[0])] = probabilities[:, model_index]

        aligned = np.clip(aligned, 1e-15, 1.0)
        row_sums = aligned.sum(axis=1, keepdims=True)
        return np.where(
            row_sums > 0,
            aligned / row_sums,
            1.0 / max(global_classes.size, 1),
        )

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
                    proba = self._predict_model_proba(fitted_model, batch)
                    labels = self._labels_from_proba(proba)
                    predict_labels.append(labels)
                    predict_proba.append(proba)
            return np.concatenate(predict_labels), np.concatenate(predict_proba)
        elif calculation_mode == 'non-batch':
            if self.problem == 'regression':
                labels = fitted_model.predict(test_data)
                proba = labels
            else:
                proba = self._predict_model_proba(fitted_model, test_data)
                labels = self._labels_from_proba(proba)
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
        Train models from independent or concatenated prepared partitions.
        """
        training_started = perf_counter()
        self._ensure_classification_classes(y_train, y_val)
        self.class_coverage_repairs_ = {}
        started = perf_counter()
        partitions = self._load_or_prepare_partitions(
            X_train=X_train,
            y_train=y_train,
            cv_fold=cv_fold,
            save_models_to_disk=save_models_to_disk,
        )
        partition_preparation_time = perf_counter() - started
        training_partitions = build_model_training_partitions(
            partitions,
            self.partition_model_mode,
        )
        validation_metric = self._normalize_validation_metric(validation_metric)
        metric_is_better = get_metric_comparator(validation_metric)
        training_request = self._build_partition_training_request(
            source_partitions=partitions,
            training_partitions=training_partitions,
            validation_metric=validation_metric,
        )

        started = perf_counter()
        self._train_partition_loop(
            partitions=training_partitions,
            X_val=X_val,
            y_val=y_val,
            class_samples=class_samples,
            cv_fold=cv_fold,
            validation_metric=validation_metric,
            metric_is_better=metric_is_better,
            train_all_chunks=train_all_chunks,
            save_models_to_disk=save_models_to_disk,
        )
        model_training_time = perf_counter() - started
        started = perf_counter()
        self._finalize_partition_training(
            partitions=training_partitions,
            X_val=X_val,
            y_val=y_val,
            metric_is_better=metric_is_better,
            validation_metric=validation_metric,
        )
        routing_finalization_time = perf_counter() - started
        self.partition_training_contract_ = self._build_partition_training_result(
            request=training_request,
            partitions=partitions,
        )
        self.runtime_diagnostics_['training'] = {
            'partition_preparation': float(partition_preparation_time),
            'chunk_training_and_validation': float(model_training_time),
            'routing_and_finalization': float(routing_finalization_time),
            'chunk_model_fit_total': float(
                sum(
                    model_info.get('timings', {}).get('fit', 0.0)
                    for model_info in self.models
                )
            ),
            'chunk_validation_inference_total': float(
                sum(
                    model_info.get('timings', {}).get('validation_inference', 0.0)
                    for model_info in self.models
                )
            ),
            'total': float(perf_counter() - training_started),
        }
        stage_seconds = {
            f"partitioning.{name}": float(value)
            for name, value in self.runtime_diagnostics_.get('partitioning', {}).items()
            if name != 'total'
        }
        if not stage_seconds:
            stage_seconds['partitioning.total'] = float(partition_preparation_time)
        stage_seconds.update({
            'training.chunk_training_and_validation': float(model_training_time),
            'training.routing_and_finalization': float(routing_finalization_time),
        })
        self.runtime_contract_ = RuntimeDiagnosticsContract(
            stage_seconds=stage_seconds,
            total_seconds=float(self.runtime_diagnostics_['training']['total']),
            cold_start=True,
            metadata={
                'strategy': self._strategy_name(),
                'n_partitions': len(partitions),
                'n_training_partitions': len(training_partitions),
                'partition_model_mode': self.partition_model_mode.value,
            },
        )

    def _build_partition_training_request(
        self,
        source_partitions: Dict[str, Any],
        training_partitions: Dict[str, Any],
        validation_metric: str,
    ) -> PartitionTrainingRequest:
        return PartitionTrainingRequest(
            problem=self.problem,
            ensemble_method=self.ensemble_method,
            validation_metric=validation_metric,
            n_partitions=len(source_partitions),
            routing_refinement=str(self.partitioner_config.get('routing_refinement', 'none')),
            partition_model_mode=self.partition_model_mode.value,
            n_training_partitions=len(training_partitions),
        )

    def _build_partition_training_result(
        self,
        request: PartitionTrainingRequest,
        partitions: Dict[str, Any],
    ) -> PartitionTrainingResult:
        return PartitionTrainingResult(
            request=request,
            partitions=partitions_to_contract(
                partitions,
                {
                    **self.partition_diagnostics_,
                    "size_contract": (
                        self.partition_size_diagnostics_contract_.to_dict()
                        if self.partition_size_diagnostics_contract_ is not None
                        else {}
                    ),
                },
            ),
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
            except ClassificationProbabilitiesRequiredError:
                raise
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
        partition_data = self._ensure_partition_class_coverage(
            partition_name,
            partition_data,
            class_samples,
        )
        model = self._create_model_instance()
        fit_started = perf_counter()
        model.fit(partition_data['feature'], partition_data['target'])
        model_fit_time = perf_counter() - fit_started
        self._save_partition_model_if_requested(model, partition_name, cv_fold, save_models_to_disk)

        inference_started = perf_counter()
        predict_labels, predict_proba = self._run_inference(model, X_val, calculation_mode='non-batch')
        validation_inference_time = perf_counter() - inference_started
        metrics = calculate_metrics(
            y_true=y_val,
            problem_type=self.problem,
            y_labels=predict_labels,
            y_proba=predict_proba if self.problem == "classification" else None,
            classes=(
                self._classification_classes()
                if self.problem == "classification"
                else None
            ),
        )
        model_info = self._build_partition_model_info(
            partition_name,
            model,
            partition_data,
            metrics,
            predict_labels,
            predict_proba,
            timings={
                'fit': model_fit_time,
                'validation_inference': validation_inference_time,
            },
        )
        self._register_partition_model(partition_name, model_info, metrics)
        self._log_partition_model_result(partition_name, model_info, metrics)
        return model_info

    def _ensure_partition_class_coverage(
        self,
        partition_name: str,
        partition_data: Dict[str, Any],
        class_samples: Any,
    ) -> Dict[str, Any]:
        before = self._class_count_map(partition_data.get('target', []))
        if getattr(self.partitioner, 'class_coverage_guaranteed_', False):
            self._record_class_coverage_repair(
                partition_name,
                before,
                before,
                status='sampler_guaranteed',
            )
            return partition_data
        if self.problem == 'classification' and class_samples:
            repaired = self.ensure_all_classes_in_chunk(
                partition_data,
                class_samples,
            )
            self._record_class_coverage_repair(
                partition_name,
                before,
                self._class_count_map(repaired.get('target', [])),
                status='representatives_added',
            )
            return repaired
        self._record_class_coverage_repair(
            partition_name,
            before,
            before,
            status='not_applied',
        )
        return partition_data

    @staticmethod
    def _class_count_map(target: Any) -> Dict[str, int]:
        values = np.asarray(target).reshape(-1)
        if values.size == 0:
            return {}
        classes, counts = np.unique(values, return_counts=True)
        return {
            str(label): int(count)
            for label, count in zip(classes, counts)
        }

    def _record_class_coverage_repair(
        self,
        partition_name: str,
        before: Dict[str, int],
        after: Dict[str, int],
        *,
        status: str,
    ) -> None:
        global_classes = (
            {str(label) for label in self._classification_classes()}
            if self.problem == 'classification'
            else set()
        )
        all_classes = global_classes | set(before) | set(after)
        rows_before = int(sum(before.values()))
        rows_after = int(sum(after.values()))
        self.class_coverage_repairs_[str(partition_name)] = {
            'status': status,
            'class_counts_before': dict(before),
            'class_counts_after': dict(after),
            'missing_classes_before': sorted(
                label for label in all_classes if before.get(label, 0) == 0
            ),
            'missing_classes_after': sorted(
                label for label in all_classes if after.get(label, 0) == 0
            ),
            'rows_added': int(rows_after - rows_before),
        }
        self.partition_diagnostics_['class_coverage_repairs'] = dict(
            self.class_coverage_repairs_
        )

    @staticmethod
    def _build_partition_model_info(
        partition_name: str,
        model: Callable,
        partition_data: Dict[str, Any],
        metrics: Dict[str, Any],
        predict_labels: np.ndarray,
        predict_proba: Optional[np.ndarray] = None,
        timings: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        return {
            'name': partition_name,
            'model': model,
            'data_size': len(partition_data['feature']),
            'metrics': metrics,
            'val_predictions': predict_labels,
            'val_probabilities': predict_proba,
            'classes': getattr(model, 'classes_', None),
            'timings': dict(timings or {}),
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
        probabilities = (
            self.ensemble_predict_proba(X_val, stage='validation')
            if self.problem == 'classification'
            else None
        )
        predictions = (
            self._labels_from_proba(probabilities)
            if probabilities is not None
            else self.ensemble_predict(X_val, stage='validation')
        )
        return calculate_metrics(
            y_true=y_val,
            y_labels=predictions,
            y_proba=probabilities,
            problem_type=self.problem,
            classes=(
                self._classification_classes()
                if self.problem == 'classification'
                else None
            ),
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
            probabilities = (
                self.ensemble_predict_proba(
                    X_val,
                    stage='validation',
                    models=selected_models,
                )
                if self.problem == 'classification'
                else None
            )
            preds = (
                self._labels_from_proba(probabilities)
                if probabilities is not None
                else self.ensemble_predict(
                    X_val,
                    stage='validation',
                    models=selected_models,
                )
            )

            return calculate_metrics(
                y_true=y_val,
                y_labels=preds,
                y_proba=probabilities,
                problem_type=self.problem,
                classes=(
                    self._classification_classes()
                    if self.problem == 'classification'
                    else None
                ),
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
            'partition_model_mode': self.partition_model_mode.value,
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

    def _legacy_ensemble_predict(self, features: pd.DataFrame, stage: str = 'inference', models: Optional[List[Dict[str, Any]]] = None) -> np.ndarray:
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

    def _ensure_active_models(
        self,
        models: Optional[List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        active_models = list(self.models if models is None else models)
        if not active_models:
            raise ValueError(
                "No trained chunk models are available. Call "
                "train_partition_models() first."
            )
        return active_models

    @staticmethod
    def _normalize_combined_weights(
        weights: np.ndarray,
        n_models: int,
    ) -> np.ndarray:
        row_sums = weights.sum(axis=1, keepdims=True)
        return np.where(
            row_sums > 0,
            weights / row_sums,
            1.0 / max(n_models, 1),
        )

    def _combined_routing_weights(
        self,
        features: pd.DataFrame,
        active_models: List[Dict[str, Any]],
    ) -> np.ndarray:
        validation_weights = self._validation_weights(active_models)
        routing_weights = self._routing_weights(features, active_models)
        combined = (
            routing_weights
            if self.router.weights_are_final()
            else routing_weights * validation_weights.reshape(1, -1)
        )
        return self._normalize_combined_weights(combined, len(active_models))

    def _model_proba_predictions(
        self,
        features: pd.DataFrame,
        stage: str,
        active_models: List[Dict[str, Any]],
    ) -> List[np.ndarray]:
        probabilities: List[np.ndarray] = []
        for model_info in active_models:
            name = str(model_info.get('name', 'model'))
            if stage == 'validation':
                stored = model_info.get('val_probabilities')
                if stored is None:
                    raise ClassificationProbabilitiesRequiredError(
                        scope=f"SamplingEnsemble.{name}",
                        message="Validation probabilities were not stored for a chunk model.",
                    )
                value = np.asarray(stored, dtype=float)
                expected_columns = self._classification_classes().size
                if value.ndim != 2 or value.shape[1] != expected_columns:
                    raise ClassificationProbabilitiesRequiredError(
                        scope=f"SamplingEnsemble.{name}",
                        message="Stored validation probabilities are not globally aligned.",
                        details={"shape": tuple(value.shape)},
                    )
                probabilities.append(value)
            elif stage == 'inference':
                probabilities.append(
                    self._predict_model_proba(
                        model_info['model'],
                        features,
                        model_name=name,
                    )
                )
            else:
                raise ValueError("stage must be 'validation' or 'inference'")
        return probabilities

    def export_expert_outputs(
        self,
        features: pd.DataFrame,
        *,
        stage: str = 'inference',
        models: Optional[List[Dict[str, Any]]] = None,
    ) -> tuple[tuple[str, ...], np.ndarray]:
        """Return fixed expert outputs for offline routing-policy replay.

        Regression output has shape ``(rows, experts)``. Classification output
        has shape ``(rows, experts, classes)`` with the global class order used
        by :meth:`ensemble_predict_proba`.
        """

        active_models = self._ensure_active_models(models)
        names = tuple(str(model.get('name', f'model_{index}')) for index, model in enumerate(active_models))
        if self.problem == 'classification':
            outputs = np.stack(
                self._model_proba_predictions(features, stage, active_models),
                axis=1,
            )
        else:
            predictions = [
                (
                    np.asarray(model_info['val_predictions'])
                    if stage == 'validation'
                    else np.asarray(model_info['model'].predict(features))
                )
                for model_info in active_models
            ]
            outputs = np.column_stack(predictions)
        return names, np.asarray(outputs, dtype=float)

    def validation_prior_weights(
        self,
        models: Optional[List[Dict[str, Any]]] = None,
    ) -> np.ndarray:
        """Expose validation-derived expert priors for deterministic replay."""

        return self._validation_weights(self._ensure_active_models(models))

    def ensemble_predict_proba(
        self,
        features: pd.DataFrame,
        stage: str = 'inference',
        models: Optional[List[Dict[str, Any]]] = None,
    ) -> np.ndarray:
        """Return class-aligned probabilities for a classification ensemble."""

        if self.problem != 'classification':
            raise ValueError(
                "ensemble_predict_proba is only available for classification"
            )
        active_models = self._ensure_active_models(models)
        model_probabilities = self._model_proba_predictions(
            features,
            stage,
            active_models,
        )
        if self.ensemble_method == 'voting':
            probabilities = np.mean(model_probabilities, axis=0)
        elif self.ensemble_method == 'weighted':
            probabilities = np.average(
                model_probabilities,
                axis=0,
                weights=self._validation_weights(active_models),
            )
        elif self.ensemble_method == 'routed_weighted':
            combined_weights = self._combined_routing_weights(
                features,
                active_models,
            )
            probabilities = np.zeros_like(model_probabilities[0], dtype=float)
            for model_index, model_proba in enumerate(model_probabilities):
                probabilities += (
                    model_proba
                    * combined_weights[:, model_index:model_index + 1]
                )
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")

        probabilities = np.clip(
            np.asarray(probabilities, dtype=float),
            1e-15,
            1.0,
        )
        row_sums = probabilities.sum(axis=1, keepdims=True)
        return np.where(
            row_sums > 0,
            probabilities / row_sums,
            1.0 / probabilities.shape[1],
        )

    def ensemble_predict(
        self,
        features: pd.DataFrame,
        stage: str = 'inference',
        models: Optional[List[Dict[str, Any]]] = None,
    ) -> np.ndarray:
        active_models = self._ensure_active_models(models)
        if self.problem == 'classification':
            return self._labels_from_proba(
                self.ensemble_predict_proba(
                    features,
                    stage=stage,
                    models=active_models,
                )
            )

        predictions = [
            (
                np.asarray(model_info['val_predictions'])
                if stage == 'validation'
                else np.asarray(model_info['model'].predict(features))
            )
            for model_info in active_models
        ]
        if self.ensemble_method == 'voting':
            return np.mean(predictions, axis=0)
        if self.ensemble_method == 'weighted':
            return np.average(
                predictions,
                axis=0,
                weights=self._validation_weights(active_models),
            )
        if self.ensemble_method == 'routed_weighted':
            combined_weights = self._combined_routing_weights(
                features,
                active_models,
            )
            return np.sum(
                np.column_stack(predictions) * combined_weights,
                axis=1,
            )
        raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")

    def ensemble_predict_proba_batch(
        self,
        features: pd.DataFrame,
        stage: str = 'inference',
        models: Optional[List[Dict[str, Any]]] = None,
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        if stage == 'validation':
            return self.ensemble_predict_proba(
                features,
                stage=stage,
                models=models,
            )
        batch_size = batch_size or self.bs_size
        total_batches = (len(features) + batch_size - 1) // batch_size
        batches = []
        batch_iter = progress_iter(
            range(total_batches),
            enabled=self.show_progress,
            total=total_batches,
            desc="Ensemble probability batches",
        )
        for batch_index in batch_iter:
            start = batch_index * batch_size
            end = min(start + batch_size, len(features))
            batch = (
                features.iloc[start:end]
                if isinstance(features, pd.DataFrame)
                else features[start:end]
            )
            batches.append(
                self.ensemble_predict_proba(
                    batch,
                    stage=stage,
                    models=models,
                )
            )
        return np.vstack(batches)

    def ensemble_predict_batch(
            self,
            features: pd.DataFrame,
            stage: str = 'inference',
            models: Optional[List[Dict[str, Any]]] = None,
            batch_size: Optional[int] = None,
    ) -> np.ndarray:
        if stage == 'validation':
            return self.ensemble_predict(
                features,
                stage=stage,
                models=models,
            )
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
