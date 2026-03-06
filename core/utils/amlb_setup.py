"""Инфраструктура для экспериментов AMLB с декомпозицией по модулям."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np
from sklearn.model_selection import KFold
from core.metrics.eval_metrics import calculate_metrics
from core.repository.constant_repo import AmlbExperimentDataset
from core.utils.amlb_config import ModelSpec, ExperimentConfig, ExperimentConfigBuilder, SamplingStrategySpec
from core.utils.amlb_dataloader import AMLBDatasetLoader
from core.utils.amlb_tracking import ExperimentTracker
from core.utils.fedot_integration import SamplingEnsemble, SingleModelImplementation

__all__ = [
    "ExperimentConfig",
    "ExperimentConfigBuilder",
    "LargeScaleAutoMLExperiment",
]


def _resolve_dataset(dataset_loader: AMLBDatasetLoader, dataset_name: str):
    all_datasets = (
        dataset_loader.get_custom_datasets()
        + dataset_loader.get_classification_datasets()
        + dataset_loader.get_regression_datasets()
    )
    for dataset in all_datasets:
        if dataset.get("name") == dataset_name:
            return dataset
    return None


class SamplingRunner:
    """Отвечает за запуск стратегий семплирования и базовых моделей."""

    def __init__(self, experiment_config: ExperimentConfig):
        self.experiment_config = experiment_config

    def _get_model_class(self, model_spec: ModelSpec, problem_type: str):
        """Получает класс модели и параметры из спецификации"""
        from lightgbm import LGBMClassifier, LGBMRegressor
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        model_name = model_spec.name.lower()

        if model_name == 'lgbm':
            if problem_type == 'classification':
                return LGBMClassifier, model_spec.params
            else:
                return LGBMRegressor, model_spec.params
        elif model_name == 'random_forest':
            if problem_type == 'classification':
                return RandomForestClassifier, model_spec.params
            else:
                return RandomForestRegressor, model_spec.params
        else:
            raise ValueError(f"Model {model_name} not supported")

    def run_sampling_ensemble(
            self, X_train, y_train, X_val, y_val, X_test, y_test,
            dataset_info: Dict, class_samples: Optional[Dict] = None, cv_fold: Optional[int] = None
    ) -> Tuple[Dict, SamplingEnsemble]:
        sampling_config = {**AmlbExperimentDataset.SAMPLING_PRESET.value}
        strategy: SamplingStrategySpec = self.experiment_config.sampling_strategies[0]
        sampling_config.update(strategy.params)
        sampling_config['strategy'] = strategy.name

        # Получаем класс модели из конфигурации
        model_spec = self.experiment_config.models[0]
        model_class, model_params = self._get_model_class(model_spec, dataset_info["type"])

        ensemble = SamplingEnsemble(
            problem=dataset_info["type"],
            partitioner_config=sampling_config,
            model_class=model_class,
            model_params=model_params,
            ensemble_method="voting",
        )

        # Передаем validation_metric если есть в model_config
        validation_metric = self.experiment_config.model_config.get('metric', None)
        ensemble.train_partition_models(X_train, y_train, X_val, y_val, class_samples, cv_fold,
                                       validation_metric=validation_metric)
        predictions = ensemble.ensemble_predict(X_test)

        metrics = calculate_metrics(y_test, predictions, None, dataset_info["type"])
        print(f'Test metrics: {metrics}')

        return metrics, ensemble

    def run_single_model(
        self, X_train, y_train, X_val, y_val, X_test, y_test,
        dataset_info: Dict, class_samples: Optional[Dict] = None, cv_fold: Optional[int] = None,
    ) -> Tuple[Dict, SingleModelImplementation]:
        sampling_config = {**AmlbExperimentDataset.SAMPLING_PRESET.value}
        strategy: SamplingStrategySpec = self.experiment_config.sampling_strategies[0]
        sampling_config.update(strategy.params)
        sampling_config['strategy'] = strategy.name

        # Получаем класс модели из конфигурации
        model_spec = self.experiment_config.models[0]
        model_class, model_params = self._get_model_class(model_spec, dataset_info["type"])

        single_model = SingleModelImplementation(
            problem=dataset_info["type"],
            model_class=model_class,
            model_params=model_params,
            partitioner_config=sampling_config,
        )

        validation_metric = self.experiment_config.model_config.get('metric', None)
        single_model.train_model(X_train, y_train, X_val, y_val, class_samples,
                                validation_metric=validation_metric)
        predictions = single_model.predict(X_test)

        metrics = calculate_metrics(y_test, predictions, None, dataset_info["type"])
        print(f'Test metrics: {metrics}')

        return metrics, single_model


class ResultLogger:
    """Сохраняет результаты и формирует отчеты."""

    def __init__(self, results_path: str = "experiment_results"):
        self.results_path = Path(results_path)
        self.results_path.mkdir(parents=True, exist_ok=True)

    def save(self, payload: Dict) -> str:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = self.results_path / f"experiment_results_{timestamp}.json"

        def convert_types(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        with open(filename, "w") as fh:
            json.dump(payload, fh, indent=2, default=convert_types)
        return str(filename)

    @staticmethod
    def report(results: Dict) -> None:
        print("\n" + "=" * 70)
        print("ИТОГОВЫЙ ОТЧЕТ ЭКСПЕРИМЕНТА")
        print("=" * 70)
        for dataset_name, result in results.items():
            print(f"\n📊 ДАТАСЕТ: {dataset_name}")
            print(f"   Размер данных: {result['train_size']} train, {result['test_size']} test")
            sampling = result.get("sampling", {})
            if "error" not in sampling:
                for metric_name, metric_value in sampling.items():
                    if isinstance(metric_value, (int, float)):
                        print(f"   {metric_name}: {metric_value}")


class LargeScaleAutoMLExperiment:
    """Оркестратор экспериментов на датасетах AutoML Benchmark."""

    def __init__(
        self,
        experiment_config: ExperimentConfig,
        results_path: str = "experiment_results",
        tracker: ExperimentTracker | None = None,
    ):
        self.config = experiment_config
        self.loader = AMLBDatasetLoader()
        self.results: Dict[str, Dict] = {}
        self.runner = SamplingRunner(experiment_config)
        self.logger = ResultLogger(results_path)
        self.tracker = tracker or ExperimentTracker(
            experiment_name=self.config.experiment_name,
            tracking_uri=self.config.tracking_uri,
        )

    def run_full_benchmark(self) -> None:
        for dataset_spec in self.config.datasets:
            dataset_info = _resolve_dataset(self.loader, dataset_spec.name)
            if not dataset_info:
                print(f"Датасет {dataset_spec.name} не найден в репозитории")
                continue
            dataset_info = {**dataset_info, **dataset_spec.params}
            result = self.run_experiment_on_dataset(dataset_info)
            if result:
                self.results[dataset_info["name"]] = result

        saved_path = self.logger.save(self.results)
        print(f"Результаты сохранены в {saved_path}")
        self.logger.report(self.results)

    def run_experiment_on_dataset(self, dataset_info: Dict) -> Dict | None:
        X, y, dataset_info = self.loader.load_dataset(dataset_info)
        if X is None:
            return None

        metrics = []
        cv_folds_data = []
        if self.config.cv_folds == 1:
            X_train, X_val, X_test, y_train, y_val, y_test = self.loader.prepare_train_val_test_balanced(
                X,
                y,
                test_size=0.1,
                val_size=0.1,
                min_samples=20,
                problem=dataset_info["type"]
            )
            cv_folds_data.append((X_train, X_val, X_test, y_train, y_val, y_test))
        else:
            kf = KFold(n_splits=self.config.cv_folds, shuffle=True, random_state=42)
            for train_idx, test_idx in kf.split(X):
                X_train, X_test, y_train, y_test = X[train_idx], X[test_idx], y[train_idx], y[test_idx]

                X_train, _, X_val, y_train, _, y_val = self.loader.prepare_train_val_test_balanced(
                    X_train,
                    y_train,
                    test_size=0.2,
                    val_size=0,
                    min_samples=20,
                    problem=dataset_info["type"]
                )
                cv_folds_data.append((X_train, X_val, X_test, y_train, y_val, y_test))

        dataset_result = {
            "dataset": dataset_info,
            "cv_folds": self.config.cv_folds,
        }

        run_obj = self.tracker.start_run(
            run_name=dataset_info["name"],
            params={"time_budget": self.config.time_budget_minutes},
        )

        for current_fold, (X_train, X_val, X_test, y_train, y_val, y_test) in enumerate(cv_folds_data):
            class_samples = self.loader.select_one_sample_per_class(X_train, y_train) \
                if dataset_info["type"] == "classification" else None

            try:
                if self.config.run_mode == 'chunks':
                    fold_metrics, ensemble = self.runner.run_sampling_ensemble(
                        X_train, y_train, X_val, y_val, X_test, y_test, dataset_info, class_samples, current_fold
                    )
                    metrics.append(fold_metrics)
                    fold_metrics["n_partitions"] = len(ensemble.models)
                    fold_metrics["partition_metrics"] = ensemble.partition_metrics
                    dataset_result[f"sampling_{current_fold}"] = fold_metrics
                    print(f'------------------ fold {current_fold} --------------')
                    print(fold_metrics)
                elif self.config.run_mode == 'mixed_chunk':
                    fold_metrics, model = self.runner.run_single_model(
                        X_train, y_train, X_val, y_val, X_test, y_test, dataset_info, class_samples, current_fold
                    )
                    metrics.append(fold_metrics)
                    dataset_result[f"sampling_{current_fold}"] = fold_metrics
                    print(f'------------------ fold {current_fold} --------------')
                    print(fold_metrics)
                self.tracker.log_metrics(metrics)
                dataset_result["version"] = self.tracker.version_label(run_obj)
            except Exception as exc:  # pragma: no cover - пример для ручного запуска
                dataset_result["sampling"] = {"error": str(exc)}
            finally:
                self.tracker.end_run()

        final_metrics = {
            k: sum(m[k] for m in metrics) / len(metrics)
            for k in metrics[0].keys()
        }
        dataset_result[f"sampling"] = final_metrics
        return dataset_result
