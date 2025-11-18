"""Инфраструктура для экспериментов AMLB с декомпозицией по модулям."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from fedot import Fedot

from core.metrics.eval_metrics import calculate_metrics
from core.repository.constant_repo import AmlbExperimentDataset
from core.utils.amlb_config import AutoMLModelSpec, ExperimentConfig, ExperimentConfigBuilder, SamplingStrategySpec
from core.utils.amlb_dataloader import AMLBDatasetLoader
from core.utils.amlb_tracking import ExperimentTracker
from core.utils.fedot_integration import FedotSamplingEnsemble


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

    def run_fedot_baseline(self, X_train, y_train, X_test, y_test, problem_type: str) -> Dict:
        params = {**AmlbExperimentDataset.FEDOT_BASELINE_PRESET.value, "problem": problem_type}
        model = Fedot(**params)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        metrics = calculate_metrics(y_test, predictions, problem_type)
        metrics["training_time"] = 0.0
        return metrics

    def run_sampling_ensemble(self, X_train, y_train, X_test, y_test, dataset_info: Dict) -> Tuple[Dict, FedotSamplingEnsemble]:
        sampling_config = {**AmlbExperimentDataset.SAMPLING_PRESET.value}
        strategy: SamplingStrategySpec = self.experiment_config.sampling_strategies[0]
        sampling_config.update(strategy.params)
        sampling_config.setdefault("n_partitions", dataset_info.get("n_partitions", 3))

        ensemble = FedotSamplingEnsemble(
            problem=dataset_info["type"],
            partitioner_config=sampling_config,
            fedot_config=AmlbExperimentDataset.FEDOT_PRESET.value,
            ensemble_method="weighted",
        )

        partitions = ensemble.prepare_data_partitions(X_train, y_train)
        ensemble.train_partition_models(partitions, X_test, y_test)
        predictions = ensemble.ensemble_predict(X_test)

        metrics = calculate_metrics(y_test, predictions, dataset_info["type"])
        metrics["training_time"] = ensemble.training_time
        metrics["n_partitions"] = len(ensemble.models)
        metrics["partition_metrics"] = ensemble.partition_metrics
        return metrics, ensemble

    def run_framework(self, model: AutoMLModelSpec, X_train, y_train, X_test, y_test, dataset_info: Dict):
        if model.name.lower() == "fedot":
            return self.run_sampling_ensemble(X_train, y_train, X_test, y_test, dataset_info)
        raise ValueError(f"Framework {model.name} пока не поддерживается")


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
            sampling = result.get("fedot_sampling", {})
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

        X_train, X_test, y_train, y_test = self.loader.prepare_train_test(X, y)
        dataset_result = {
            "dataset": dataset_info,
            "train_size": len(X_train),
            "test_size": len(X_test),
        }

        run_obj = self.tracker.start_run(
            run_name=dataset_info["name"],
            params={"time_budget": self.config.time_budget_minutes},
        )

        try:
            metrics, ensemble_model = self.runner.run_sampling_ensemble(
                X_train, y_train, X_test, y_test, dataset_info
            )
            dataset_result["fedot_sampling"] = metrics
            self.tracker.log_metrics(metrics)
            dataset_result["version"] = self.tracker.version_label(run_obj)
        except Exception as exc:  # pragma: no cover - пример для ручного запуска
            dataset_result["fedot_sampling"] = {"error": str(exc)}
        finally:
            self.tracker.end_run()

        return dataset_result


__all__ = [
    "ExperimentConfig",
    "ExperimentConfigBuilder",
    "LargeScaleAutoMLExperiment",
]
