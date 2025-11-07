# full_experiment.py
import pandas as pd
import numpy as np
import time
import json
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score

from core.repository.constant_repo import AmlbExperimentDataset
from examples.experiments.amlb_dataloader import AMLBDatasetLoader
from examples.experiments.fedot_integration import FedotSamplingEnsemble


class LargeScaleAutoMLExperiment:
    """
    Полный эксперимент по сравнению Fedot + Sampling-Zoo с другими методами
    """

    def __init__(self, results_path: str = "experiment_results"):
        self.results_path = results_path
        self.loader = AMLBDatasetLoader()
        self.results = {}

    def run_fedot_baseline(self, X_train, y_train, X_test, y_test, problem_type):
        """Запуск стандартного Fedot без семплирования"""
        print("Запуск Fedot baseline...")

        start_time = time.time()
        baseline_params = AmlbExperimentDataset.FEDOT_BASELINE_PRESET.value
        baseline_params['problem'] = problem_type
        fedot_model = Fedot(**baseline_params)
        fedot_model.fit(X_train, y_train)
        predictions = fedot_model.predict(X_test)

        training_time = time.time() - start_time
        metrics = self._calculate_metrics(y_test, predictions, problem_type)
        metrics['training_time'] = training_time
        metrics['data_size'] = len(X_train)

        return metrics

    def run_fedot_sampling_ensemble(self, X_train, y_train, X_test, y_test, problem_type):
        """Запуск Fedot с интеллектуальным семплированием"""
        print("Запуск Fedot + Sampling-Zoo ensemble...")

        start_time = time.time()

        # Создаем ансамбль с семплированием
        ensemble = FedotSamplingEnsemble(problem=problem_type,
                                         partitioner_config=AmlbExperimentDataset.SAMPLING_PRESET.value,
                                         fedot_config=AmlbExperimentDataset.FEDOT_PRESET.value,
                                         ensemble_method='weighted'
                                         )

        # Разбиваем данные на партиции
        partitions = ensemble.prepare_data_partitions(X_train, y_train)

        # Обучаем модели на партициях
        ensemble.train_partition_models(partitions)

        # Получаем предсказания ансамбля
        predictions = ensemble.ensemble_predict(X_test)

        training_time = time.time() - start_time
        metrics = self._calculate_metrics(y_test, predictions, problem_type)
        metrics['training_time'] = training_time
        metrics['data_size'] = len(X_train)
        metrics['n_partitions'] = len(ensemble.models)
        metrics['partition_metrics'] = ensemble.partition_metrics

        return metrics, ensemble

    def _calculate_metrics(self, y_true, y_pred, problem_type):
        """Вычисляет метрики качества"""
        if problem_type == 'classification':
            return {
                'accuracy': accuracy_score(y_true, y_pred),
                'f1_macro': f1_score(y_true, y_pred, average='macro'),
                'f1_weighted': f1_score(y_true, y_pred, average='weighted')
            }
        else:  # regression
            return {
                'mse': mean_squared_error(y_true, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                'r2': r2_score(y_true, y_pred)
            }

    def run_experiment_on_dataset(self, dataset_info):
        """Запускает полный эксперимент на одном датасете"""
        print(f"\n{'=' * 50}")
        print(f"ЭКСПЕРИМЕНТ: {dataset_info['name']}")
        print(f"{'=' * 50}")

        # Загружаем данные
        X, y, dataset_info = self.loader.load_dataset(dataset_info)
        if X is None:
            return None

        # Разделяем на train/test
        X_train, X_test, y_train, y_test = self.loader.prepare_train_test(X, y)

        results = {
            'dataset': dataset_info,
            'train_size': len(X_train),
            'test_size': len(X_test)
        }

        # 1. Fedot baseline
        print("\n1. Тестирование Fedot baseline...")
        try:
            baseline_metrics = self.run_fedot_baseline(
                X_train, y_train, X_test, y_test,
                dataset_info['type']
            )
            results['fedot_baseline'] = baseline_metrics
            print(f"   Baseline metrics: {baseline_metrics}")
        except Exception as e:
            print(f"   Ошибка в baseline: {str(e)}")
            results['fedot_baseline'] = {'error': str(e)}

        # 2. Fedot + Sampling-Zoo
        print("\n2. Тестирование Fedot + Sampling-Zoo...")
        try:
            sampling_metrics, ensemble_model = self.run_fedot_sampling_ensemble(
                X_train, y_train, X_test, y_test,
                dataset_info['type']
            )
            results['fedot_sampling'] = sampling_metrics
            print(f"   Sampling ensemble metrics: {sampling_metrics}")
        except Exception as e:
            print(f"   Ошибка в sampling ensemble: {str(e)}")
            results['fedot_sampling'] = {'error': str(e)}

        # 3. Сравнение с AMLB benchmark (заглушка - нужны реальные данные из статьи)
        print("\n3. Сравнение с AMLB benchmark...")
        amlb_comparison = self._compare_with_amlb_benchmark(dataset_info['name'], results)
        results['amlb_comparison'] = amlb_comparison

        return results

    def _compare_with_amlb_benchmark(self, dataset_name, results):
        """Сравнивает результаты с AMLB benchmark"""
        # Здесь должны быть реальные данные из статьи AMLB
        # Пока используем заглушку с ожидаемыми улучшениями

        amlb_baselines = AmlbExperimentDataset.AMLB_EXPERIMENT_RESULTS.value

        comparison = {}
        if dataset_name in amlb_baselines:
            baseline = amlb_baselines[dataset_name]
            our_results = results.get('fedot_sampling', {})

            for metric, amlb_value in baseline.items():
                if metric in our_results:
                    improvement = our_results[metric] - amlb_value
                    comparison[metric] = {
                        'amlb': amlb_value,
                        'our_result': our_results[metric],
                        'improvement': improvement,
                        'improvement_pct': (improvement / amlb_value) * 100
                    }

        return comparison

    def run_full_benchmark(self):
        """Запускает полный бенчмарк на всех датасетах"""
        all_datasets = (self.loader.get_classification_datasets() +
                        self.loader.get_regression_datasets())

        for dataset_info in all_datasets[:3]:  # Начнем с 3 датасетов для теста
            result = self.run_experiment_on_dataset(dataset_info)
            if result:
                self.results[dataset_info['name']] = result

                # Сохраняем промежуточные результаты
                self.save_results()

    def save_results(self):
        """Сохраняет результаты эксперимента"""
        import os
        os.makedirs(self.results_path, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{self.results_path}/experiment_results_{timestamp}.json"

        with open(filename, 'w') as f:
            # Конвертируем numpy types в native Python types для JSON
            def convert_types(obj):
                if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                                    np.int16, np.int32, np.int64, np.uint8,
                                    np.uint16, np.uint32, np.uint64)):
                    return int(obj)
                elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                    return float(obj)
                elif isinstance(obj, (np.ndarray,)):
                    return obj.tolist()
                return obj

            json.dump(self.results, f, indent=2, default=convert_types)

        print(f"Результаты сохранены в {filename}")

    def generate_report(self):
        """Генерирует итоговый отчет"""
        print("\n" + "=" * 70)
        print("ИТОГОВЫЙ ОТЧЕТ ЭКСПЕРИМЕНТА")
        print("=" * 70)

        for dataset_name, result in self.results.items():
            print(f"\n📊 ДАТАСЕТ: {dataset_name}")
            print(f"   Размер данных: {result['train_size']} train, {result['test_size']} test")

            baseline = result.get('fedot_baseline', {})
            sampling = result.get('fedot_sampling', {})

            if 'error' not in baseline and 'error' not in sampling:
                # Сравниваем метрики
                if 'accuracy' in baseline:  # Классификация
                    print(f"   Точность:")
                    print(f"     Baseline: {baseline['accuracy']:.4f}")
                    print(f"     Sampling: {sampling['accuracy']:.4f}")
                    improvement = sampling['accuracy'] - baseline['accuracy']
                    print(f"     Улучшение: {improvement:+.4f}")

                elif 'rmse' in baseline:  # Регрессия
                    print(f"   RMSE:")
                    print(f"     Baseline: {baseline['rmse']:.4f}")
                    print(f"     Sampling: {sampling['rmse']:.4f}")
                    improvement = baseline['rmse'] - sampling['rmse']  # Меньше = лучше
                    print(f"     Улучшение: {improvement:+.4f}")

                # Время обучения
                print(f"   Время обучения:")
                print(f"     Baseline: {baseline['training_time']:.2f} сек")
                print(f"     Sampling: {sampling['training_time']:.2f} сек")
                time_diff = sampling['training_time'] - baseline['training_time']
                print(f"     Разница: {time_diff:+.2f} сек")
