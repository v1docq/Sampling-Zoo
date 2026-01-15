# fedot_sampling_integration.py
import pickle
import os
from scipy.stats import mode
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Callable
from fedot.api.main import Fedot
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.repository.tasks import Task, TaskTypesEnum
from lightgbm import LGBMRegressor, LGBMClassifier
from tqdm import tqdm

from core.api.api_main import SamplingStrategyFactory
from core.metrics.eval_metrics import calculate_metrics, get_metric_comparator
from core.repository.constant_repo import AmlbExperimentDataset
from core.repository.model_repo import SamplingModels, SupportingModels


class FedotSamplingEnsemble:
    """
    Интеграция Sampling-Zoo с Fedot для работы с большими датасетами
    через интеллектуальное семплирование и ансамблирование
    """

    def __init__(self,
                 problem: str,
                 partitioner_config: Dict[str, Any] = None,
                 fedot_config: Dict[str, Any] = None,
                 ensemble_method: str = 'voting'):

        self.problem = problem
        self.partitioner_config = partitioner_config or {
            'strategy': 'feature_clustering',
            'n_clusters': 3,
            'method': 'kmeans'
        }
        self.fedot_config = fedot_config or {
            'timeout': 10,
            'preset': 'best_quality',
            'cv_folds': 3
        }
        self.ensemble_method = ensemble_method
        self.bs_size = 1000
        self.partitions = None
        self.models = []
        self.partition_metrics = {}

    def prepare_data_partitions(self,
                                features: pd.DataFrame,
                                target: pd.Series,
                                random_state: int = 42) -> Dict[str, pd.DataFrame]:
        """
        Разбивает данные на интеллектуальные поднаборы с помощью Sampling-Zoo
        """
        try:
            # Создаем стратегию семплирования
            factory = SamplingStrategyFactory()
            partitioner = factory.create_strategy(
                strategy_type=self.partitioner_config['strategy'],
                n_partitions=self.partitioner_config['n_partitions'],
                random_state=random_state
            )

            # Применяем семплирование
            if self.partitioner_config['strategy'] in ['difficulty', 'uncertainty']:
                difficulty_model_class = LGBMClassifier if self.problem == 'classification' else LGBMRegressor
                difficulty_model = difficulty_model_class(n_estimators=50, n_jobs=-1)
                partitioner.fit(
                    features,
                    target=target,
                    problem=self.problem,
                    model=difficulty_model,
                    chunks_percent=self.partitioner_config['chunks_percent']
                )
                self.partitions = partitioner.get_partitions(features, target)
            elif self.partitioner_config['strategy'].__contains__('stratified'):
                features['target'] = target
                partitioner.fit(data=features, target=features.columns.to_list(), data_target=features['target'])
                self.partitions = partitioner.get_partitions(features, target=features['target'])
                for chunk in self.partitions:
                    del self.partitions[chunk]['feature']['target']
            else:
                partitioner.fit(features)
                self.partitions = partitioner.get_partitions(features, target)
            print(f"Создано {len(self.partitions)} поднаборов данных:")
            print(f"Число семплов в 1 поднаборе -  {len(self.partitions['chunk_0']['feature'])}")
            return self.partitions

        except ImportError:
            raise ImportError("Sampling-Zoo не установлен. Установите его из https://github.com/v1docq/Sampling-Zoo")

    def _define_fedot_setup(self):
        # Определяем тип задачи для Fedot
        if self.problem == 'classification':
            task = Task(TaskTypesEnum.classification)
            self.fedot_config['available_operations'] = AmlbExperimentDataset.FEDOT_MODELS_FOR_CLF.value
            init_assumption = SupportingModels.fedot_clf_pipelines.value['lgbm'].build()
        elif self.problem == 'regression':
            task = Task(TaskTypesEnum.regression)
            # self.fedot_config['available_operations'] = AmlbExperimentDataset.FEDOT_MODELS_FOR_REG.value
            init_assumption = SupportingModels.fedot_reg_pipelines.value['lgbmreg'].build()
        else:
            raise ValueError("Problem type must be 'classification' or 'regression'")
        return task, init_assumption

    def _run_inference(self, fitted_model: Callable, test_data: pd.DataFrame,
                       calculation_mode: str = 'batch', batch_size: int = None):
        if calculation_mode == 'batch':
            predict_labels, predict_proba = [], []
            batch_size = batch_size if batch_size is not None else self.bs_size
            batch_data = [test_data.iloc[i:i + self.bs_size] for i in list(range(0, len(test_data), batch_size))]
            for batch in tqdm(batch_data):
                labels = fitted_model.predict(batch)
                predict_labels.append(labels)
                if self.problem == 'regression':
                    predict_proba.append(labels)
                else:
                    predict_proba.append(fitted_model.predict_proba(batch))
            return np.concatenate(predict_labels), np.concatenate(predict_proba)
        elif calculation_mode == 'non-batch':
            if self.problem == 'regression':
                labels = fitted_model.predict(test_data)
                proba = labels
            else:
                # TODO predict_proba почему то возвращает массив (489838, 4) хотя классов 23
                proba = fitted_model.predict_proba(test_data)
                # proba = fitted_model.predict_proba(test_data)
                # labels = proba.argmax(axis=1)
                labels = fitted_model.predict(test_data)
                proba = None
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

        X_extra = np.stack(X_extra)
        y_extra = np.array(y_extra)

        chunk['feature'] = np.vstack([chunk['feature'], X_extra])
        chunk['target'] = np.hstack([chunk['target'], y_extra])

        return chunk

    def train_partition_models(self, X_train, y_train, X_val, y_val, class_samples, cv_fold):
        """
        Обучает отдельные Fedot модели на каждой партиции
        """
        os.makedirs("dumps", exist_ok=True)
        if 'load_filename' in self.partitioner_config:
            with open(f"dumps/{self.partitioner_config['load_filename']}_{cv_fold}.pkl", "rb") as f:
                partitions = pickle.load(f)
        else:
            partitions = self.prepare_data_partitions(X_train, y_train)
            if 'save_filename' in self.partitioner_config:
                with open(f"dumps/{self.partitioner_config['save_filename']}_{cv_fold}.pkl", "wb") as f:
                    pickle.dump(partitions, f)
        task, init_assumption = self._define_fedot_setup()
        validation_results = []
        best_validation_result, best_validation_result_not_updated = None, 0
        if 'metric' in self.fedot_config:
            validation_metric = self.fedot_config['metric']
            if validation_metric == 'f1':
                validation_metric = 'f1_weighted'
        else:
            validation_metric = 'f1_weighted' if self.problem == 'classification' else 'rmse'
        metric_is_better = get_metric_comparator(validation_metric)
        for partition_name, partition_data in partitions.items():
            print(f"Обучение модели для поднабора {partition_name}...")
            if self.problem == 'classification' and class_samples:
                partition_data = self.ensure_all_classes_in_chunk(partition_data, class_samples)

            try:
                # Создаем Fedot модель
                fitted_fedot_model = Fedot(problem=self.problem,
                                           task_params=task.task_params,
                                           initial_assumption=init_assumption,
                                           **self.fedot_config)
                # Обучаем на поднаборе
                fitted_fedot_model.fit(features=partition_data['feature'], target=partition_data['target'])
                with open(f"dumps/{partition_name}_{cv_fold}.pkl", "wb") as f:
                    pickle.dump(fitted_fedot_model, f)

                # Инференс на валидационных данных
                predict_labels, predict_proba = self._run_inference(fitted_fedot_model, X_val, calculation_mode='non-batch')
                # Сохраняем модель и метрики
                metrics = calculate_metrics(y_true=y_val,
                                            problem_type=self.problem,
                                            y_labels=predict_labels,
                                            y_proba=None
                                            )
                model_info = {
                    'name': partition_name,
                    'model': fitted_fedot_model,
                    'data_size': len(partition_data['feature']),
                    'metrics': metrics,
                    'val_predictions': predict_labels,
                }

                self.models.append(model_info)
                self.partition_metrics[partition_name] = metrics

                print(f"Модель {partition_name} обучена. Размер данных: {model_info['data_size']}")

                predictions = self.ensemble_predict(X_val, stage='validation')
                current_validation_result = calculate_metrics(
                    y_true=y_val, y_labels=predictions, y_proba=None, problem_type=self.problem
                )[validation_metric]
                print(f"Текущая валидационная метрика - {current_validation_result}")
                validation_results.append(current_validation_result)
                if best_validation_result is None or metric_is_better(current_validation_result, best_validation_result):
                    best_validation_result = current_validation_result
                    best_validation_result_not_updated = 0
                else:
                    best_validation_result_not_updated += 1
                if metric_is_better(np.mean(validation_results[:-10]), current_validation_result):
                    del self.models[-1]
                if best_validation_result_not_updated >= 10:
                    break


            except Exception as e:
                print(f"Ошибка при обучении модели {partition_name}: {str(e)}")
                continue

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
            preds = []
            for i in indices:
                mi = self.models[i]
                pred = mi['val_predictions']
                preds.append(pred)

            stacked = np.column_stack(preds)
            final_pred, _ = mode(stacked, axis=1)

            score = calculate_metrics(
                y_true=y_val,
                y_labels=final_pred.ravel(),
                y_proba=None,
                problem_type=self.problem
            )[validation_metric]

            return score

        while remaining:
            best_candidate = None
            best_candidate_score = best_score

            for i in remaining:
                candidate = selected + [i]
                score = evaluate(candidate)

                if best_candidate_score is None or score > best_candidate_score:
                    best_candidate = i
                    best_candidate_score = score

            if best_candidate is None:
                break

            selected.append(best_candidate)
            remaining.remove(best_candidate)
            best_score = best_candidate_score

        self.models = [self.models[i] for i in selected]

        return selected, best_score

    def ensemble_predict(self, features: pd.DataFrame, stage: str = 'inference') -> np.ndarray:
        """
        Ансамблирование предсказаний всех моделей
        """
        if not self.models:
            raise ValueError("Модели не обучены. Сначала вызовите train_partition_models()")

        predictions = []

        for model_info in self.models:
            if stage == 'validation':
                pred = model_info['val_predictions']
            elif stage == 'inference':
                pred = model_info['model'].predict(features.to_numpy() if isinstance(features, pd.DataFrame) else features)
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
            weights = [metrics.get('f1_weighted', 0.5) for metrics in self.partition_metrics.values()]
            weights = np.array(weights) / sum(weights)

            if self.problem == 'classification':
                # Для классификации: взвешенное голосование по вероятностям
                proba_predictions = []
                for model_info in self.models:
                    # Получаем вероятности если доступно
                    try:
                        proba = model_info['model'].predict_proba(features.to_numpy() if isinstance(features, pd.DataFrame) else features)
                        proba_predictions.append(proba)
                    except:
                        # Fallback to hard voting
                        proba_predictions.append(pd.get_dummies(model_info['model'].predict(features)))

                weighted_proba = np.average(proba_predictions, axis=0, weights=weights)
                return np.argmax(weighted_proba, axis=1)

            elif self.problem == 'regression':
                return np.average(predictions, axis=0, weights=weights)

        else:
            raise ValueError(f"Неизвестный метод ансамблирования: {self.ensemble_method}")

class FedotImplementation(FedotSamplingEnsemble):

    def __init__(self,
                 problem: str,
                 partitioner_config: Dict[str, Any] = None,
                 fedot_config: Dict[str, Any] = None):

        self.problem = problem
        self.partitioner_config = partitioner_config or {
            'strategy': 'feature_clustering',
            'n_clusters': 3,
            'method': 'kmeans'
        }
        self.fedot_config = fedot_config or {
            'timeout': 10,
            'preset': 'best_quality',
            'cv_folds': 3
        }
        self.bs_size = 1000
        self.model = None

    def train_model(self, X_train, y_train, X_val, y_val, class_samples, data_percent: float = 0.3):
        partitions = self.prepare_data_partitions(X_train, y_train)
        task, init_assumption = self._define_fedot_setup()
        if 'metric' in self.fedot_config:
            validation_metric = self.fedot_config['metric']
            if validation_metric == 'f1':
                validation_metric = 'f1_weighted'
        else:
            validation_metric = 'f1_weighted' if self.problem == 'classification' else 'rmse'
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

        self.model = Fedot(problem=self.problem,
                           task_params=task.task_params,
                           initial_assumption=init_assumption,
                           **self.fedot_config)
        self.model.fit(features=data, target=target)

        predict_labels, predict_proba = self._run_inference(self.model, X_val, calculation_mode='non-batch')
        metrics = calculate_metrics(y_true=y_val,
                                    problem_type=self.problem,
                                    y_labels=predict_labels,
                                    y_proba=None
                                    )
        print(f"Валидационные метрики - {metrics}")
        validation_result = calculate_metrics(
            y_true=y_val, y_labels=predict_labels, y_proba=predict_proba, problem_type=self.problem
        )[validation_metric]
        print(f"Валидационная метрика - {validation_result}")


    def predict(self, features: pd.DataFrame) -> np.ndarray:
        if not self.model:
            raise ValueError("Модель не обучена. Сначала вызовите train_model()")

        return self.model.predict(features)
