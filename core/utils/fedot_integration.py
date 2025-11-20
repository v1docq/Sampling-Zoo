# fedot_sampling_integration.py
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Callable
from fedot.api.main import Fedot
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.repository.tasks import Task, TaskTypesEnum
from tqdm import tqdm

from core.api.api_main import SamplingStrategyFactory
from core.metrics.eval_metrics import calculate_metrics
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
                partitioner.fit(features, target=target)
                self.partitions = partitioner.get_partitions(features, target)
            elif self.partitioner_config['strategy'] in ['stratified']:
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
            self.fedot_config['available_operations'] = AmlbExperimentDataset.FEDOT_MODELS_FOR_CLF.value
            init_assumption = SupportingModels.fedot_reg_pipelines.value['rfr'].build()
        else:
            raise ValueError("Problem type must be 'classification' or 'regression'")
        return task, init_assumption

    def _run_inference(self, fitted_model: Callable, test_data: pd.DataFrame, batch_size: int = None):
        # batch prediction
        predict_labels, predict_proba = [], []
        batch_size = batch_size if batch_size is not None else self.bs_size
        batch_data = [test_data.iloc[i:i + self.bs_size] for i in list(range(0, len(test_data), batch_size))]
        for batch in tqdm(batch_data):
            predict_labels.append(fitted_model.predict(batch))
            predict_proba.append(fitted_model.predict_proba(batch))
        return predict_labels, predict_proba

    def train_partition_models(self, partitions: Dict[str, Dict], X_test, y_test):
        """
        Обучает отдельные Fedot модели на каждой партиции
        """
        task, init_assumption = self._define_fedot_setup()
        for partition_name, partition_data in partitions.items():
            print(f"Обучение модели для поднабора {partition_name}...")

            try:
                # Создаем Fedot модель
                fitted_fedot_model = Fedot(problem=self.problem,
                                           task_params=task.task_params,
                                           initial_assumption=init_assumption,
                                           **self.fedot_config)
                # Обучаем на партиции
                fitted_fedot_model.fit(features=partition_data['feature'], target=partition_data['target'])
                predict_labels, predict_proba = self._run_inference(fitted_fedot_model, X_test)
                # Сохраняем модель и метрики
                metrics = calculate_metrics(y_true=y_test,
                                            problem_type=self.problem,
                                            y_labels=np.concatenate(predict_labels),
                                            y_proba=np.concatenate(predict_proba, axis=0)
                                            )
                model_info = {
                    'name': partition_name,
                    'model': fitted_fedot_model,
                    'data_size': len(partition_data['feature']),
                    'metrics': metrics
                }

                self.models.append(model_info)
                self.partition_metrics[partition_name] = metrics

                print(f"Модель {partition_name} обучена. Размер данных: {model_info['data_size']}")

            except Exception as e:
                print(f"Ошибка при обучении модели {partition_name}: {str(e)}")
                continue

    def ensemble_predict(self, features: pd.DataFrame) -> np.ndarray:
        """
        Ансамблирование предсказаний всех моделей
        """
        if not self.models:
            raise ValueError("Модели не обучены. Сначала вызовите train_partition_models()")

        predictions = []

        for model_info in self.models:
            pred = model_info['model'].predict(features)
            predictions.append(pred)

        # Различные стратегии ансамблирования
        if self.ensemble_method == 'voting':
            # Для классификации - мажоритарное голосование
            if self.problem == 'classification':
                from scipy.stats import mode
                stacked_preds = np.column_stack(predictions)
                final_pred, _ = mode(stacked_preds, axis=1)
                return final_pred.ravel()

            # Для регрессии - усреднение
            elif self.problem == 'regression':
                return np.mean(predictions, axis=0)

        elif self.ensemble_method == 'weighted':
            # Взвешенное голосование на основе качества моделей
            weights = [metrics.get('f1', 0.5) for metrics in self.partition_metrics.values()]
            weights = np.array(weights) / sum(weights)

            if self.problem == 'classification':
                # Для классификации: взвешенное голосование по вероятностям
                proba_predictions = []
                for model_info in self.models:
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

        else:
            raise ValueError(f"Неизвестный метод ансамблирования: {self.ensemble_method}")
