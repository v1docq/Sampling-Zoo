# fedot_sampling_integration.py
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from fedot.api.main import Fedot
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.repository.tasks import Task, TaskTypesEnum


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
            # Импортируем ваш фреймворк
            from sampling_zoo import SamplingStrategyFactory

            # Создаем стратегию семплирования
            factory = SamplingStrategyFactory()
            partitioner = factory.create_strategy(
                self.partitioner_config['strategy'],
                n_clusters=self.partitioner_config['n_clusters'],
                method=self.partitioner_config.get('method', 'kmeans'),
                random_state=random_state
            )

            # Применяем семплирование
            if self.partitioner_config['strategy'] in ['difficulty', 'uncertainty']:
                partitioner.fit(features, target=target)
            else:
                partitioner.fit(features)

            partitions_indices = partitioner.get_partitions()

            # Создаем разделенные датасеты
            self.partitions = {}
            for partition_name, indices in partitions_indices.items():
                partition_features = features.iloc[indices]
                partition_target = target.iloc[indices]
                self.partitions[partition_name] = {
                    'features': partition_features,
                    'target': partition_target
                }

            print(f"Создано {len(self.partitions)} партиций:")
            for name, data in self.partitions.items():
                print(f"  {name}: {len(data['features'])} samples")

            return self.partitions

        except ImportError:
            raise ImportError("Sampling-Zoo не установлен. Установите его из https://github.com/v1docq/Sampling-Zoo")

    def train_partition_models(self, partitions: Dict[str, Dict]):
        """
        Обучает отдельные Fedot модели на каждой партиции
        """
        self.models = []

        # Определяем тип задачи для Fedot
        if self.problem == 'classification':
            task = Task(TaskTypesEnum.classification)
        elif self.problem == 'regression':
            task = Task(TaskTypesEnum.regression)
        else:
            raise ValueError("Problem type must be 'classification' or 'regression'")

        for partition_name, partition_data in partitions.items():
            print(f"Обучение модели для партиции {partition_name}...")

            try:
                # Создаем Fedot модель
                fedot_model = Fedot(
                    problem=self.problem,
                    task_params=task.task_params,
                    **self.fedot_config
                )

                # Обучаем на партиции
                fedot_model.fit(
                    features=partition_data['features'],
                    target=partition_data['target']
                )

                # Сохраняем модель и метрики
                model_info = {
                    'name': partition_name,
                    'model': fedot_model,
                    'data_size': len(partition_data['features']),
                    'metrics': fedot_model.get_metrics()
                }

                self.models.append(model_info)
                self.partition_metrics[partition_name] = fedot_model.get_metrics()

                print(f"  Модель {partition_name} обучена. Размер данных: {model_info['data_size']}")

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