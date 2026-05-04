# model_integration.py
import pickle
import os
from scipy.stats import mode
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Callable
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from tqdm import tqdm

from sampling_zoo.core.api.api_main import SamplingStrategyFactory
from sampling_zoo.core.metrics.eval_metrics import calculate_metrics, get_metric_comparator

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
                 ensemble_method: str = 'voting'):

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
        self.bs_size = 1000
        self.partitions = None
        self.partitioner = None
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
            strategy_name = self.partitioner_config['strategy']
            reserved_config_keys = {
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
            }
            strategy_kwargs = {
                key: value
                for key, value in self.partitioner_config.items()
                if key not in reserved_config_keys
            }
            strategy_kwargs.setdefault('n_partitions', self.partitioner_config.get('n_partitions', 5))
            strategy_kwargs.setdefault('random_state', random_state)
            if strategy_name == 'feature_clustering':
                allowed_keys = {'n_partitions', 'method', 'feature_engineering', 'random_state'}
                strategy_kwargs = {key: value for key, value in strategy_kwargs.items() if key in allowed_keys}
            elif strategy_name == 'random':
                allowed_keys = {'n_partitions', 'random_state', 'chunks_percent'}
                strategy_kwargs = {key: value for key, value in strategy_kwargs.items() if key in allowed_keys}

            if strategy_name in ['difficulty', 'uncertainty']:
                support_model = (
                    (LGBMClassifier(n_estimators=50, n_jobs=-1, verbosity=-1) if LGBMClassifier is not None
                     else RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=random_state))
                    if self.problem == 'classification'
                    else (LGBMRegressor(n_estimators=50, n_jobs=-1, verbosity=-1) if LGBMRegressor is not None
                          else RandomForestRegressor(n_estimators=50, n_jobs=-1, random_state=random_state))
                )
                strategy_kwargs.update({
                    'problem': self.problem,
                    'model': support_model,
                    'chunks_percent': self.partitioner_config.get('chunks_percent', 100),
                })

            partitioner = factory.create_strategy(
                strategy_type=strategy_name,
                **strategy_kwargs,
            )
            self.partitioner = partitioner

            # Применяем семплирование
            if strategy_name in ['difficulty', 'uncertainty']:
                partitioner.fit(
                    features,
                    target=target,
                )
                self.partitions = partitioner.get_partitions(features, target)
            elif strategy_name.__contains__('stratified'):
                features['target'] = target
                partitioner.fit(data=features, target=features.columns.to_list(), data_target=features['target'])
                self.partitions = partitioner.get_partitions(features, target=features['target'])
                for chunk in self.partitions:
                    del self.partitions[chunk]['feature']['target']
            else:
                partitioner.fit(features)
                self.partitions = partitioner.get_partitions(features, target)
            self.partitions = self._apply_budget_policy_to_partitions(
                partitions=self.partitions,
                total_rows=len(features),
                random_state=random_state,
            )
            print(f"Создано {len(self.partitions)} поднаборов данных:")
            if self.partitions:
                sample_key = next(iter(self.partitions))
                sample_payload = self.partitions[sample_key]
                if isinstance(sample_payload, dict) and "feature" in sample_payload:
                    sample_size = len(sample_payload["feature"])
                else:
                    sample_size = len(sample_payload)
                print(f"Число семплов в 1 поднаборе -  {sample_size}")
            return self.partitions

        except ImportError:
            raise ImportError("Sampling-Zoo не установлен. Установите его из https://github.com/v1docq/Sampling-Zoo")

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
            for batch in tqdm(batch_data):
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
        Обучает отдельные ML модели на каждой партиции
        """
        if save_models_to_disk or 'load_filename' in self.partitioner_config or 'save_filename' in self.partitioner_config:
            os.makedirs("dumps", exist_ok=True)
        if 'load_filename' in self.partitioner_config:
            with open(f"dumps/{self.partitioner_config['load_filename']}_{cv_fold}.pkl", "rb") as f:
                partitions = pickle.load(f)
        else:
            partitions = self.prepare_data_partitions(X_train, y_train)
            if 'save_filename' in self.partitioner_config:
                with open(f"dumps/{self.partitioner_config['save_filename']}_{cv_fold}.pkl", "wb") as f:
                    pickle.dump(partitions, f)

        validation_results = []
        best_validation_result, best_validation_result_not_updated = None, 0

        if validation_metric is None:
            validation_metric = 'f1_weighted' if self.problem == 'classification' else 'rmse'
        elif validation_metric == 'f1':
            validation_metric = 'f1_weighted'

        metric_is_better = get_metric_comparator(validation_metric)

        for partition_name, partition_data in partitions.items():
            print(f"Обучение модели для поднабора {partition_name}...")
            if self.problem == 'classification' and class_samples:
                partition_data = self.ensure_all_classes_in_chunk(partition_data, class_samples)

            try:
                # Создаем экземпляр модели
                model = self._create_model_instance()

                # Обучаем на поднаборе
                model.fit(partition_data['feature'], partition_data['target'])

                if save_models_to_disk:
                    with open(f"dumps/{partition_name}_{cv_fold}.pkl", "wb") as f:
                        pickle.dump(model, f)

                # Инференс на валидационных данных
                predict_labels, predict_proba = self._run_inference(model, X_val, calculation_mode='non-batch')

                # Сохраняем модель и метрики
                metrics = calculate_metrics(
                    y_true=y_val,
                    problem_type=self.problem,
                    y_labels=predict_labels,
                    y_proba=predict_proba if self.problem == "classification" else None,
                )
                model_info = {
                    'name': partition_name,
                    'model': model,
                    'data_size': len(partition_data['feature']),
                    'metrics': metrics,
                    'val_predictions': predict_labels,
                }

                self.models.append(model_info)
                self.partition_metrics[partition_name] = metrics

                print(f"Модель {partition_name} обучена. Размер данных: {model_info['data_size']}")
                print(f"Метрики модели {partition_name}: {metrics}")

                predictions = self.ensemble_predict(X_val, stage='validation')
                ensemble_metrics = calculate_metrics(
                    y_true=y_val,
                    y_labels=predictions,
                    y_proba=None,
                    problem_type=self.problem,
                )
                current_validation_result = ensemble_metrics[validation_metric]
                print(f"Метрики ансамбля после {partition_name}: {ensemble_metrics}")
                print(f"Текущая валидационная метрика - {current_validation_result}")
                validation_results.append(current_validation_result)

                if best_validation_result is None or metric_is_better(current_validation_result, best_validation_result):
                    best_validation_result = current_validation_result
                    best_validation_result_not_updated = 0
                else:
                    best_validation_result_not_updated += 1

                if not train_all_chunks:
                    if len(validation_results) > 10 and metric_is_better(np.mean(validation_results[:-10]), current_validation_result):
                        del self.models[-1]
                    if best_validation_result_not_updated >= 10:
                        break

            except Exception as e:
                print(f"Ошибка при обучении модели {partition_name}: {str(e)}")
                continue

        if self.models:
            full_metrics = calculate_metrics(
                y_true=y_val,
                y_labels=self.ensemble_predict(X_val, stage='validation'),
                y_proba=None,
                problem_type=self.problem,
            )
            print(f"Метрики ансамбля до сокращения: {full_metrics}")

            selected, best_score = self.select_best_models_forward(
                X_val=X_val,
                y_val=y_val,
                metric_is_better=metric_is_better,
                validation_metric=validation_metric,
            )

            reduced_metrics = calculate_metrics(
                y_true=y_val,
                y_labels=self.ensemble_predict(X_val, stage='validation'),
                y_proba=None,
                problem_type=self.problem,
            )
            print(f"Метрики ансамбля после сокращения: {reduced_metrics}")
            print(f"Лучшая валидационная метрика после сокращения ({validation_metric}): {best_score}")

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
            print(f"Forward selection: models={len(selected)} {validation_metric}={best_score}")

        self.models = [self.models[i] for i in selected]

        return selected, best_score

    def _validation_weights(self, active_models: List[Dict[str, Any]]) -> np.ndarray:
        """
        Converts validation metrics into non-negative model priors.
        For regression lower RMSE/MAE is better; for classification higher F1/accuracy is better.
        """
        raw_weights = []
        eps = 1e-8
        for model_info in active_models:
            metrics = model_info.get('metrics', {}) or {}
            if self.problem == 'regression':
                if 'rmse' in metrics and np.isfinite(metrics['rmse']):
                    raw_weights.append(1.0 / (float(metrics['rmse']) + eps))
                elif 'mae' in metrics and np.isfinite(metrics['mae']):
                    raw_weights.append(1.0 / (float(metrics['mae']) + eps))
                else:
                    raw_weights.append(1.0)
            else:
                value = metrics.get('f1_weighted', metrics.get('accuracy', 1.0))
                raw_weights.append(max(float(value), eps) if np.isfinite(value) else 1.0)

        weights = np.asarray(raw_weights, dtype=float)
        if not np.all(np.isfinite(weights)) or weights.sum() <= 0:
            weights = np.ones(len(active_models), dtype=float)
        return weights / weights.sum()

    def _routing_weights(self, features: pd.DataFrame, active_models: List[Dict[str, Any]]) -> np.ndarray:
        """
        Returns row-wise routing probabilities aligned with active_models.
        If the sampler cannot route new points, falls back to uniform routing.
        """
        n_samples = len(features)
        n_models = len(active_models)
        if self.partitioner is None:
            return np.full((n_samples, n_models), 1.0 / max(n_models, 1))

        model_names = [model_info.get('name') for model_info in active_models]
        try:
            if hasattr(self.partitioner, 'predict_partition_proba'):
                proba = np.asarray(self.partitioner.predict_partition_proba(features), dtype=float)
                partition_names = list(getattr(self.partitioner, 'partition_names_', []))
                if partition_names and proba.shape[1] == len(partition_names):
                    name_to_col = {name: idx for idx, name in enumerate(partition_names)}
                    aligned = np.zeros((proba.shape[0], n_models), dtype=float)
                    for model_idx, name in enumerate(model_names):
                        if name in name_to_col:
                            aligned[:, model_idx] = proba[:, name_to_col[name]]
                    if aligned.sum() > 0:
                        row_sums = aligned.sum(axis=1, keepdims=True)
                        aligned = np.where(row_sums > 0, aligned / row_sums, 1.0 / n_models)
                        return aligned

            if hasattr(self.partitioner, 'predict_partitions'):
                labels = np.asarray(self.partitioner.predict_partitions(features))
                aligned = np.full((labels.shape[0], n_models), 0.0, dtype=float)
                for model_idx, name in enumerate(model_names):
                    try:
                        label_id = int(str(name).split('_')[-1])
                    except Exception:
                        label_id = model_idx
                    aligned[:, model_idx] = (labels == label_id).astype(float)
                row_sums = aligned.sum(axis=1, keepdims=True)
                aligned = np.where(row_sums > 0, aligned / row_sums, 1.0 / n_models)
                return aligned
        except Exception:
            pass

        return np.full((n_samples, n_models), 1.0 / max(n_models, 1))

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
        for batch_idx in range(total_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, n_samples)
            remaining = total_batches - batch_idx
            print(f"Batch {batch_idx + 1}/{total_batches} (remaining: {remaining - 1})")
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
        print(f"Валидационные метрики - {metrics}")
        validation_result = metrics[validation_metric]
        print(f"Валидационная метрика ({validation_metric}) - {validation_result}")

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Предсказание на новых данных"""
        if not self.model:
            raise ValueError("Модель не обучена. Сначала вызовите train_model()")

        return self.model.predict(features)
