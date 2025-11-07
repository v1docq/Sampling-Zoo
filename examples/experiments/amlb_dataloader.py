# amlb_datasets.py
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from core.repository.constant_repo import AmlbExperimentDataset


class AMLBDatasetLoader:
    """
    Загрузчик датасетов из AMLB Benchmark
    """

    @staticmethod
    def get_classification_datasets():
        """Классификация - наибольшее число samples"""
        return AmlbExperimentDataset.CLF_DATASET.value

    @staticmethod
    def get_regression_datasets():
        """Регрессия - наибольшее число samples"""
        return AmlbExperimentDataset.REG_DATASET.value

    def load_dataset(self, dataset_info):
        """Загружает датасет по его описанию"""
        try:
            X, y = fetch_openml(data_id=dataset_info['openml_id'],
                                return_X_y=True, as_frame=True)

            # Предобработка
            if dataset_info['type'] == 'classification':
                # Кодируем целевой признак для классификации
                le = LabelEncoder()
                y = le.fit_transform(y)

            # Обработка пропущенных значений
            if isinstance(X, pd.DataFrame):
                X = X.fillna(X.mean(numeric_only=True))

            print(f"Загружен датасет {dataset_info['name']}: {X.shape[0]} samples, {X.shape[1]} features")

            return X, y, dataset_info

        except Exception as e:
            print(f"Ошибка при загрузке {dataset_info['name']}: {str(e)}")
            return None, None, None

    def prepare_train_test(self, X, y, test_size=0.2, random_state=42):
        """Разделяет данные на обучающую и тестовую выборки"""
        return train_test_split(X, y, test_size=test_size,
                                random_state=random_state,
                                stratify=y if len(np.unique(y)) < 100 else None)