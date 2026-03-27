# amlb_datasets.py
import pandas as pd
import numpy as np
from pathlib import Path
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
    @staticmethod
    def get_custom_datasets():
        """Регрессия - наибольшее число samples"""
        return AmlbExperimentDataset.AMLB_CUSTOM_DATASET.value

    @staticmethod
    def resolve_dataset_path(raw_path: str) -> Path:
        candidate = Path(raw_path)
        root_dir = Path(__file__).resolve().parents[2]

        search_paths = []
        if candidate.is_absolute():
            search_paths.append(candidate)
        else:
            search_paths.extend(
                [
                    Path.cwd() / candidate,
                    root_dir / candidate,
                    root_dir / "examples" / "api_example" / candidate.name,
                    root_dir / "examples" / "api_example" / "dataset" / candidate.name,
                ]
            )

        for path in search_paths:
            if path.exists():
                return path.resolve()

        raise FileNotFoundError(f"Dataset path not found for '{raw_path}'")

    def load_dataset(self, dataset_info, as_frame: bool = False, preserve_categorical: bool = False):
        """Загружает датасет по его описанию"""

        try:
            if 'path' in dataset_info.keys():
                resolved_path = self.resolve_dataset_path(dataset_info['path'])
                df = pd.read_csv(resolved_path)
                # df = df.drop_duplicates()
                y = df[dataset_info['target']]
                del df[dataset_info['target']]
                X = df
            else:
                X, y = fetch_openml(data_id=dataset_info['openml_id'],
                                    return_X_y=True, as_frame=True)

            # Предобработка
            if dataset_info['type'] == 'classification':
                # Кодируем целевой признак для классификации
                le = LabelEncoder()
                y = le.fit_transform(y)
            else:
                y = y.to_numpy()

            # Обработка пропущенных значений
            if isinstance(X, pd.DataFrame):
                numeric_columns = X.select_dtypes(include=['number', 'bool']).columns.tolist()
                categorical_columns = [col for col in X.columns if col not in numeric_columns]

                if numeric_columns:
                    X[numeric_columns] = X[numeric_columns].apply(pd.to_numeric, errors='coerce')
                    X[numeric_columns] = X[numeric_columns].fillna(X[numeric_columns].median(numeric_only=True))

                if preserve_categorical:
                    for col in categorical_columns:
                        X[col] = X[col].astype('string').fillna('__missing__').astype('category')
                else:
                    X[categorical_columns] = X[categorical_columns].fillna('__missing__')

                if not as_frame:
                    X = X.to_numpy()

            if as_frame:
                if not isinstance(X, pd.DataFrame):
                    X = pd.DataFrame(X)
                if not isinstance(y, pd.Series):
                    y = pd.Series(y, name=dataset_info.get('target', 'target'))
                else:
                    y = y.reset_index(drop=True)
                X = X.reset_index(drop=True)

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

    @staticmethod
    def prepare_train_val_test_balanced(
        X, y, test_size=0.1, val_size=0.1, min_samples=20, problem='regression', random_state=42
    ):
        if problem == 'regression':
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y, test_size=test_size+val_size, random_state=random_state,
            )
            if val_size == 0:
                X_val, X_test, y_val, y_test = None, X_temp, None, y_temp
            else:
                X_val, X_test, y_val, y_test = train_test_split(
                    X_temp, y_temp, test_size=test_size / (test_size + val_size), random_state=random_state,
                )
            return X_train, X_val, X_test, y_train, y_val, y_test

        classes, counts = np.unique(y, return_counts=True)
        rare_classes = classes[counts < min_samples]
        common_classes = classes[counts >= min_samples]

        rare_mask = np.isin(y, rare_classes)
        common_mask = np.isin(y, common_classes)

        X_rare, y_rare = X[rare_mask], y[rare_mask]
        X_common, y_common = X[common_mask], y[common_mask]

        X_train, X_temp, y_train, y_temp = train_test_split(
            X_common, y_common, test_size=test_size+val_size, random_state=random_state, stratify=y_common
        )

        if val_size == 0:
            X_val, X_test, y_val, y_test = None, X_temp, None, y_temp
        else:
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=test_size / (test_size + val_size), random_state=random_state,
                stratify=y_temp
            )

        for cls in rare_classes:
            cls_indices = np.where(y_rare == cls)[0]
            if len(cls_indices) > 0:
                # Первый пример идет в тест
                idx_test = cls_indices[0]
                X_test = np.vstack([X_test, X_rare[idx_test:idx_test + 1]])
                y_test = np.hstack([y_test, y_rare[idx_test:idx_test + 1]])

                # Остальные в train
                if len(cls_indices) > 1:
                    X_train = np.vstack([X_train, X_rare.iloc[cls_indices[1:]]])
                    y_train = np.hstack([y_train, y_rare.iloc[cls_indices[1:]]])

        return X_train, X_val, X_test, y_train, y_val, y_test

    @staticmethod
    def select_one_sample_per_class(X, y, random_state=42):
        rng = np.random.default_rng(random_state)
        unique_classes = np.unique(y)

        class_samples = {}

        for cls in unique_classes:
            indices = np.where(y == cls)[0]
            idx = rng.choice(indices)
            class_samples[cls] = (X[idx], y[idx])

        return class_samples
