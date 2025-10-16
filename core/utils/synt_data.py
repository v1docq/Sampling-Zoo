import sys
import os
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


def create_synt_tabular_data(n_samples=1000, n_features=20, problem_type='classification'):
    """Создает пример табличных данных"""
    if problem_type == 'classification':
        X, y = make_classification(
            n_samples=n_samples, n_features=n_features,
            n_informative=10, n_redundant=5, random_state=42
        )
    else:
        X, y = make_regression(
            n_samples=n_samples, n_features=n_features,
            n_informative=10, random_state=42
        )

    feature_names = [f'feature_{i}' for i in range(n_features)]
    return pd.DataFrame(X, columns=feature_names), y

def create_synt_time_series_data(n_series=10, n_timesteps=1000):
    """Создает пример данных временных рядов"""
    data = []
    for series_id in range(n_series):
        for t in range(n_timesteps):
            data.append({
                'series_id': series_id,
                'timestamp': pd.Timestamp('2023-01-01') + pd.Timedelta(hours=t),
                'value': np.sin(0.1 * t + series_id) + 0.1 * np.random.randn(),
                'feature_1': np.random.randn(),
                'feature_2': np.random.randn()
            })
    return pd.DataFrame(data)