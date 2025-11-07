import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix

from core.repository.constant_repo import SyntDataset

def create_distrib_dataset(n_samples: int = 10000):
    # Параметры распределений

    mu1, sigma1 = 5, 2  # параметры для первого признака
    mu2, sigma2 = -3, 1.5  # параметры для второго признака

    # Создание данных с заданными параметрами
    data = pd.DataFrame({
        'feature_1': np.random.normal(mu1, sigma1, n_samples),
        'feature_2': np.random.normal(mu2, sigma2, n_samples),
        'target': np.random.randint(0, 2, n_samples)
    })

    # Вывод исходных статистик
    print("Original data statistics:")
    print("Feature 1 (μ={}, σ={}):".format(mu1, sigma1))
    print("  Mean: {:.3f}, Std: {:.3f}, Var: {:.3f}".format(
        data['feature_1'].mean(),
        data['feature_1'].std(),
        data['feature_1'].var()
    ))
    print("Feature 2 (μ={}, σ={}):".format(mu2, sigma2))
    print("  Mean: {:.3f}, Std: {:.3f}, Var: {:.3f}\n".format(
        data['feature_2'].mean(),
        data['feature_2'].std(),
        data['feature_2'].var()
    ))

    return data
def create_sklearn_dataset(taks_type: str,
                           dataset_params: dict = None):
    dataset_params = SyntDataset.DATASET_DEFAULT_PARAMS.value[taks_type] if dataset_params is None else dataset_params
    X, y = SyntDataset.DATASET_GENERATORS.value[taks_type](**dataset_params)
    data = pd.DataFrame(X, columns=['feature_1', 'feature_2'])
    data['target'] = y
    return data


def create_noisy_dataset(n_samples: int = 100):
    # Создание данных: линейная комбинация признаков задаёт вероятность класса
    np.random.seed(42)
    feature_1 = np.random.randn(n_samples)
    feature_2 = np.random.randn(n_samples) * 0.5
    # линейная смесь + шум
    logit = 2.0 * feature_1 - 1.0 * feature_2 + 0.2 * np.random.randn(n_samples)
    prob = 1 / (1 + np.exp(-logit))
    target = (prob > 0.5).astype(int)
    data = pd.DataFrame({'feature_1': feature_1, 'feature_2': feature_2, 'target': target})
    return data


def confusion_matrix_analysis(partitions, predictions):
    # Анализ ошибок по каждому разделу (partition)
    for name, part in partitions.items():
        idx = part['feature'].index.to_numpy()
        y_true = np.asarray(part['target'])
        y_pred = predictions[idx]
        label = name

        if idx.size == 0:
            print(f"{label}: 0 samples (skipped)")
            continue

        errors = (y_true != y_pred).sum()
        total = len(idx)
        pct = errors / total
        cm = confusion_matrix(y_true, y_pred)
        print(f"{label}: {errors} errors out of {total} ({pct:.2%})")
        print("Confusion matrix for", label)
        print(cm)
