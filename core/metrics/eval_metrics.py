import numpy as np
import operator
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score, log_loss

LOWER_IS_BETTER = {"rmse", "mae", "mse", "logloss", "cross_entropy"}
HIGHER_IS_BETTER = {"accuracy", "f1", "f1_macro", "f1_weighted", "recall", "precision", "roc_auc"}

def calculate_metrics(y_true, y_labels, y_proba, problem_type):
    """Вычисляет метрики качества"""
    if problem_type == 'classification':
        y_pred = y_labels.flatten()
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            #'log_loss': log_loss(y_true=y_true, y_pred=y_proba),
            'f1_macro': f1_score(y_true, y_pred, average='macro'),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted')
        }
    else:  # regression
        y_pred = y_labels
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred)}


def get_metric_comparator(metric_name: str):
    """Возвращает функцию comparator(a, b), которая возвращает True если a лучше b"""
    name = metric_name.lower()

    if name in LOWER_IS_BETTER:
        return operator.lt
    elif name in HIGHER_IS_BETTER:
        return operator.gt
    else:
        raise ValueError(f"Unknown metric type for '{metric_name}'")