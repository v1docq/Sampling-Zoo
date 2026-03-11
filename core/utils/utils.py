import pandas as pd
import numpy as np

def safe_index(data, idx):
    """
    Универсальное индексирование для pandas и numpy
    """
    if isinstance(data, (pd.DataFrame, pd.Series)):
        return data.iloc[idx]
    elif isinstance(data, np.ndarray):
        return data[idx]
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")


def to_numpy(data) -> np.ndarray:
    """
    Безопасно приводит pandas/array-like данные к numpy.
    """
    if isinstance(data, (pd.DataFrame, pd.Series)):
        return data.to_numpy()
    return np.asarray(data)


def to_dataframe(data, columns=None, index=None) -> pd.DataFrame:
    """
    Безопасно приводит array-like данные к DataFrame.
    """
    if isinstance(data, pd.DataFrame):
        return data
    if isinstance(data, pd.Series):
        return data.to_frame()
    return pd.DataFrame(data, columns=columns, index=index)


def to_series(data, index=None, name=None) -> pd.Series:
    """
    Безопасно приводит array-like данные к Series.
    """
    if isinstance(data, pd.Series):
        return data
    return pd.Series(data, index=index, name=name)
