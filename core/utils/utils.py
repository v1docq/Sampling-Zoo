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