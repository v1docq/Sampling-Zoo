from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def value_series(df: pd.DataFrame, column: str, default: Any = np.nan) -> pd.Series:
    if column in df.columns:
        return df[column]
    return pd.Series([default] * len(df), index=df.index)


def task_key(dataset_name: str) -> str:
    return str(dataset_name).split("__task_")[0]


def default_rmt_reference_metrics_path() -> Path:
    return Path(__file__).resolve().parent / "benchmark_metrics" / "AMLB_regression_suite_040526.csv"


def load_reference_metrics(path: Path | None = None) -> pd.DataFrame:
    reference_path = path or default_rmt_reference_metrics_path()
    if not reference_path.exists():
        return pd.DataFrame(columns=["Task", "foundational"])
    return pd.read_csv(reference_path)
