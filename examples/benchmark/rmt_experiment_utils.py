from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

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


def budget_ratio_tag(budget_ratio: float) -> str:
    return f"{int(round(float(budget_ratio) * 100)):02d}"


def markdown_table(
    frame: pd.DataFrame,
    columns: Sequence[str] | None = None,
    *,
    float_digits: int = 6,
) -> str:
    """Render a DataFrame without the optional pandas ``tabulate`` dependency."""

    selected = list(frame.columns if columns is None else columns)
    selected = [column for column in selected if column in frame.columns]
    if frame.empty or not selected:
        return "Нет доступных записей."
    lines = [
        "| " + " | ".join(selected) + " |",
        "|" + "|".join("---" for _ in selected) + "|",
    ]
    for _, row in frame.loc[:, selected].iterrows():
        lines.append(
            "| "
            + " | ".join(
                _format_markdown_value(row[column], float_digits)
                for column in selected
            )
            + " |"
        )
    return "\n".join(lines)


def _format_markdown_value(value: Any, float_digits: int) -> str:
    if value is None or (
        isinstance(value, (float, np.floating)) and np.isnan(value)
    ):
        return "-"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{int(float_digits)}f}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def default_rmt_reference_metrics_path() -> Path:
    return Path(__file__).resolve().parent / "benchmark_metrics" / "AMLB_regression_suite_040526.csv"


def load_reference_metrics(path: Path | None = None) -> pd.DataFrame:
    reference_path = path or default_rmt_reference_metrics_path()
    if not reference_path.exists():
        return pd.DataFrame(columns=["Task", "foundational"])
    return pd.read_csv(reference_path)
