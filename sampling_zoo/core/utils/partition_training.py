"""Pure transforms from sampled partitions to model-training partitions."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from sampling_zoo.core.experiment.contracts import PartitionModelMode


def normalize_partition_model_mode(value: Any) -> PartitionModelMode:
    normalized = str(
        value or PartitionModelMode.INDEPENDENT.value
    ).strip().lower()
    try:
        return PartitionModelMode(normalized)
    except ValueError as exc:
        supported = ", ".join(mode.value for mode in PartitionModelMode)
        raise ValueError(
            f"partition_model_mode must be one of: {supported}"
        ) from exc


def build_model_training_partitions(
    partitions: Mapping[str, Any],
    mode: PartitionModelMode,
) -> dict[str, Any]:
    if mode is PartitionModelMode.INDEPENDENT:
        return dict(partitions)
    if not partitions:
        raise ValueError("Cannot concatenate empty partitions")

    partition_values = tuple(partitions.values())
    if not all(
        isinstance(partition, Mapping)
        and "feature" in partition
        and "target" in partition
        for partition in partition_values
    ):
        raise ValueError(
            "Concatenated partition training requires feature/target mappings"
        )
    concatenated = {
            "feature": _concatenate_values(
                tuple(partition["feature"] for partition in partition_values)
            ),
            "target": _concatenate_values(
                tuple(partition["target"] for partition in partition_values)
            ),
    }
    if all("sample_weight" in partition for partition in partition_values):
        concatenated["sample_weight"] = _concatenate_values(
            tuple(partition["sample_weight"] for partition in partition_values)
        )
    return {"concatenated_budget": concatenated}


def _concatenate_values(values: tuple[Any, ...]) -> Any:
    if all(isinstance(value, pd.DataFrame) for value in values):
        return pd.concat(values, ignore_index=True)
    if all(isinstance(value, pd.Series) for value in values):
        return pd.concat(values, ignore_index=True)
    return np.concatenate(tuple(np.asarray(value) for value in values), axis=0)
