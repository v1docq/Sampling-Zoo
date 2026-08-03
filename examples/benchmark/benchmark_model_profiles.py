"""Typed model profiles used by benchmark model factories."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from enum import Enum
import math
from numbers import Real
from typing import Any, Mapping


class TabPFNModelProfile(str, Enum):
    """Canonical TabPFN model names exposed by the benchmark registry."""

    IN_CONTEXT = "tabpfn_in_context"
    FINETUNED = "tabpfn_finetuned"


@dataclass(frozen=True)
class TabPFNFinetuneConfig:
    """Validated subset of the public TabPFN fine-tuning parameters."""

    epochs: int = 30
    time_limit: int | None = None
    learning_rate: float = 1e-5
    validation_split_ratio: float = 0.1
    early_stopping: bool = True
    early_stopping_patience: int = 8
    n_estimators_finetune: int = 2
    n_estimators_validation: int = 2
    n_estimators_final_inference: int = 8
    require_cuda: bool = True

    def __post_init__(self) -> None:
        _require_positive_int("epochs", self.epochs)
        if self.time_limit is not None:
            _require_positive_int("time_limit", self.time_limit)
        _require_positive_real("learning_rate", self.learning_rate)
        _require_open_unit_interval(
            "validation_split_ratio",
            self.validation_split_ratio,
        )
        if not isinstance(self.early_stopping, bool):
            raise ValueError("early_stopping must be boolean")
        _require_positive_int(
            "early_stopping_patience",
            self.early_stopping_patience,
        )
        _require_positive_int(
            "n_estimators_finetune",
            self.n_estimators_finetune,
        )
        _require_positive_int(
            "n_estimators_validation",
            self.n_estimators_validation,
        )
        _require_positive_int(
            "n_estimators_final_inference",
            self.n_estimators_final_inference,
        )
        if not isinstance(self.require_cuda, bool):
            raise ValueError("require_cuda must be boolean")

    def model_kwargs(self, *, device: str, seed: int) -> dict[str, Any]:
        values = asdict(self)
        values.pop("require_cuda")
        return {
            "device": device,
            "random_state": seed,
            **values,
        }


def normalize_tabpfn_finetune_config(
    value: TabPFNFinetuneConfig | Mapping[str, Any] | None,
) -> TabPFNFinetuneConfig:
    if value is None:
        return TabPFNFinetuneConfig()
    if isinstance(value, TabPFNFinetuneConfig):
        return value
    if not isinstance(value, Mapping):
        raise TypeError(
            "tabpfn_finetune_config must be a mapping or "
            "TabPFNFinetuneConfig"
        )

    supported = {field.name for field in fields(TabPFNFinetuneConfig)}
    unknown = sorted(set(value) - supported)
    if unknown:
        raise ValueError(
            "Unsupported TabPFN fine-tuning parameter(s): "
            f"{unknown}"
        )
    return TabPFNFinetuneConfig(**dict(value))


def _require_positive_int(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _require_positive_real(name: str, value: Any) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive finite number")


def _require_open_unit_interval(name: str, value: Any) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        or not 0 < value < 1
    ):
        raise ValueError(f"{name} must be in (0, 1)")
