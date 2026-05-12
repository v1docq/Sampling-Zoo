"""Structured expected failures for experiment contract flows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class ExperimentContractError(Exception):
    """Expected experiment-contract failure represented as data."""

    scope: str
    message: str
    code: str = "experiment_contract_error"
    details: Mapping[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.code} at {self.scope}: {self.message}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "scope": self.scope,
            "message": self.message,
            "details": dict(self.details),
        }


class InvalidExperimentConfigError(ExperimentContractError):
    """Invalid or unsupported experiment config payload."""


class UnavailableExperimentDependencyError(ExperimentContractError):
    """Expected optional runtime dependency failure."""


class EmptyExperimentInputError(ExperimentContractError):
    """Expected empty dataset, partition, or model collection failure."""


class ClassificationProbabilitiesRequiredError(ExperimentContractError):
    """Classification benchmark requires class-aligned probability estimates."""

    def __init__(
        self,
        scope: str,
        message: str = "Classification AMLB runs require predict_proba output.",
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(
            scope=scope,
            message=message,
            code="classification_probabilities_required",
            details=details or {},
        )
