"""Pure contracts and fitting for budget-quality scaling laws."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np
from scipy.optimize import least_squares


@dataclass(frozen=True)
class ScalingLawFitSpec:
    min_points: int = 4
    exponent_bounds: Tuple[float, float] = (0.01, 3.0)
    bootstrap_iterations: int = 500
    degradation_levels: Tuple[float, ...] = (0.01, 0.03, 0.05)
    random_state: int = 42

    def __post_init__(self) -> None:
        if int(self.min_points) < 3:
            raise ValueError("min_points must be at least 3")
        lower, upper = (float(value) for value in self.exponent_bounds)
        if not 0.0 < lower < upper:
            raise ValueError("exponent_bounds must be positive and increasing")
        if int(self.bootstrap_iterations) < 0:
            raise ValueError("bootstrap_iterations must be non-negative")
        levels = tuple(float(value) for value in self.degradation_levels)
        if not levels or any(not 0.0 < value < 1.0 for value in levels):
            raise ValueError("degradation_levels must contain values in (0, 1)")
        object.__setattr__(self, "exponent_bounds", (lower, upper))
        object.__setattr__(self, "degradation_levels", levels)


@dataclass(frozen=True)
class ScalingLawObservation:
    budget_ratio: float
    primary_value: float
    primary_metric: str
    replicate_id: str
    is_full_reference: bool = False

    def __post_init__(self) -> None:
        if not 0.0 < float(self.budget_ratio) <= 1.0:
            raise ValueError("budget_ratio must be in (0, 1]")
        if not np.isfinite(float(self.primary_value)):
            raise ValueError("primary_value must be finite")
        if not str(self.primary_metric).strip() or not str(self.replicate_id).strip():
            raise ValueError("primary_metric and replicate_id must be non-empty")

    @property
    def loss(self) -> float:
        return primary_value_to_loss(self.primary_metric, self.primary_value)


@dataclass(frozen=True)
class ScalingLawFitResult:
    status: str
    primary_metric: str
    asymptotic_loss: float
    scale: float
    exponent: float
    exponent_at_boundary: bool
    evidence_status: str
    exponent_confidence_interval: Tuple[float, float] | None
    r_squared: float
    mean_absolute_error: float
    reference_primary_value: float
    reference_loss: float
    required_budget_by_degradation: Tuple[Tuple[float, float | None], ...]
    n_budget_points: int
    n_observations: int

    def predict_loss(self, budget_ratio: float) -> float:
        budget = float(budget_ratio)
        if not 0.0 < budget <= 1.0:
            raise ValueError("budget_ratio must be in (0, 1]")
        return float(self.asymptotic_loss + self.scale * budget ** (-self.exponent))

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "primary_metric": self.primary_metric,
            "asymptotic_loss": self.asymptotic_loss,
            "scale": self.scale,
            "exponent": self.exponent,
            "exponent_at_boundary": self.exponent_at_boundary,
            "evidence_status": self.evidence_status,
            "exponent_confidence_interval": self.exponent_confidence_interval,
            "r_squared": self.r_squared,
            "mean_absolute_error": self.mean_absolute_error,
            "reference_primary_value": self.reference_primary_value,
            "reference_loss": self.reference_loss,
            "required_budget_by_degradation": {
                f"{int(level * 100)}pct": budget
                for level, budget in self.required_budget_by_degradation
            },
            "n_budget_points": self.n_budget_points,
            "n_observations": self.n_observations,
        }


class BudgetScalingLawFitter:
    """Fit ``loss(b) = floor + scale * b**(-exponent)``."""

    def __init__(self, spec: ScalingLawFitSpec | None = None) -> None:
        self.spec = spec or ScalingLawFitSpec()

    def fit(
        self,
        observations: Sequence[ScalingLawObservation],
    ) -> ScalingLawFitResult:
        values = tuple(observations)
        if not values:
            raise ValueError("observations must be non-empty")
        metrics = {item.primary_metric for item in values}
        if len(metrics) != 1:
            raise ValueError("all scaling-law observations must use one metric")
        primary_metric = next(iter(metrics))
        budget, loss = self._aggregate(values)
        if budget.size < self.spec.min_points:
            raise ValueError(
                f"at least {self.spec.min_points} distinct budget points are required"
            )
        reference_values = [item for item in values if item.is_full_reference]
        if not reference_values:
            raise ValueError("at least one full-reference observation is required")
        reference_primary = float(
            np.median([item.primary_value for item in reference_values])
        )
        reference_loss = primary_value_to_loss(primary_metric, reference_primary)
        parameters = self._fit_parameters(budget, loss, reference_loss)
        predicted = self._curve(budget, parameters)
        residual = loss - predicted
        denominator = float(np.sum((loss - np.mean(loss)) ** 2))
        r_squared = (
            1.0 - float(np.sum(residual**2)) / denominator
            if denominator > np.finfo(float).eps
            else 1.0
        )
        exponent_interval = self._bootstrap_exponent(values)
        required = tuple(
            (
                level,
                self._required_budget(
                    parameters,
                    primary_metric=primary_metric,
                    reference_loss=reference_loss,
                    degradation=level,
                ),
            )
            for level in self.spec.degradation_levels
        )
        exponent_at_boundary = self._exponent_at_boundary(parameters[2])
        evidence_status = self._evidence_status(
            n_budget_points=int(budget.size),
            r_squared=float(r_squared),
            exponent_at_boundary=exponent_at_boundary,
        )
        return ScalingLawFitResult(
            status="fitted",
            primary_metric=primary_metric,
            asymptotic_loss=float(parameters[0]),
            scale=float(parameters[1]),
            exponent=float(parameters[2]),
            exponent_at_boundary=exponent_at_boundary,
            evidence_status=evidence_status,
            exponent_confidence_interval=exponent_interval,
            r_squared=float(r_squared),
            mean_absolute_error=float(np.mean(np.abs(residual))),
            reference_primary_value=reference_primary,
            reference_loss=float(reference_loss),
            required_budget_by_degradation=required,
            n_budget_points=int(budget.size),
            n_observations=len(values),
        )

    @staticmethod
    def _aggregate(
        observations: Sequence[ScalingLawObservation],
    ) -> tuple[np.ndarray, np.ndarray]:
        budgets = sorted({float(item.budget_ratio) for item in observations})
        losses = [
            np.median(
                [
                    item.loss
                    for item in observations
                    if float(item.budget_ratio) == budget
                ]
            )
            for budget in budgets
        ]
        return np.asarray(budgets, dtype=float), np.asarray(losses, dtype=float)

    def _fit_parameters(
        self,
        budget: np.ndarray,
        loss: np.ndarray,
        reference_loss: float,
    ) -> np.ndarray:
        initial_scale = max(float(np.max(loss) - reference_loss), 1e-8)
        result = least_squares(
            lambda free_parameters: self._curve(
                budget,
                self._anchored_parameters(reference_loss, free_parameters),
            )
            - loss,
            x0=np.asarray([initial_scale, 0.5], dtype=float),
            bounds=(
                np.asarray(
                    [0.0, self.spec.exponent_bounds[0]],
                    dtype=float,
                ),
                np.asarray(
                    [np.inf, self.spec.exponent_bounds[1]],
                    dtype=float,
                ),
            ),
            loss="soft_l1",
        )
        if not result.success or not np.all(np.isfinite(result.x)):
            raise RuntimeError("scaling-law optimization did not converge")
        return self._anchored_parameters(reference_loss, result.x)

    @staticmethod
    def _anchored_parameters(
        reference_loss: float,
        free_parameters: np.ndarray,
    ) -> np.ndarray:
        scale, exponent = (float(value) for value in free_parameters)
        return np.asarray(
            [float(reference_loss) - scale, scale, exponent],
            dtype=float,
        )

    def _exponent_at_boundary(self, exponent: float) -> bool:
        lower, upper = self.spec.exponent_bounds
        tolerance = 1e-5 * max(1.0, abs(lower), abs(upper))
        return bool(
            abs(float(exponent) - lower) <= tolerance
            or abs(float(exponent) - upper) <= tolerance
        )

    @staticmethod
    def _evidence_status(
        *,
        n_budget_points: int,
        r_squared: float,
        exponent_at_boundary: bool,
    ) -> str:
        if exponent_at_boundary:
            return "boundary_limited"
        if n_budget_points < 6:
            return "exploratory_sparse_grid"
        if r_squared < 0.80:
            return "poor_fit"
        return "supported"

    @staticmethod
    def _curve(budget: np.ndarray, parameters: np.ndarray) -> np.ndarray:
        floor, scale, exponent = parameters
        return floor + scale * np.power(budget, -exponent)

    def _bootstrap_exponent(
        self,
        observations: Sequence[ScalingLawObservation],
    ) -> Tuple[float, float] | None:
        if self.spec.bootstrap_iterations == 0:
            return None
        replicate_ids = sorted({item.replicate_id for item in observations})
        if len(replicate_ids) < 2:
            return None
        by_replicate = {
            replicate: tuple(
                item for item in observations if item.replicate_id == replicate
            )
            for replicate in replicate_ids
        }
        rng = np.random.default_rng(self.spec.random_state)
        exponents = []
        for _ in range(self.spec.bootstrap_iterations):
            sampled = rng.choice(replicate_ids, size=len(replicate_ids), replace=True)
            bootstrap_values = tuple(
                item
                for replicate in sampled
                for item in by_replicate[str(replicate)]
            )
            budget, loss = self._aggregate(bootstrap_values)
            if budget.size < self.spec.min_points:
                continue
            try:
                reference = float(
                    np.median(
                        [item.loss for item in bootstrap_values if item.is_full_reference]
                    )
                )
                exponents.append(
                    float(self._fit_parameters(budget, loss, reference)[2])
                )
            except RuntimeError:
                continue
        if len(exponents) < max(20, self.spec.bootstrap_iterations // 10):
            return None
        lower, upper = np.quantile(exponents, (0.025, 0.975))
        return float(lower), float(upper)

    @staticmethod
    def _required_budget(
        parameters: np.ndarray,
        *,
        primary_metric: str,
        reference_loss: float,
        degradation: float,
    ) -> float | None:
        floor, scale, exponent = (float(value) for value in parameters)
        if primary_metric == "roc_auc":
            target_loss = reference_loss + float(degradation)
        else:
            target_loss = reference_loss * (1.0 + float(degradation))
        available = target_loss - floor
        if available <= 0.0 or scale <= 0.0:
            return None
        budget = (scale / available) ** (1.0 / exponent)
        if not np.isfinite(budget) or budget > 1.0:
            return None
        return float(max(budget, np.finfo(float).eps))


def primary_value_to_loss(metric: str, value: float) -> float:
    normalized = str(metric).strip().lower()
    primary = float(value)
    if normalized == "roc_auc":
        return float(1.0 - primary)
    if normalized in {"rmse", "log_loss"}:
        return primary
    raise ValueError("scaling laws support rmse, roc_auc, or log_loss")
