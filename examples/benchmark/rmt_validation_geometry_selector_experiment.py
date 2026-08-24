"""Phase A.1 benchmark for validation-selected RMT routing geometry."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Optional, Sequence

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT_DIR = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = Path(__file__).resolve().parent
for module_path in (ROOT_DIR, BENCHMARK_DIR):
    if str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))

from rmt_routing_geometry_experiment import (
    DEFAULT_ROUTING_BUDGETS,
    DEFAULT_ROUTING_CLASSIFICATION_TASKS,
    DEFAULT_ROUTING_REGRESSION_TASKS,
    DEFAULT_ROUTING_SEEDS,
    PreparedRoutingDataset,
    RoutingGeometryExperimentConfig,
    RoutingGeometryExperimentOrchestrator,
    RoutingOuterSplit,
)
from routing_geometry_replay import (
    FittedEnsembleRoutingReplay,
    RoutingGeometryArm,
    default_validation_selector_arms,
    default_validation_selector_policy,
)
from sampling_zoo.core.experiment.routing_replay import (
    RoutingGeometrySelectionPolicy,
    RoutingReplayResult,
    ValidationRoutingGeometrySelector,
)


DEFAULT_SELECTOR_ARM_NAME = "A7_validation_selected_top3"


@dataclass(frozen=True)
class RoutingGeometrySelectorExperimentConfig(RoutingGeometryExperimentConfig):
    """Selector config with independent calibration and geometry-selection rows."""

    selector_fraction: float = 0.50
    selector_arm_name: str = DEFAULT_SELECTOR_ARM_NAME
    minimum_improvement: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if not 0.0 < float(self.selector_fraction) < 1.0:
            raise ValueError("selector_fraction must be in (0, 1)")
        if not str(self.selector_arm_name).strip():
            raise ValueError("selector_arm_name must be non-empty")
        if float(self.minimum_improvement) < 0:
            raise ValueError("minimum_improvement must be non-negative")


@dataclass(frozen=True)
class PreparedRoutingSelectorDataset(PreparedRoutingDataset):
    X_selection: pd.DataFrame
    y_selection: pd.Series


class ValidationRoutingGeometryExperimentOrchestrator(
    RoutingGeometryExperimentOrchestrator
):
    """Fit experts once, calibrate temperatures, select geometry, then test."""

    config: RoutingGeometrySelectorExperimentConfig

    def __init__(
        self,
        config: RoutingGeometrySelectorExperimentConfig,
        *,
        arms: Optional[Sequence[RoutingGeometryArm]] = None,
        selector: Optional[ValidationRoutingGeometrySelector] = None,
    ) -> None:
        selected_arms = tuple(arms or default_validation_selector_arms())
        super().__init__(config, arms=selected_arms)
        self.selector = selector or ValidationRoutingGeometrySelector()
        self.selection_policy = default_validation_selector_policy(
            minimum_improvement=float(config.minimum_improvement)
        )
        self._validate_selector_arms(self.selection_policy)

    def _prepare_dataset(
        self,
        dataset,
        seed: int,
        *,
        outer_split: RoutingOuterSplit,
    ) -> PreparedRoutingSelectorDataset:
        prepared = super()._prepare_dataset(
            dataset,
            seed,
            outer_split=outer_split,
        )
        stratify = self._selection_stratify_target(
            prepared.y_validation,
            problem_type=prepared.dataset.problem_type,
            selector_fraction=float(self.config.selector_fraction),
        )
        X_calibration, X_selection, y_calibration, y_selection = train_test_split(
            prepared.X_validation,
            prepared.y_validation,
            test_size=float(self.config.selector_fraction),
            random_state=int(seed) + 10_000,
            stratify=stratify,
        )
        return PreparedRoutingSelectorDataset(
            dataset=prepared.dataset,
            X_train=prepared.X_train,
            X_validation=self._frame(X_calibration),
            X_selection=self._frame(X_selection),
            X_test=prepared.X_test,
            y_train=prepared.y_train,
            y_validation=self._series(y_calibration),
            y_selection=self._series(y_selection),
            y_test=prepared.y_test,
            seed=prepared.seed,
        )

    def _run_leaf(
        self,
        *,
        prepared: PreparedRoutingSelectorDataset,
        budget: float,
        model_name: str,
        model_factory: Any,
    ) -> None:
        identity = self._leaf_identity(prepared, budget, model_name)
        record_names = tuple(arm.name for arm in self.arms) + (
            self.config.selector_arm_name,
        )
        pending_names = {
            name
            for name in record_names
            if self._arm_key({**identity, "arm_name": name})
            not in self.completed_arm_keys_
        }
        if not pending_names:
            return

        try:
            ensemble = self._fit_ensemble(
                prepared=prepared,
                budget=budget,
                model_factory=model_factory,
            )
            replay = FittedEnsembleRoutingReplay(show_progress=self.config.show_progress)
            calibration_results = replay.run_validation(
                ensemble=ensemble,
                X_val=prepared.X_validation,
                y_val=prepared.y_validation,
                arms=self.arms,
                metadata=identity,
            )
            temperatures = {
                result.arm_name: result.temperature for result in calibration_results
            }
            selection_results = replay.run_evaluation(
                ensemble=ensemble,
                features=prepared.X_selection,
                target=prepared.y_selection,
                arms=self.arms,
                temperatures=temperatures,
                metadata=identity,
            )
            selection = self.selector.select(selection_results, self.selection_policy)
            test_results = replay.run_evaluation(
                ensemble=ensemble,
                features=prepared.X_test,
                target=prepared.y_test,
                arms=self.arms,
                temperatures=temperatures,
                metadata=identity,
            )
            calibration_by_arm = self._results_by_arm(calibration_results)
            selection_by_arm = self._results_by_arm(selection_results)
            test_by_arm = self._results_by_arm(test_results)

            for arm in self.arms:
                if arm.name not in pending_names:
                    continue
                self._append_record(
                    self._fixed_arm_record(
                        identity=identity,
                        ensemble=ensemble,
                        prepared=prepared,
                        arm=arm,
                        calibration=calibration_by_arm[arm.name],
                        validation=selection_by_arm[arm.name],
                        test=test_by_arm[arm.name],
                    )
                )

            if self.config.selector_arm_name in pending_names:
                selected_name = selection.selected.arm_name
                selected_arm = next(arm for arm in self.arms if arm.name == selected_name)
                self._append_record(
                    {
                        **self._fixed_arm_record(
                            identity=identity,
                            ensemble=ensemble,
                            prepared=prepared,
                            arm=selected_arm,
                            calibration=calibration_by_arm[selected_name],
                            validation=selection_by_arm[selected_name],
                            test=test_by_arm[selected_name],
                        ),
                        "arm_name": self.config.selector_arm_name,
                        "selected_arm_name": selected_name,
                        "selection": selection.summary(),
                    }
                )
        except Exception as exc:
            remaining_names = (
                name
                for name in pending_names
                if self._arm_key({**identity, "arm_name": name})
                not in self.completed_arm_keys_
            )
            for arm_name in sorted(remaining_names):
                self._append_record(
                    {
                        **identity,
                        "status": "failed",
                        "arm_name": arm_name,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )

    def _fixed_arm_record(
        self,
        *,
        identity: Mapping[str, Any],
        ensemble: Any,
        prepared: PreparedRoutingSelectorDataset,
        arm: RoutingGeometryArm,
        calibration: RoutingReplayResult,
        validation: RoutingReplayResult,
        test: RoutingReplayResult,
    ) -> dict[str, Any]:
        return {
            **identity,
            "status": "completed",
            "arm_name": arm.name,
            "selected_arm_name": arm.name,
            "geometry_spec": arm.spec.to_dict(),
            "selected_temperature": float(test.temperature),
            "n_calibration": int(len(prepared.X_validation)),
            "n_selection": int(len(prepared.X_selection)),
            "calibration": calibration.summary(),
            "validation": validation.summary(),
            "test": test.summary(),
            "n_experts": len(ensemble.models),
            "partition_sizes": {
                name: int(len(indices))
                for name, indices in (ensemble.partitioner.partitions or {}).items()
            },
            "sampler_diagnostics": dict(
                getattr(ensemble.partitioner, "diagnostics_", {})
            ),
        }

    def _leaf_identity(
        self,
        prepared: PreparedRoutingSelectorDataset,
        budget: float,
        model_name: str,
    ) -> dict[str, Any]:
        identity = super()._leaf_identity(prepared, budget, model_name)
        identity["n_validation"] = int(
            len(prepared.X_validation) + len(prepared.X_selection)
        )
        return identity

    def _validate_selector_arms(self, policy: RoutingGeometrySelectionPolicy) -> None:
        arm_names = tuple(arm.name for arm in self.arms)
        if arm_names != policy.candidate_arm_names:
            raise ValueError(
                "Selector arms must match policy candidate_arm_names in stable order"
            )

    @staticmethod
    def _results_by_arm(
        results: Sequence[RoutingReplayResult],
    ) -> dict[str, RoutingReplayResult]:
        return {result.arm_name: result for result in results}

    @staticmethod
    def _selection_stratify_target(
        target: pd.Series,
        *,
        problem_type: str,
        selector_fraction: float,
    ) -> Optional[pd.Series]:
        if problem_type != "classification":
            return None
        class_counts = target.value_counts(dropna=False)
        selection_size = int(math.ceil(len(target) * selector_fraction))
        calibration_size = len(target) - selection_size
        if (
            class_counts.empty
            or int(class_counts.min()) < 2
            or min(selection_size, calibration_size) < len(class_counts)
        ):
            return None
        return target


def run_rmt_validation_geometry_selector_experiment(
    *,
    regression_tasks: Optional[Sequence[str]] = None,
    classification_tasks: Optional[Sequence[str]] = None,
    models: Sequence[str] = ("lightgbm",),
    budget_ratios: Sequence[float] = DEFAULT_ROUTING_BUDGETS,
    seeds: Sequence[int] = DEFAULT_ROUTING_SEEDS,
    max_train_rows: Optional[int] = 100_000,
    selector_fraction: float = 0.50,
    minimum_improvement: float = 0.0,
    output_dir: Optional[str | Path] = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    config = RoutingGeometrySelectorExperimentConfig(
        regression_tasks=(
            DEFAULT_ROUTING_REGRESSION_TASKS
            if regression_tasks is None
            else tuple(regression_tasks)
        ),
        classification_tasks=(
            DEFAULT_ROUTING_CLASSIFICATION_TASKS
            if classification_tasks is None
            else tuple(classification_tasks)
        ),
        models=models,
        budget_ratios=budget_ratios,
        seeds=seeds,
        max_train_rows=max_train_rows,
        selector_fraction=selector_fraction,
        minimum_improvement=minimum_improvement,
        output_dir=None if output_dir is None else Path(output_dir),
        show_progress=show_progress,
    )
    return ValidationRoutingGeometryExperimentOrchestrator(config).run()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-train-rows", type=int, default=100_000)
    parser.add_argument("--selector-fraction", type=float, default=0.50)
    parser.add_argument("--minimum-improvement", type=float, default=0.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_ROUTING_SEEDS))
    parser.add_argument(
        "--budgets",
        type=float,
        nargs="+",
        default=list(DEFAULT_ROUTING_BUDGETS),
    )
    parser.add_argument("--models", nargs="+", default=["lightgbm"])
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    table = run_rmt_validation_geometry_selector_experiment(
        models=tuple(args.models),
        budget_ratios=tuple(args.budgets),
        seeds=tuple(args.seeds),
        max_train_rows=args.max_train_rows,
        selector_fraction=args.selector_fraction,
        minimum_improvement=args.minimum_improvement,
        output_dir=args.output_dir,
        show_progress=not args.no_progress,
    )
    print(table.to_string(index=False))
