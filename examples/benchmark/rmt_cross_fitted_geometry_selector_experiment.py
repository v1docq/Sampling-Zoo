"""Phase A.2 benchmark for cross-fitted selection of RMT routing geometry."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Optional, Sequence

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = Path(__file__).resolve().parent
for module_path in (ROOT_DIR, BENCHMARK_DIR):
    if str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))

from rmt_experiment_utils import json_ready  # noqa: E402
from rmt_routing_geometry_experiment import (  # noqa: E402
    DEFAULT_ROUTING_BUDGETS,
    DEFAULT_ROUTING_SEEDS,
    PreparedRoutingDataset,
    RoutingGeometryExperimentConfig,
    RoutingGeometryExperimentOrchestrator,
)
from routing_geometry_replay import (  # noqa: E402
    CrossFittedRoutingReplaySelection,
    FittedEnsembleRoutingReplay,
    RoutingGeometryArm,
    default_validation_selector_arms,
)
from sampling_zoo.core.experiment.routing_geometry_selection import (  # noqa: E402
    ClassificationRoutingGuardSpec,
    CrossFittedRoutingGeometrySelectionSpec,
    CrossFittedRoutingGeometrySelector,
)
from sampling_zoo.core.experiment.errors import ResumeCompatibilityError  # noqa: E402
from sampling_zoo.core.experiment.routing_replay import (  # noqa: E402
    RoutingReplayEvaluator,
    RoutingReplayResult,
)


DEFAULT_CROSS_FITTED_REGRESSION_TASKS: tuple[str, ...] = (
    "Brazilian_houses",
    "elevators",
)
DEFAULT_CROSS_FITTED_CLASSIFICATION_TASKS: tuple[str, ...] = (
    "adult",
    "jannis",
)
DEFAULT_CROSS_FITTED_SELECTOR_ARM_NAME = "A8_cross_fitted_selected"
DEFAULT_TAIL_GUARDED_SELECTOR_ARM_NAME = "A9_tail_guarded_cross_fitted_selected"
DEFAULT_CLASSIFICATION_GUARDED_SELECTOR_ARM_NAME = (
    "A10_classification_guarded_cross_fitted_selected"
)
DEFAULT_TAIL_GUARD_REGRESSION_TASKS: tuple[str, ...] = (
    "diamonds",
    "OnlineNewsPopularity",
)
DEFAULT_TAIL_GUARD_CLASSIFICATION_TASKS: tuple[str, ...] = (
    "bank-marketing",
    "covertype",
)
DEFAULT_CLASSIFICATION_GUARD_BINARY_TASKS: tuple[str, ...] = (
    "adult",
    "bank-marketing",
    "PhishingWebsites",
)
DEFAULT_CLASSIFICATION_GUARD_MULTICLASS_TASKS: tuple[str, ...] = (
    "car",
    "segment",
    "vehicle",
)


@dataclass(frozen=True)
class CrossFittedRoutingGeometryExperimentConfig(RoutingGeometryExperimentConfig):
    """Small-grid Phase A.2 protocol with stability and non-inferiority gates."""

    regression_tasks: Sequence[str] = DEFAULT_CROSS_FITTED_REGRESSION_TASKS
    classification_tasks: Sequence[str] = DEFAULT_CROSS_FITTED_CLASSIFICATION_TASKS
    selection_folds: int = 5
    selector_arm_name: str = DEFAULT_CROSS_FITTED_SELECTOR_ARM_NAME
    confidence_level: float = 0.95
    bootstrap_iterations: int = 2_000
    min_positive_fold_fraction: float = 2.0 / 3.0
    noninferiority_margin: float = 0.005
    regression_tail_guard: bool = False
    tail_risk_quantile: float = 0.90
    tail_noninferiority_margin: float = 0.005
    regression_stratification_bins: Optional[int] = None
    classification_routing_guard: bool = False
    roc_auc_absolute_margin: float = 0.005
    log_loss_relative_margin: float = 0.01
    classification_primary_selection_margin: Optional[float] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if int(self.selection_folds) < 3:
            raise ValueError("selection_folds must be at least 3")
        if not str(self.selector_arm_name).strip():
            raise ValueError("selector_arm_name must be non-empty")
        if not 0.0 < float(self.confidence_level) < 1.0:
            raise ValueError("confidence_level must be in (0, 1)")
        if int(self.bootstrap_iterations) < 100:
            raise ValueError("bootstrap_iterations must be at least 100")
        if not 0.0 < float(self.min_positive_fold_fraction) <= 1.0:
            raise ValueError("min_positive_fold_fraction must be in (0, 1]")
        if float(self.noninferiority_margin) < 0.0:
            raise ValueError("noninferiority_margin must be non-negative")
        if not 0.5 <= float(self.tail_risk_quantile) < 1.0:
            raise ValueError("tail_risk_quantile must be in [0.5, 1)")
        if float(self.tail_noninferiority_margin) < 0.0:
            raise ValueError("tail_noninferiority_margin must be non-negative")
        if (
            self.regression_stratification_bins is not None
            and int(self.regression_stratification_bins) < 2
        ):
            raise ValueError("regression_stratification_bins must be at least 2")
        if self.regression_tail_guard and self.regression_stratification_bins is None:
            raise ValueError(
                "regression_stratification_bins is required when "
                "regression_tail_guard is enabled"
            )
        if float(self.roc_auc_absolute_margin) < 0.0:
            raise ValueError("roc_auc_absolute_margin must be non-negative")
        if float(self.log_loss_relative_margin) < 0.0:
            raise ValueError("log_loss_relative_margin must be non-negative")
        if (
            self.classification_primary_selection_margin is not None
            and float(self.classification_primary_selection_margin) < 0.0
        ):
            raise ValueError(
                "classification_primary_selection_margin must be non-negative"
            )


class CrossFittedRoutingGeometryExperimentOrchestrator(
    RoutingGeometryExperimentOrchestrator
):
    """Fit experts once and select A2/A5/A6 only from inner validation folds."""

    config: CrossFittedRoutingGeometryExperimentConfig

    def __init__(
        self,
        config: CrossFittedRoutingGeometryExperimentConfig,
        *,
        arms: Optional[Sequence[RoutingGeometryArm]] = None,
        selector: Optional[CrossFittedRoutingGeometrySelector] = None,
    ) -> None:
        if config.output_dir is None:
            prefix = (
                "run_rmt_tail_guarded_geometry_selector_"
                if config.regression_tail_guard
                else "run_rmt_cross_fitted_geometry_selector_"
            )
            run_id = f"{prefix}{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            config = replace(
                config,
                output_dir=BENCHMARK_DIR / "results" / run_id,
            )
        selected_arms = tuple(arms or default_validation_selector_arms())
        super().__init__(config, arms=selected_arms)
        spec = CrossFittedRoutingGeometrySelectionSpec(
            reference_arm=selected_arms[0].name,
            candidate_arms=tuple(arm.name for arm in selected_arms[1:]),
            min_folds=int(config.selection_folds),
            confidence_level=float(config.confidence_level),
            bootstrap_iterations=int(config.bootstrap_iterations),
            min_positive_fold_fraction=float(config.min_positive_fold_fraction),
            noninferiority_margin=float(config.noninferiority_margin),
            tail_guard_primary_metrics=(
                ("rmse",) if config.regression_tail_guard else ()
            ),
            tail_noninferiority_margin=float(config.tail_noninferiority_margin),
            classification_guard=(
                ClassificationRoutingGuardSpec(
                    roc_auc_absolute_margin=float(
                        config.roc_auc_absolute_margin
                    ),
                    log_loss_relative_margin=float(
                        config.log_loss_relative_margin
                    ),
                    primary_selection_margin=(
                        None
                        if config.classification_primary_selection_margin is None
                        else float(
                            config.classification_primary_selection_margin
                        )
                    ),
                )
                if config.classification_routing_guard
                else None
            ),
            random_state=int(config.seeds[0]),
        )
        self.selector = selector or CrossFittedRoutingGeometrySelector(spec)
        self.selection_records_: list[dict[str, Any]] = []
        self.selection_keys_: set[tuple[str, int, float, str]] = set()
        self.fold_rows_: list[dict[str, Any]] = []
        self.summary_rows_: list[dict[str, Any]] = []

    def _initialize_artifacts(self) -> None:
        self._validate_resume_selector_config()
        super()._initialize_artifacts()
        if self.output_dir_ is None:
            raise RuntimeError("Artifacts are not initialized")
        self.selection_records_ = self._load_jsonl(
            self.output_dir_ / "cross_fitted_geometry_runs.jsonl"
        )
        self.selection_keys_ = {
            self._selection_key(record)
            for record in self.selection_records_
            if record.get("status") == "completed"
        }
        self.fold_rows_ = self._load_csv_records(
            self.output_dir_ / "geometry_selection_fold_losses.csv"
        )
        self.summary_rows_ = self._load_csv_records(
            self.output_dir_ / "geometry_selection_summary.csv"
        )

    def _validate_resume_selector_config(self) -> None:
        if self.config.output_dir is None:
            return
        output_dir = Path(self.config.output_dir)
        metadata_path = output_dir / "run_meta.json"
        if not metadata_path.exists():
            return
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise ResumeCompatibilityError(
                scope="routing_geometry.resume",
                message="Existing run metadata cannot be read.",
                code="routing_geometry_resume_metadata_invalid",
                details={"path": str(metadata_path)},
            ) from exc
        existing = payload.get("config", {}) if isinstance(payload, dict) else {}
        expected = {
            "selector_arm_name": self.config.selector_arm_name,
            "selection_folds": int(self.config.selection_folds),
            "regression_tail_guard": bool(self.config.regression_tail_guard),
            "tail_risk_quantile": float(self.config.tail_risk_quantile),
            "tail_noninferiority_margin": float(
                self.config.tail_noninferiority_margin
            ),
            "regression_stratification_bins": (
                None
                if self.config.regression_stratification_bins is None
                else int(self.config.regression_stratification_bins)
            ),
            "classification_routing_guard": bool(
                self.config.classification_routing_guard
            ),
            "roc_auc_absolute_margin": float(
                self.config.roc_auc_absolute_margin
            ),
            "log_loss_relative_margin": float(
                self.config.log_loss_relative_margin
            ),
            "classification_primary_selection_margin": (
                None
                if self.config.classification_primary_selection_margin is None
                else float(
                    self.config.classification_primary_selection_margin
                )
            ),
        }
        defaults = {
            "regression_tail_guard": False,
            "tail_risk_quantile": 0.90,
            "tail_noninferiority_margin": 0.005,
            "regression_stratification_bins": None,
            "classification_routing_guard": False,
            "roc_auc_absolute_margin": 0.005,
            "log_loss_relative_margin": 0.01,
            "classification_primary_selection_margin": None,
        }
        mismatches = {
            name: {
                "existing": existing.get(name, defaults.get(name)),
                "requested": value,
            }
            for name, value in expected.items()
            if existing.get(name, defaults.get(name)) != value
        }
        if mismatches:
            raise ResumeCompatibilityError(
                scope="routing_geometry.resume",
                message=(
                    "Existing selector artifacts use incompatible selection semantics."
                ),
                code="routing_geometry_resume_selector_mismatch",
                details={"path": str(output_dir), "mismatches": mismatches},
            )

    def _run_leaf(
        self,
        *,
        prepared: PreparedRoutingDataset,
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
        selection_key = self._selection_key(identity)
        if not pending_names and self._selection_artifacts_complete(identity):
            return

        try:
            ensemble = self._fit_ensemble(
                prepared=prepared,
                budget=budget,
                model_factory=model_factory,
            )
            replay = FittedEnsembleRoutingReplay(
                evaluator=RoutingReplayEvaluator(
                    regression_tail_quantile=float(self.config.tail_risk_quantile),
                ),
                show_progress=self.config.show_progress,
            )
            selection = replay.select_geometry_cross_fitted(
                ensemble=ensemble,
                X_val=prepared.X_validation,
                y_val=prepared.y_validation,
                arms=self.arms,
                selector=self.selector,
                n_folds=int(self.config.selection_folds),
                random_state=int(prepared.seed) + 20_000,
                regression_stratification_bins=(
                    int(self.config.regression_stratification_bins)
                    if self.config.regression_stratification_bins is not None
                    else None
                ),
                metadata=identity,
            )
            validation_by_arm = self._results_by_arm(selection.validation_results)
            temperatures = {
                result.arm_name: result.temperature
                for result in selection.validation_results
            }
            test_results = replay.run_evaluation(
                ensemble=ensemble,
                features=prepared.X_test,
                target=prepared.y_test,
                arms=self.arms,
                temperatures=temperatures,
                expert_priors=selection.expert_priors,
                metadata=identity,
            )
            test_by_arm = self._results_by_arm(test_results)
            self._persist_selection_artifacts(identity, selection)

            for arm in self.arms:
                if arm.name not in pending_names:
                    continue
                self._append_record(
                    self._fixed_arm_record(
                        identity=identity,
                        ensemble=ensemble,
                        prepared=prepared,
                        arm=arm,
                        validation=validation_by_arm[arm.name],
                        test=test_by_arm[arm.name],
                    )
                )

            if self.config.selector_arm_name in pending_names:
                selected_name = selection.decision.selected_arm
                self._append_record(
                    {
                        **self._fixed_arm_record(
                            identity=identity,
                            ensemble=ensemble,
                            prepared=prepared,
                            arm=selection.selected_arm,
                            validation=validation_by_arm[selected_name],
                            test=test_by_arm[selected_name],
                        ),
                        "arm_name": self.config.selector_arm_name,
                        "selected_arm_name": selected_name,
                        "selection": selection.decision.summary(),
                    }
                )
        except Exception as exc:
            for arm_name in sorted(pending_names):
                if (
                    self._arm_key({**identity, "arm_name": arm_name})
                    in self.completed_arm_keys_
                ):
                    continue
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
        prepared: PreparedRoutingDataset,
        arm: RoutingGeometryArm,
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
            "n_inner_folds": int(self.config.selection_folds),
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

    def _persist_selection_artifacts(
        self,
        identity: Mapping[str, Any],
        selection: CrossFittedRoutingReplaySelection,
    ) -> None:
        if self.output_dir_ is None:
            raise RuntimeError("Artifacts are not initialized")
        key = self._selection_key(identity)
        record = json_ready(
            {
                **identity,
                "status": "completed",
                "selector_arm_name": self.config.selector_arm_name,
                "selected_arm_name": selection.decision.selected_arm,
                "selection": selection.decision.summary(),
                "validation": selection.validation_result.summary(),
                "expert_priors": selection.expert_priors,
            }
        )
        existing_fold_keys = {
            self._fold_key(row) for row in self.fold_rows_
        }
        for score in selection.fold_scores:
            row = json_ready({**identity, **score.summary()})
            if self._fold_key(row) not in existing_fold_keys:
                self.fold_rows_.append(row)
                existing_fold_keys.add(self._fold_key(row))
        existing_summary_keys = {
            self._summary_key(row) for row in self.summary_rows_
        }
        for evidence in selection.decision.evidence:
            summary = evidence.summary()
            primary_interval = summary.pop("confidence_interval")
            robust_interval = summary.pop("robust_confidence_interval")
            tail_interval = summary.pop("tail_confidence_interval")
            row = json_ready(
                {
                    **identity,
                    "selection_status": selection.decision.status,
                    "selected_arm_name": selection.decision.selected_arm,
                    **summary,
                    "confidence_lower": (
                        None if primary_interval is None else primary_interval[0]
                    ),
                    "confidence_upper": (
                        None if primary_interval is None else primary_interval[1]
                    ),
                    "robust_confidence_lower": (
                        None if robust_interval is None else robust_interval[0]
                    ),
                    "robust_confidence_upper": (
                        None if robust_interval is None else robust_interval[1]
                    ),
                    "tail_confidence_lower": (
                        None if tail_interval is None else tail_interval[0]
                    ),
                    "tail_confidence_upper": (
                        None if tail_interval is None else tail_interval[1]
                    ),
                }
            )
            if self._summary_key(row) not in existing_summary_keys:
                self.summary_rows_.append(row)
                existing_summary_keys.add(self._summary_key(row))
        pd.DataFrame(self.fold_rows_).to_csv(
            self.output_dir_ / "geometry_selection_fold_losses.csv",
            index=False,
        )
        pd.DataFrame(self.summary_rows_).to_csv(
            self.output_dir_ / "geometry_selection_summary.csv",
            index=False,
        )
        if key not in self.selection_keys_:
            with (self.output_dir_ / "cross_fitted_geometry_runs.jsonl").open(
                "a",
                encoding="utf-8",
            ) as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            self.selection_records_.append(record)
            self.selection_keys_.add(key)

    def _build_result_table(self) -> pd.DataFrame:
        result = super()._build_result_table()
        if result.empty:
            return result
        selections = [record.get("selection", {}) or {} for record in self.records_]
        result["selection_status"] = [value.get("status") for value in selections]
        result["selection_reference_arm"] = [
            value.get("reference_arm") for value in selections
        ]
        result["selection_reason"] = [value.get("reason") for value in selections]
        return result

    @staticmethod
    def _results_by_arm(
        results: Sequence[RoutingReplayResult],
    ) -> dict[str, RoutingReplayResult]:
        return {result.arm_name: result for result in results}

    @staticmethod
    def _selection_key(
        record: Mapping[str, Any],
    ) -> tuple[str, int, float, str]:
        return (
            str(record.get("dataset")),
            int(record.get("seed", -1)),
            round(float(record.get("budget_ratio", -1.0)), 12),
            str(record.get("model")),
        )

    def _selection_artifacts_complete(
        self,
        identity: Mapping[str, Any],
    ) -> bool:
        selection_key = self._selection_key(identity)
        if selection_key not in self.selection_keys_:
            return False
        available_fold_keys = {self._fold_key(row) for row in self.fold_rows_}
        expected_fold_keys = {
            (*selection_key, arm.name, str(fold_id))
            for arm in self.arms
            for fold_id in range(int(self.config.selection_folds))
        }
        available_summary_keys = {
            self._summary_key(row) for row in self.summary_rows_
        }
        expected_summary_keys = {
            (*selection_key, arm_name)
            for arm_name in self.selector.spec.candidate_arms
        }
        return (
            expected_fold_keys <= available_fold_keys
            and expected_summary_keys <= available_summary_keys
        )

    @classmethod
    def _fold_key(
        cls,
        record: Mapping[str, Any],
    ) -> tuple[str, int, float, str, str, str]:
        return (
            *cls._selection_key(record),
            str(record.get("arm_name")),
            str(record.get("fold_id")),
        )

    @classmethod
    def _summary_key(
        cls,
        record: Mapping[str, Any],
    ) -> tuple[str, int, float, str, str]:
        return (
            *cls._selection_key(record),
            str(record.get("arm_name")),
        )

    @staticmethod
    def _load_jsonl(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        records = []
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid cross-fitted selection record at line {line_number}"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(
                    f"Cross-fitted selection record at line {line_number} must be an object"
                )
            records.append(value)
        return records

    @staticmethod
    def _load_csv_records(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        return pd.read_csv(path).to_dict(orient="records")


def run_rmt_cross_fitted_geometry_selector_experiment(
    *,
    regression_tasks: Optional[Sequence[str]] = None,
    classification_tasks: Optional[Sequence[str]] = None,
    models: Sequence[str] = ("lightgbm",),
    budget_ratios: Sequence[float] = DEFAULT_ROUTING_BUDGETS,
    seeds: Sequence[int] = DEFAULT_ROUTING_SEEDS,
    n_partitions: int = 5,
    max_train_rows: Optional[int] = 100_000,
    sampler_extra_params: Optional[Mapping[str, Any]] = None,
    selection_folds: int = 5,
    regression_tail_guard: bool = False,
    tail_risk_quantile: float = 0.90,
    tail_noninferiority_margin: float = 0.005,
    regression_stratification_bins: Optional[int] = None,
    classification_routing_guard: bool = False,
    roc_auc_absolute_margin: float = 0.005,
    log_loss_relative_margin: float = 0.01,
    classification_primary_selection_margin: Optional[float] = None,
    selector_arm_name: Optional[str] = None,
    output_dir: Optional[str | Path] = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    effective_selector_arm_name = selector_arm_name or (
        DEFAULT_TAIL_GUARDED_SELECTOR_ARM_NAME
        if regression_tail_guard
        else DEFAULT_CROSS_FITTED_SELECTOR_ARM_NAME
    )
    config = CrossFittedRoutingGeometryExperimentConfig(
        regression_tasks=(
            DEFAULT_CROSS_FITTED_REGRESSION_TASKS
            if regression_tasks is None
            else tuple(regression_tasks)
        ),
        classification_tasks=(
            DEFAULT_CROSS_FITTED_CLASSIFICATION_TASKS
            if classification_tasks is None
            else tuple(classification_tasks)
        ),
        models=models,
        budget_ratios=budget_ratios,
        seeds=seeds,
        n_partitions=n_partitions,
        max_train_rows=max_train_rows,
        sampler_extra_params=dict(sampler_extra_params or {}),
        selection_folds=selection_folds,
        selector_arm_name=effective_selector_arm_name,
        regression_tail_guard=regression_tail_guard,
        tail_risk_quantile=tail_risk_quantile,
        tail_noninferiority_margin=tail_noninferiority_margin,
        regression_stratification_bins=regression_stratification_bins,
        classification_routing_guard=classification_routing_guard,
        roc_auc_absolute_margin=roc_auc_absolute_margin,
        log_loss_relative_margin=log_loss_relative_margin,
        classification_primary_selection_margin=(
            classification_primary_selection_margin
        ),
        output_dir=None if output_dir is None else Path(output_dir),
        show_progress=show_progress,
    )
    return CrossFittedRoutingGeometryExperimentOrchestrator(config).run()


def run_rmt_classification_guarded_geometry_selector_experiment(
    *,
    classification_tasks: Optional[Sequence[str]] = None,
    models: Sequence[str] = ("lightgbm",),
    budget_ratios: Sequence[float] = DEFAULT_ROUTING_BUDGETS,
    seeds: Sequence[int] = DEFAULT_ROUTING_SEEDS,
    n_partitions: int = 5,
    max_train_rows: Optional[int] = 100_000,
    sampler_extra_params: Optional[Mapping[str, Any]] = None,
    selection_folds: int = 5,
    roc_auc_absolute_margin: float = 0.005,
    log_loss_relative_margin: float = 0.01,
    classification_primary_selection_margin: float = 0.0,
    output_dir: Optional[str | Path] = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Run the AMLB-compatible guarded geometry selector on six tasks."""

    default_tasks = (
        *DEFAULT_CLASSIFICATION_GUARD_BINARY_TASKS,
        *DEFAULT_CLASSIFICATION_GUARD_MULTICLASS_TASKS,
    )
    return run_rmt_cross_fitted_geometry_selector_experiment(
        regression_tasks=(),
        classification_tasks=(
            default_tasks
            if classification_tasks is None
            else tuple(classification_tasks)
        ),
        models=models,
        budget_ratios=budget_ratios,
        seeds=seeds,
        n_partitions=n_partitions,
        max_train_rows=max_train_rows,
        sampler_extra_params=sampler_extra_params,
        selection_folds=selection_folds,
        classification_routing_guard=True,
        roc_auc_absolute_margin=roc_auc_absolute_margin,
        log_loss_relative_margin=log_loss_relative_margin,
        classification_primary_selection_margin=(
            classification_primary_selection_margin
        ),
        selector_arm_name=DEFAULT_CLASSIFICATION_GUARDED_SELECTOR_ARM_NAME,
        output_dir=output_dir,
        show_progress=show_progress,
    )


def run_rmt_tail_guarded_geometry_selector_experiment(
    *,
    regression_tasks: Optional[Sequence[str]] = None,
    classification_tasks: Optional[Sequence[str]] = None,
    models: Sequence[str] = ("lightgbm",),
    budget_ratios: Sequence[float] = DEFAULT_ROUTING_BUDGETS,
    seeds: Sequence[int] = DEFAULT_ROUTING_SEEDS,
    n_partitions: int = 5,
    max_train_rows: Optional[int] = 100_000,
    sampler_extra_params: Optional[Mapping[str, Any]] = None,
    selection_folds: int = 5,
    tail_risk_quantile: float = 0.90,
    tail_noninferiority_margin: float = 0.005,
    regression_stratification_bins: int = 10,
    output_dir: Optional[str | Path] = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Run Phase A.3 with RMSE tail-risk gates on previously unseen tasks."""

    return run_rmt_cross_fitted_geometry_selector_experiment(
        regression_tasks=(
            DEFAULT_TAIL_GUARD_REGRESSION_TASKS
            if regression_tasks is None
            else tuple(regression_tasks)
        ),
        classification_tasks=(
            DEFAULT_TAIL_GUARD_CLASSIFICATION_TASKS
            if classification_tasks is None
            else tuple(classification_tasks)
        ),
        models=models,
        budget_ratios=budget_ratios,
        seeds=seeds,
        n_partitions=n_partitions,
        max_train_rows=max_train_rows,
        sampler_extra_params=sampler_extra_params,
        selection_folds=selection_folds,
        regression_tail_guard=True,
        tail_risk_quantile=tail_risk_quantile,
        tail_noninferiority_margin=tail_noninferiority_margin,
        regression_stratification_bins=regression_stratification_bins,
        selector_arm_name=DEFAULT_TAIL_GUARDED_SELECTOR_ARM_NAME,
        output_dir=output_dir,
        show_progress=show_progress,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-train-rows", type=int, default=100_000)
    parser.add_argument("--selection-folds", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_ROUTING_SEEDS))
    parser.add_argument(
        "--budgets",
        type=float,
        nargs="+",
        default=list(DEFAULT_ROUTING_BUDGETS),
    )
    parser.add_argument("--models", nargs="+", default=["lightgbm"])
    parser.add_argument("--tail-risk-guard", action="store_true")
    parser.add_argument("--tail-risk-quantile", type=float, default=0.90)
    parser.add_argument("--tail-noninferiority-margin", type=float, default=0.005)
    parser.add_argument("--regression-stratification-bins", type=int, default=10)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    runner = (
        run_rmt_tail_guarded_geometry_selector_experiment
        if args.tail_risk_guard
        else run_rmt_cross_fitted_geometry_selector_experiment
    )
    table = runner(
        models=tuple(args.models),
        budget_ratios=tuple(args.budgets),
        seeds=tuple(args.seeds),
        max_train_rows=args.max_train_rows,
        selection_folds=args.selection_folds,
        **(
            {
                "tail_risk_quantile": args.tail_risk_quantile,
                "tail_noninferiority_margin": args.tail_noninferiority_margin,
                "regression_stratification_bins": args.regression_stratification_bins,
            }
            if args.tail_risk_guard
            else {}
        ),
        output_dir=args.output_dir,
        show_progress=not args.no_progress,
    )
    print(table.to_string(index=False))
