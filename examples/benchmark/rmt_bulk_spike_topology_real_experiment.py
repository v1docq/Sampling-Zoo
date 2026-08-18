"""Real-data pilot for strict exact-budget bulk/spike topology selection."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field, replace
from datetime import datetime
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Any, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


ROOT_DIR = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = Path(__file__).resolve().parent
for module_path in (ROOT_DIR, BENCHMARK_DIR):
    if str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))

from benchmark_datasets import RawDatasetBundle  # noqa: E402
from benchmark_sampling_strategies import make_chunking_strategy_configs  # noqa: E402
from rmt_cross_fitted_geometry_selector_experiment import (  # noqa: E402
    DEFAULT_TAIL_GUARDED_SELECTOR_ARM_NAME,
)
from rmt_experiment_utils import json_ready, markdown_table  # noqa: E402
from rmt_routing_geometry_experiment import (  # noqa: E402
    PreparedRoutingDataset,
    RoutingGeometryExperimentConfig,
    RoutingGeometryExperimentOrchestrator,
    RoutingOuterSplit,
)
from routing_geometry_replay import (  # noqa: E402
    FittedEnsembleRoutingReplay,
    default_validation_selector_arms,
)
from sampling_zoo.core.experiment.routing_geometry_selection import (  # noqa: E402
    ClassificationRoutingGuardSpec,
    CrossFittedRoutingGeometrySelectionSpec,
    CrossFittedRoutingGeometrySelector,
    RoutingGeometryFoldScore,
    RoutingGeometryGuardMetricScore,
    RoutingGeometryTailRiskScore,
)
from sampling_zoo.core.experiment.routing_replay import (  # noqa: E402
    upper_tail_mean_absolute_error,
)
from sampling_zoo.core.metrics.eval_metrics import calculate_metrics  # noqa: E402
from sampling_zoo.core.sampling_strategies.spectral.bulk_spike_topology import (  # noqa: E402
    BulkSpikeTopologyMode,
)
from sampling_zoo.core.utils.sampling_ensemble import SamplingEnsemble  # noqa: E402


DEFAULT_REAL_TOPOLOGY_REGRESSION_TASKS: tuple[str, ...] = (
    "diamonds",
    "OnlineNewsPopularity",
)
DEFAULT_REAL_TOPOLOGY_CLASSIFICATION_TASKS: tuple[str, ...] = (
    "bank-marketing",
    "covertype",
)
DEFAULT_REAL_TOPOLOGY_BUDGETS: tuple[float, ...] = (0.01, 0.05, 0.10, 0.20)
DEFAULT_REAL_TOPOLOGY_SEEDS: tuple[int, ...] = (42, 43, 44, 45, 46)
REAL_TOPOLOGY_ARMS: tuple[str, ...] = (
    "B0_standard_A9",
    "B1_bulk_single_spike",
    "B2_bulk_multi_spike",
    "B3_validation_selected",
)

STRICT_RESEARCH_GATE_PROFILE = "strict_research"
PRACTICAL_EXPLORATION_GATE_PROFILE = "practical_exploration"


@dataclass(frozen=True)
class BulkSpikeGateProfile:
    """Predeclared practical tolerances for the aggregate B3 gate."""

    name: str
    primary_harm_margin: float
    tail_harm_margin: float
    brier_harm_margin: float
    ece_harm_margin: float
    f1_macro_harm_margin: float
    worst_class_recall_harm_margin: float

    def classification_harm_margins(self) -> dict[str, float]:
        return {
            "brier_score_gain_vs_b0": self.brier_harm_margin,
            "expected_calibration_error_gain_vs_b0": self.ece_harm_margin,
            "f1_macro_gain_vs_b0": self.f1_macro_harm_margin,
            "worst_class_recall_gain_vs_b0": self.worst_class_recall_harm_margin,
        }

    def to_dict(self) -> dict[str, float | str]:
        return {
            "name": self.name,
            "primary_harm_margin": self.primary_harm_margin,
            "tail_harm_margin": self.tail_harm_margin,
            "brier_harm_margin": self.brier_harm_margin,
            "ece_harm_margin": self.ece_harm_margin,
            "f1_macro_harm_margin": self.f1_macro_harm_margin,
            "worst_class_recall_harm_margin": self.worst_class_recall_harm_margin,
        }


BULK_SPIKE_GATE_PROFILES: dict[str, BulkSpikeGateProfile] = {
    STRICT_RESEARCH_GATE_PROFILE: BulkSpikeGateProfile(
        name=STRICT_RESEARCH_GATE_PROFILE,
        primary_harm_margin=0.005,
        tail_harm_margin=0.01,
        brier_harm_margin=0.01,
        ece_harm_margin=0.01,
        f1_macro_harm_margin=0.01,
        worst_class_recall_harm_margin=0.01,
    ),
    PRACTICAL_EXPLORATION_GATE_PROFILE: BulkSpikeGateProfile(
        name=PRACTICAL_EXPLORATION_GATE_PROFILE,
        primary_harm_margin=0.01,
        tail_harm_margin=0.015,
        brier_harm_margin=0.01,
        ece_harm_margin=0.02,
        f1_macro_harm_margin=0.01,
        worst_class_recall_harm_margin=0.01,
    ),
}


def get_bulk_spike_gate_profile(name: str) -> BulkSpikeGateProfile:
    try:
        return BULK_SPIKE_GATE_PROFILES[str(name)]
    except KeyError as exc:
        supported = ", ".join(sorted(BULK_SPIKE_GATE_PROFILES))
        raise ValueError(
            f"Unknown bulk/spike gate profile {name!r}; expected one of: {supported}"
        ) from exc


@dataclass(frozen=True)
class FullDatasetReferenceContract:
    dataset: str
    problem_type: str
    seed: int
    model: str
    n_train: int
    n_test: int
    test: Mapping[str, Any]
    runtime: Mapping[str, float]
    complexity: Mapping[str, Any] = field(default_factory=dict)
    status: str = "completed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "problem_type": self.problem_type,
            "seed": int(self.seed),
            "model": self.model,
            "n_train": int(self.n_train),
            "n_test": int(self.n_test),
            "test": dict(self.test),
            "runtime": dict(self.runtime),
            "complexity": dict(self.complexity),
            "status": self.status,
        }


@dataclass(frozen=True)
class BulkSpikeRealExperimentConfig(RoutingGeometryExperimentConfig):
    regression_tasks: Sequence[str] = DEFAULT_REAL_TOPOLOGY_REGRESSION_TASKS
    classification_tasks: Sequence[str] = DEFAULT_REAL_TOPOLOGY_CLASSIFICATION_TASKS
    budget_ratios: Sequence[float] = DEFAULT_REAL_TOPOLOGY_BUDGETS
    seeds: Sequence[int] = DEFAULT_REAL_TOPOLOGY_SEEDS
    selector_fraction: float = 0.50
    selection_folds: int = 5
    topology_min_mean_gain: float = 0.01
    topology_min_median_gain: float = 0.005
    topology_min_positive_fold_fraction: float = 0.80
    topology_worst_harm_margin: float = 0.005
    tail_harm_margin: float = 0.01
    gate_profile: str = STRICT_RESEARCH_GATE_PROFILE
    null_resamples: int = 8
    topology_min_partition_size: int = 32

    def __post_init__(self) -> None:
        super().__post_init__()
        if not 0.0 < float(self.selector_fraction) < 1.0:
            raise ValueError("selector_fraction must be in (0, 1)")
        if int(self.selection_folds) < 3:
            raise ValueError("selection_folds must be at least 3")
        if int(self.null_resamples) < 2:
            raise ValueError("null_resamples must be at least 2")
        get_bulk_spike_gate_profile(self.gate_profile)

    @property
    def expected_records(self) -> int:
        return (
            (len(self.regression_tasks) + len(self.classification_tasks))
            * len(self.models)
            * len(self.budget_ratios)
            * len(self.seeds)
            * len(REAL_TOPOLOGY_ARMS)
        )

    @property
    def expected_reference_records(self) -> int:
        return (
            (len(self.regression_tasks) + len(self.classification_tasks))
            * len(self.models)
            * len(self.seeds)
        )


@dataclass(frozen=True)
class PreparedBulkSpikeDataset(PreparedRoutingDataset):
    X_selection: pd.DataFrame
    y_selection: pd.Series


class BulkSpikeRealExperimentOrchestrator(RoutingGeometryExperimentOrchestrator):
    """Fit B0-B2, select B3 away from the outer test split, then persist."""

    config: BulkSpikeRealExperimentConfig

    def __init__(
        self,
        config: BulkSpikeRealExperimentConfig,
        *,
        datasets: Optional[Sequence[RawDatasetBundle]] = None,
    ) -> None:
        if config.output_dir is None:
            run_id = "run_rmt_bulk_spike_real_" + datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )
            config = replace(config, output_dir=BENCHMARK_DIR / "results" / run_id)
        super().__init__(config, arms=default_validation_selector_arms())
        self._provided_datasets = None if datasets is None else tuple(datasets)
        self.reference_records_: dict[
            tuple[str, int, str], dict[str, Any]
        ] = {}
        self.geometry_selector = CrossFittedRoutingGeometrySelector(
            CrossFittedRoutingGeometrySelectionSpec(
                reference_arm=self.arms[0].name,
                candidate_arms=tuple(arm.name for arm in self.arms[1:]),
                min_folds=int(config.selection_folds),
                bootstrap_iterations=1_000,
                min_positive_fold_fraction=2.0 / 3.0,
                noninferiority_margin=0.005,
                tail_guard_primary_metrics=("rmse",),
                tail_noninferiority_margin=float(config.tail_harm_margin),
                classification_guard=ClassificationRoutingGuardSpec(),
                random_state=int(config.seeds[0]),
            )
        )

    def _load_datasets(self) -> list[RawDatasetBundle]:
        if self._provided_datasets is not None:
            return list(self._provided_datasets)
        return super()._load_datasets()

    def _write_metadata(self, *, status: str) -> None:
        super()._write_metadata(status=status)
        if self.output_dir_ is None:
            return
        path = self.output_dir_ / "run_meta.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["full_reference_record_count"] = len(self.reference_records_)
        payload["expected_full_reference_records"] = (
            self.config.expected_reference_records
        )
        payload["gate_profile"] = self._gate_profile().to_dict()
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _gate_profile(self) -> BulkSpikeGateProfile:
        return get_bulk_spike_gate_profile(self.config.gate_profile)

    def _initialize_artifacts(self) -> None:
        super()._initialize_artifacts()
        if self.output_dir_ is None:
            raise RuntimeError("Artifacts are not initialized")
        path = self.output_dir_ / "bulk_spike_full_references.jsonl"
        records = self._load_jsonl_records(path)
        self.reference_records_ = {
            self._reference_key(record): record
            for record in records
            if record.get("status") == "completed"
        }
        self._write_metadata(status="running")

    def _prepare_dataset(
        self,
        dataset: RawDatasetBundle,
        seed: int,
        *,
        outer_split: RoutingOuterSplit,
    ) -> PreparedBulkSpikeDataset:
        prepared = super()._prepare_dataset(
            dataset,
            seed,
            outer_split=outer_split,
        )
        stratify = (
            prepared.y_validation
            if prepared.dataset.problem_type == "classification"
            else None
        )
        X_calibration, X_selection, y_calibration, y_selection = train_test_split(
            prepared.X_validation,
            prepared.y_validation,
            test_size=float(self.config.selector_fraction),
            random_state=int(seed) + 10_000,
            stratify=stratify,
        )
        return PreparedBulkSpikeDataset(
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
        prepared: PreparedBulkSpikeDataset,
        budget: float,
        model_name: str,
        model_factory: Any,
    ) -> None:
        identity = self._leaf_identity(prepared, budget, model_name)
        self._ensure_full_dataset_reference(
            prepared=prepared,
            model_name=model_name,
            model_factory=model_factory,
        )
        pending = {
            arm
            for arm in REAL_TOPOLOGY_ARMS
            if self._arm_key({**identity, "arm_name": arm})
            not in self.completed_arm_keys_
        }
        if not pending:
            return
        try:
            evaluations = self._fit_and_evaluate_topologies(
                prepared=prepared,
                budget=budget,
                model_factory=model_factory,
                identity=identity,
            )
            selected_arm, selection = self._select_topology(
                evaluations,
                prepared=prepared,
            )
            evaluations["B3_validation_selected"] = {
                **evaluations[selected_arm],
                "selected_topology_arm": selected_arm,
                "topology_selection": selection,
            }
            for arm in REAL_TOPOLOGY_ARMS:
                if arm not in pending:
                    continue
                evaluation = evaluations[arm]
                self._append_record(
                    {
                        **identity,
                        "status": "completed",
                        "arm_name": arm,
                        "selected_arm_name": evaluation.get(
                            "selected_topology_arm",
                            arm,
                        ),
                        "n_calibration": len(prepared.X_validation),
                        "n_selection": len(prepared.X_selection),
                        "validation": self._without_stored_outputs(
                            evaluation["selection"]
                        ),
                        "test": evaluation["test"],
                        "n_experts": evaluation["n_experts"],
                        "partition_sizes": evaluation["partition_sizes"],
                        "sampler_diagnostics": evaluation["sampler_diagnostics"],
                        "runtime": evaluation["runtime"],
                        "complexity": evaluation["complexity"],
                        "topology_selection": evaluation.get("topology_selection"),
                    }
                )
        except Exception as exc:
            for arm in sorted(pending):
                self._append_record(
                    {
                        **identity,
                        "status": "failed",
                        "arm_name": arm,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )

    def _ensure_full_dataset_reference(
        self,
        *,
        prepared: PreparedBulkSpikeDataset,
        model_name: str,
        model_factory: Any,
    ) -> dict[str, Any]:
        key = (prepared.dataset.name, int(prepared.seed), str(model_name))
        existing = self.reference_records_.get(key)
        if existing is not None:
            return existing
        X_reference, y_reference = self._full_reference_training_data(prepared)
        fit_started = perf_counter()
        model = model_factory()
        model.fit(X_reference, y_reference)
        fit_seconds = float(perf_counter() - fit_started)
        inference_started = perf_counter()
        if prepared.dataset.problem_type == "classification":
            probabilities = self._aligned_model_probabilities(
                model,
                prepared.X_test,
                np.unique(prepared.y_train),
            )
            classes = np.unique(prepared.y_train)
            prediction = classes[np.argmax(probabilities, axis=1)]
        else:
            probabilities = None
            prediction = np.asarray(model.predict(prepared.X_test))
        inference_seconds = float(perf_counter() - inference_started)
        metrics = calculate_metrics(
            prepared.y_test,
            prediction,
            probabilities,
            prepared.dataset.problem_type,
            classes=(
                np.unique(prepared.y_train)
                if prepared.dataset.problem_type == "classification"
                else None
            ),
        )
        if prepared.dataset.problem_type == "regression":
            metrics["mae"] = float(
                np.mean(
                    np.abs(
                        np.asarray(prepared.y_test, dtype=float)
                        - np.asarray(prediction, dtype=float)
                    )
                )
            )
            metrics["tail_mean_absolute_error"] = (
                upper_tail_mean_absolute_error(
                    prepared.y_test,
                    prediction,
                    quantile=0.90,
                )
            )
        primary = self._primary_metric(
            prepared.dataset.problem_type,
            prepared.y_train,
        )
        contract = FullDatasetReferenceContract(
            dataset=prepared.dataset.name,
            problem_type=prepared.dataset.problem_type,
            seed=prepared.seed,
            model=model_name,
            n_train=len(X_reference),
            n_test=len(prepared.X_test),
            test={
                "primary_metric": primary,
                "primary_value": float(metrics[primary]),
                "metrics": {
                    name: float(value) for name, value in metrics.items()
                },
            },
            runtime={
                "fit_seconds": fit_seconds,
                "inference_seconds": inference_seconds,
            },
            complexity=self._models_complexity((model,), prepared.X_test),
        )
        payload = json_ready(contract.to_dict())
        self.reference_records_[key] = payload
        self._append_reference_record(payload)
        return payload

    @staticmethod
    def _full_reference_training_data(
        prepared: PreparedBulkSpikeDataset,
    ) -> tuple[pd.DataFrame, pd.Series]:
        features = pd.concat(
            (
                prepared.X_train,
                prepared.X_validation,
                prepared.X_selection,
            ),
            axis=0,
            ignore_index=True,
        )
        target = pd.concat(
            (
                prepared.y_train,
                prepared.y_validation,
                prepared.y_selection,
            ),
            axis=0,
            ignore_index=True,
        )
        if len(features) != len(target):
            raise RuntimeError("Full-reference features and target do not align")
        return features, target

    @staticmethod
    def _aligned_model_probabilities(
        model: Any,
        features: pd.DataFrame,
        global_classes: np.ndarray,
    ) -> np.ndarray:
        if not hasattr(model, "predict_proba"):
            raise RuntimeError("classification_probabilities_required")
        raw = np.asarray(model.predict_proba(features), dtype=float)
        model_classes = np.asarray(getattr(model, "classes_", ()))
        if raw.ndim != 2 or raw.shape[1] != model_classes.size:
            raise ValueError("Model probabilities do not align with model classes")
        aligned = np.zeros((len(features), len(global_classes)), dtype=float)
        positions = {label: index for index, label in enumerate(global_classes)}
        for source, label in enumerate(model_classes):
            if label in positions:
                aligned[:, positions[label]] = raw[:, source]
        row_sums = np.sum(aligned, axis=1, keepdims=True)
        if np.any(row_sums <= 0.0):
            raise ValueError("Aligned model probabilities have zero total mass")
        return aligned / row_sums

    def _append_reference_record(self, record: Mapping[str, Any]) -> None:
        if self.output_dir_ is None:
            raise RuntimeError("Artifacts are not initialized")
        with (self.output_dir_ / "bulk_spike_full_references.jsonl").open(
            "a",
            encoding="utf-8",
        ) as handle:
            handle.write(json.dumps(json_ready(dict(record)), ensure_ascii=False) + "\n")
        pd.DataFrame(self.reference_records_.values()).to_json(
            self.output_dir_ / "bulk_spike_full_references.json",
            orient="records",
            indent=2,
            force_ascii=False,
        )
        self._write_metadata(status="running")

    @staticmethod
    def _load_jsonl_records(path: Path) -> list[dict[str, Any]]:
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
                    f"Invalid full-data reference at line {line_number}"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError("Full-data reference record must be an object")
            records.append(value)
        return records

    @staticmethod
    def _reference_key(record: Mapping[str, Any]) -> tuple[str, int, str]:
        return (
            str(record.get("dataset")),
            int(record.get("seed", -1)),
            str(record.get("model")),
        )

    def _fit_and_evaluate_topologies(
        self,
        *,
        prepared: PreparedBulkSpikeDataset,
        budget: float,
        model_factory: Any,
        identity: Mapping[str, Any],
    ) -> dict[str, dict[str, Any]]:
        result = {}
        for arm, topology in (
            ("B0_standard_A9", "standard"),
            ("B1_bulk_single_spike", "bulk_single_spike"),
            ("B2_bulk_multi_spike", "bulk_multi_spike"),
        ):
            started = perf_counter()
            ensemble = self._fit_topology_ensemble(
                prepared=prepared,
                budget=budget,
                model_factory=model_factory,
                expert_topology=topology,
            )
            fit_seconds = float(perf_counter() - started)
            if arm == "B0_standard_A9":
                (
                    selection_result,
                    test_result,
                    selected_geometry,
                    test_inference_seconds,
                ) = self._evaluate_b0_a9(
                    ensemble,
                    prepared=prepared,
                    identity=identity,
                )
                selection_summary = self._replay_summary_with_outputs(
                    selection_result,
                    prepared,
                )
                test_summary = test_result.summary()
                test_summary["inference_seconds"] = test_inference_seconds
            else:
                selection_summary = self._evaluate_ensemble(
                    ensemble,
                    prepared.X_selection,
                    prepared.y_selection,
                    stage="inference",
                    store_outputs=True,
                )
                test_summary = self._evaluate_ensemble(
                    ensemble,
                    prepared.X_test,
                    prepared.y_test,
                    stage="inference",
                )
                selected_geometry = (
                    "hierarchical_"
                    + str(ensemble.partitioner.topology_spec_.spike_router)
                )
            sampler = ensemble.partitioner
            result[arm] = {
                "selection": selection_summary,
                "test": test_summary,
                "n_experts": len(ensemble.models),
                "partition_sizes": {
                    name: int(len(rows)) for name, rows in sampler.partitions.items()
                },
                "sampler_diagnostics": dict(sampler.diagnostics_),
                "runtime": {
                    "fit_seconds": fit_seconds,
                    "inference_seconds": test_summary["inference_seconds"],
                },
                "complexity": self._model_complexity(
                    ensemble,
                    prepared.X_selection,
                ),
                "selected_geometry_arm": selected_geometry,
                "topology_mode": (
                    "standard"
                    if sampler.bulk_spike_topology_ is None
                    else sampler.bulk_spike_topology_.mode.value
                ),
            }
        return result

    def _fit_topology_ensemble(
        self,
        *,
        prepared: PreparedBulkSpikeDataset,
        budget: float,
        model_factory: Any,
        expert_topology: str,
    ) -> SamplingEnsemble:
        sampler_params = {
            "view_strategy": "gaussian",
            "embedding_mode": "sv_scaled",
            "router": "spectral",
            "routing_representation": "source_centroid",
            "routing_metric": "median_scaled_euclidean",
            "routing_kernel": "softmax",
            "selection_method": "uniform",
            "null_diagnostic_enabled": True,
            "null_model_policies": (
                "feature_permutation",
                "moment_matched_gaussian",
                "view_resampling",
            ),
            "null_resamples": int(self.config.null_resamples),
            "component_diagnostic_enabled": True,
            "expert_topology": expert_topology,
            "topology_min_partition_size": int(
                self.config.topology_min_partition_size
            ),
            "show_progress": self.config.show_progress,
            **dict(self.config.sampler_extra_params),
        }
        config = make_chunking_strategy_configs(
            problem_type=prepared.dataset.problem_type,
            strategy_names=("rmt_contraction",),
            n_partitions=self.config.n_partitions,
            seed=prepared.seed,
            ensemble_method="routed_weighted",
            budget_ratio=budget,
            force_chunking=True,
            extra_strategy_params={"rmt_contraction": sampler_params},
        )["rmt_contraction"]
        ensemble = SamplingEnsemble(
            problem=prepared.dataset.problem_type,
            partitioner_config=config,
            model_factory=model_factory,
            ensemble_method="routed_weighted",
            show_progress=self.config.show_progress,
        )
        ensemble.train_partition_models(
            X_train=prepared.X_train,
            y_train=prepared.y_train,
            X_val=prepared.X_validation,
            y_val=prepared.y_validation,
            class_samples=(
                self._class_representatives(prepared.X_train, prepared.y_train)
                if prepared.dataset.problem_type == "classification"
                else None
            ),
            cv_fold=1,
            validation_metric=self._primary_metric(
                prepared.dataset.problem_type,
                prepared.y_train,
            ),
            train_all_chunks=True,
            save_models_to_disk=False,
        )
        return ensemble

    def _evaluate_b0_a9(
        self,
        ensemble: SamplingEnsemble,
        *,
        prepared: PreparedBulkSpikeDataset,
        identity: Mapping[str, Any],
    ) -> tuple[Any, Any, str, float]:
        replay = FittedEnsembleRoutingReplay(show_progress=self.config.show_progress)
        selection = replay.select_geometry_cross_fitted(
            ensemble=ensemble,
            X_val=prepared.X_validation,
            y_val=prepared.y_validation,
            arms=self.arms,
            selector=self.geometry_selector,
            n_folds=int(self.config.selection_folds),
            random_state=int(prepared.seed) + 20_000,
            regression_stratification_bins=(
                10 if prepared.dataset.problem_type == "regression" else None
            ),
            metadata=dict(identity),
        )
        validation_by_arm = {
            result.arm_name: result for result in selection.validation_results
        }
        selected_validation = validation_by_arm[selection.decision.selected_arm]
        evaluation = replay.run_evaluation(
            ensemble=ensemble,
            features=prepared.X_selection,
            target=prepared.y_selection,
            arms=(selection.selected_arm,),
            temperatures={
                selection.selected_arm.name: selected_validation.temperature
            },
            expert_priors=selection.expert_priors,
            metadata=dict(identity),
        )[0]
        inference_started = perf_counter()
        test = replay.run_evaluation(
            ensemble=ensemble,
            features=prepared.X_test,
            target=prepared.y_test,
            arms=(selection.selected_arm,),
            temperatures={
                selection.selected_arm.name: selected_validation.temperature
            },
            expert_priors=selection.expert_priors,
            metadata=dict(identity),
        )[0]
        test_inference_seconds = float(perf_counter() - inference_started)
        return (
            evaluation,
            test,
            selection.decision.selected_arm,
            test_inference_seconds,
        )

    def _evaluate_ensemble(
        self,
        ensemble: SamplingEnsemble,
        features: pd.DataFrame,
        target: pd.Series,
        *,
        stage: str,
        store_outputs: bool = False,
    ) -> dict[str, Any]:
        started = perf_counter()
        probabilities = (
            ensemble.ensemble_predict_proba(features, stage=stage)
            if ensemble.problem == "classification"
            else None
        )
        prediction = (
            ensemble.ensemble_predict(features, stage=stage)
            if probabilities is None
            else np.asarray(ensemble.classes_)[np.argmax(probabilities, axis=1)]
        )
        inference_seconds = float(perf_counter() - started)
        metrics = calculate_metrics(
            target,
            prediction,
            probabilities,
            ensemble.problem,
            classes=(ensemble.classes_ if ensemble.problem == "classification" else None),
        )
        if ensemble.problem == "regression":
            error = np.abs(np.asarray(target) - np.asarray(prediction))
            metrics["mae"] = float(np.mean(error))
            metrics["tail_mean_absolute_error"] = (
                upper_tail_mean_absolute_error(target, prediction, quantile=0.90)
            )
        primary = self._primary_metric(ensemble.problem, target)
        summary = {
            "primary_metric": primary,
            "primary_value": float(metrics[primary]),
            "metrics": {name: float(value) for name, value in metrics.items()},
            "routing": ensemble.build_routing_diagnostics(features),
            "inference_seconds": inference_seconds,
        }
        if store_outputs:
            summary["predictions"] = np.asarray(prediction).tolist()
            summary["probabilities"] = (
                None
                if probabilities is None
                else np.asarray(probabilities).tolist()
            )
        return summary

    @staticmethod
    def _replay_summary_with_outputs(
        result: Any,
        prepared: PreparedBulkSpikeDataset,
    ) -> dict[str, Any]:
        summary = result.summary()
        blended = np.asarray(result.blended_output)
        if prepared.dataset.problem_type == "classification":
            classes = np.unique(prepared.y_train)
            summary["predictions"] = classes[np.argmax(blended, axis=1)].tolist()
            summary["probabilities"] = blended.tolist()
        else:
            summary["predictions"] = blended.tolist()
            summary["probabilities"] = None
        summary["inference_seconds"] = float(
            result.diagnostics.get("inference_seconds", 0.0)
        )
        return summary

    @staticmethod
    def _without_stored_outputs(summary: Mapping[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in summary.items()
            if key not in {"predictions", "probabilities"}
        }

    def _select_topology(
        self,
        evaluations: Mapping[str, Mapping[str, Any]],
        *,
        prepared: PreparedBulkSpikeDataset,
    ) -> tuple[str, dict[str, Any]]:
        eligible = []
        b1 = evaluations["B1_bulk_single_spike"]
        b2 = evaluations["B2_bulk_multi_spike"]
        b1_split = b1["sampler_diagnostics"]["spectral_component_split"]
        b2_split = b2["sampler_diagnostics"]["spectral_component_split"]
        if int(b1_split["n_spikes"]) >= 1 and b1["n_experts"] >= 2:
            eligible.append("B1_bulk_single_spike")
        if (
            int(b2_split["n_spikes"]) >= 2
            and b2["topology_mode"] == BulkSpikeTopologyMode.MULTI_SPIKE.value
            and b2["n_experts"] >= 3
        ):
            eligible.append("B2_bulk_multi_spike")
        if not eligible:
            return "B0_standard_A9", {
                "status": "fallback_to_b0",
                "reason": "no_spectrally_eligible_bulk_spike_topology",
                "eligible_candidate_arms": [],
            }
        folds = np.array_split(
            np.arange(len(prepared.y_selection)),
            self.config.selection_folds,
        )
        scores = []
        for fold_id, rows in enumerate(folds):
            for arm in ("B0_standard_A9", *eligible):
                split_metrics = self._slice_selection_metrics(
                    evaluations[arm]["selection"],
                    rows,
                    prepared,
                )
                scores.append(
                    self._topology_fold_score(
                        arm_name=arm,
                        fold_id=str(fold_id),
                        summary=split_metrics,
                        problem_type=prepared.dataset.problem_type,
                    )
                )
        selector = CrossFittedRoutingGeometrySelector(
            CrossFittedRoutingGeometrySelectionSpec(
                reference_arm="B0_standard_A9",
                candidate_arms=tuple(eligible),
                min_folds=int(self.config.selection_folds),
                bootstrap_iterations=1_000,
                min_positive_fold_fraction=float(
                    self.config.topology_min_positive_fold_fraction
                ),
                min_mean_relative_gain=float(self.config.topology_min_mean_gain),
                min_median_relative_gain=float(
                    self.config.topology_min_median_gain
                ),
                noninferiority_margin=0.0,
                require_robust_metric=(
                    prepared.dataset.problem_type == "regression"
                ),
                tail_guard_primary_metrics=("rmse",),
                tail_noninferiority_margin=float(self.config.tail_harm_margin),
                classification_guard=ClassificationRoutingGuardSpec(),
                random_state=int(prepared.seed) + 30_000,
            )
        )
        decision = selector.select(scores)
        return decision.selected_arm, {
            **decision.summary(),
            "eligible_candidate_arms": eligible,
        }

    @staticmethod
    def _slice_selection_metrics(
        summary: Mapping[str, Any],
        rows: np.ndarray,
        prepared: PreparedBulkSpikeDataset,
    ) -> dict[str, Any]:
        predictions = summary.get("predictions")
        probabilities = summary.get("probabilities")
        if predictions is None:
            raise RuntimeError("Topology selection requires stored validation predictions")
        target = prepared.y_selection.iloc[rows]
        problem = prepared.dataset.problem_type
        metrics = calculate_metrics(
            target,
            np.asarray(predictions)[rows],
            None if probabilities is None else np.asarray(probabilities)[rows],
            problem,
            classes=(np.unique(prepared.y_train) if problem == "classification" else None),
        )
        metric = "rmse" if problem == "regression" else (
            "roc_auc" if prepared.y_train.nunique() == 2 else "log_loss"
        )
        if problem == "regression":
            fold_prediction = np.asarray(predictions, dtype=float)[rows]
            absolute_error = np.abs(np.asarray(target, dtype=float) - fold_prediction)
            metrics["mae"] = float(np.mean(absolute_error))
            metrics["tail_mean_absolute_error"] = upper_tail_mean_absolute_error(
                target,
                fold_prediction,
                quantile=0.90,
            )
        return {
            "primary_metric": metric,
            "primary_value": float(metrics[metric]),
            "metrics": {name: float(value) for name, value in metrics.items()},
            "evaluation_rows": int(len(rows)),
        }

    @staticmethod
    def _topology_fold_score(
        *,
        arm_name: str,
        fold_id: str,
        summary: Mapping[str, Any],
        problem_type: str,
    ) -> RoutingGeometryFoldScore:
        metric = str(summary["primary_metric"])
        direction = "higher" if metric == "roc_auc" else "lower"
        metrics = summary["metrics"]
        if problem_type == "regression":
            return RoutingGeometryFoldScore(
                arm_name=arm_name,
                fold_id=fold_id,
                primary_metric=metric,
                primary_value=float(summary["primary_value"]),
                primary_direction=direction,
                evaluation_rows=int(summary["evaluation_rows"]),
                selected_temperature=1.0,
                robust_metric="mae",
                robust_value=float(metrics["mae"]),
                robust_direction="lower",
                tail_risk=RoutingGeometryTailRiskScore(
                    metric="tail_mean_absolute_error",
                    value=float(metrics["tail_mean_absolute_error"]),
                    direction="lower",
                    quantile=0.90,
                ),
            )
        guard_spec = ClassificationRoutingGuardSpec()
        guards = tuple(
            RoutingGeometryGuardMetricScore(
                metric=item.metric,
                value=float(metrics[item.metric]),
                direction=item.direction,
            )
            for item in guard_spec.metric_guards
        )
        return RoutingGeometryFoldScore(
            arm_name=arm_name,
            fold_id=fold_id,
            primary_metric=metric,
            primary_value=float(summary["primary_value"]),
            primary_direction=direction,
            evaluation_rows=int(summary["evaluation_rows"]),
            selected_temperature=1.0,
            guard_metrics=guards,
        )

    @staticmethod
    def _model_complexity(
        ensemble: SamplingEnsemble,
        features: pd.DataFrame,
    ) -> dict[str, Any]:
        return BulkSpikeRealExperimentOrchestrator._models_complexity(
            tuple(model_info["model"] for model_info in ensemble.models),
            features,
        )

    @staticmethod
    def _models_complexity(
        models: Sequence[Any],
        features: pd.DataFrame,
    ) -> dict[str, Any]:
        trees = []
        leaves = []
        splits = []
        depths = []
        importances = []
        shap_profiles = []
        for model in models:
            booster = getattr(model, "booster_", None)
            if booster is None:
                continue
            dump = booster.dump_model()
            tree_info = dump.get("tree_info", [])
            trees.append(len(tree_info))
            leaves.append(sum(int(tree.get("num_leaves", 0)) for tree in tree_info))
            splits.append(sum(max(0, int(tree.get("num_leaves", 0)) - 1) for tree in tree_info))
            depths.append(
                max(
                    (
                        BulkSpikeRealExperimentOrchestrator._tree_depth(
                            tree.get("tree_structure", {})
                        )
                        for tree in tree_info
                    ),
                    default=0,
                )
            )
            values = np.asarray(getattr(model, "feature_importances_", []), dtype=float)
            if values.size:
                importances.append(values / max(np.sum(values), 1.0))
            try:
                sample = features.iloc[: min(len(features), 512)]
                contributions = np.asarray(
                    model.predict(sample, pred_contrib=True),
                    dtype=float,
                )
                if contributions.ndim == 3:
                    contributions = np.mean(
                        np.abs(contributions[..., :-1]),
                        axis=(0, 1),
                    )
                else:
                    block_width = int(getattr(model, "n_features_in_", 0)) + 1
                    if (
                        block_width > 1
                        and contributions.shape[1] > block_width
                        and contributions.shape[1] % block_width == 0
                    ):
                        contributions = contributions.reshape(
                            len(contributions),
                            -1,
                            block_width,
                        )
                        contributions = np.mean(
                            np.abs(contributions[..., :-1]),
                            axis=(0, 1),
                        )
                    else:
                        contributions = np.mean(
                            np.abs(contributions[:, :-1]),
                            axis=0,
                        )
                if contributions.size and np.sum(contributions) > 0:
                    shap_profiles.append(contributions / np.sum(contributions))
            except Exception:
                # Optional interpretability diagnostics must not fail a model run.
                pass
        stability = None
        if len(importances) > 1:
            similarities = []
            for left in range(len(importances)):
                for right in range(left + 1, len(importances)):
                    denominator = np.linalg.norm(importances[left]) * np.linalg.norm(
                        importances[right]
                    )
                    similarities.append(
                        float(
                            np.dot(importances[left], importances[right])
                            / max(denominator, 1e-12)
                        )
                    )
            stability = float(np.mean(similarities))
        shap_stability = BulkSpikeRealExperimentOrchestrator._mean_cosine_similarity(
            shap_profiles
        )
        return {
            "tree_count": int(sum(trees)),
            "leaf_count": int(sum(leaves)),
            "split_count": int(sum(splits)),
            "max_depth": int(max(depths, default=0)),
            "feature_importance_cosine_stability": stability,
            "mean_absolute_shap_cosine_stability": shap_stability,
        }

    @staticmethod
    def _mean_cosine_similarity(vectors: Sequence[np.ndarray]) -> Optional[float]:
        if len(vectors) < 2:
            return None
        similarities = []
        for left in range(len(vectors)):
            for right in range(left + 1, len(vectors)):
                if vectors[left].shape != vectors[right].shape:
                    continue
                denominator = np.linalg.norm(vectors[left]) * np.linalg.norm(
                    vectors[right]
                )
                similarities.append(
                    float(
                        np.dot(vectors[left], vectors[right])
                        / max(denominator, 1e-12)
                    )
                )
        return None if not similarities else float(np.mean(similarities))

    @staticmethod
    def _tree_depth(node: Mapping[str, Any]) -> int:
        if "left_child" not in node and "right_child" not in node:
            return 0
        return 1 + max(
            BulkSpikeRealExperimentOrchestrator._tree_depth(node.get("left_child", {})),
            BulkSpikeRealExperimentOrchestrator._tree_depth(node.get("right_child", {})),
        )

    def _build_result_table(self) -> pd.DataFrame:
        result = super()._build_result_table()
        if result.empty:
            return result
        for index, record in enumerate(self.records_):
            reference = self.reference_records_.get(
                (
                    str(record.get("dataset")),
                    int(record.get("seed", -1)),
                    str(record.get("model")),
                ),
                {},
            )
            for group in ("runtime", "complexity"):
                for name, value in (record.get(group, {}) or {}).items():
                    result.loc[index, f"{group}_{name}"] = value
            result.loc[index, "topology_selection"] = json.dumps(
                record.get("topology_selection"),
                ensure_ascii=False,
            )
            result.loc[index, "partition_total_rows"] = int(
                sum((record.get("partition_sizes", {}) or {}).values())
            )
            partition_sizes = record.get("partition_sizes", {}) or {}
            partition_total = max(int(sum(partition_sizes.values())), 1)
            for partition_name in ("bulk", "spike"):
                partition_rows = int(partition_sizes.get(partition_name, 0))
                result.loc[index, f"{partition_name}_train_rows"] = partition_rows
                result.loc[index, f"{partition_name}_train_share"] = (
                    partition_rows / partition_total
                )
            routing = (
                (record.get("validation", {}) or {}).get("routing", {}) or {}
            )
            hard_counts = routing.get("hard_assignment_counts", {}) or {}
            soft_mass = routing.get("soft_assignment_mass", {}) or {}
            if not isinstance(hard_counts, Mapping):
                hard_counts = {}
            if not isinstance(soft_mass, Mapping):
                soft_mass = {}
            hard_total = float(sum(hard_counts.values()))
            soft_total = float(sum(soft_mass.values()))
            for partition_name in ("bulk", "spike"):
                hard_rows = int(hard_counts.get(partition_name, 0))
                result.loc[index, f"{partition_name}_routing_hard_rows"] = hard_rows
                result.loc[index, f"{partition_name}_routing_hard_share"] = (
                    hard_rows / hard_total if hard_total else np.nan
                )
                soft_value = float(soft_mass.get(partition_name, 0.0))
                result.loc[index, f"{partition_name}_routing_soft_share"] = (
                    soft_value / soft_total if soft_total else np.nan
                )
            topology = (
                (record.get("sampler_diagnostics", {}) or {}).get(
                    "bulk_spike_topology"
                )
                or {}
            )
            result.loc[index, "topology_mode"] = topology.get("mode", "standard")
            result.loc[index, "topology_exact_budget"] = topology.get(
                "exact_budget", True
            )
            result.loc[index, "topology_unique_rows"] = topology.get(
                "unique_rows", result.loc[index, "partition_total_rows"]
            )
            result.loc[index, "topology_total_budget"] = topology.get(
                "total_budget", result.loc[index, "partition_total_rows"]
            )
            result.loc[index, "class_coverage_guaranteed"] = (
                (record.get("sampler_diagnostics", {}) or {}).get(
                    "class_coverage_guaranteed"
                )
            )
            reference_test = reference.get("test", {}) or {}
            result.loc[index, "full_reference_primary_metric"] = (
                reference_test.get("primary_metric")
            )
            result.loc[index, "full_reference_primary_value"] = (
                reference_test.get("primary_value")
            )
            for name, value in (reference_test.get("metrics", {}) or {}).items():
                result.loc[index, f"full_reference_{name}"] = value
            for name, value in (reference.get("runtime", {}) or {}).items():
                result.loc[index, f"full_reference_{name}"] = value
            for name, value in (reference.get("complexity", {}) or {}).items():
                result.loc[index, f"full_reference_complexity_{name}"] = value
        lower = result["test_primary_metric"].isin({"rmse", "log_loss"})
        result["degradation_vs_full"] = np.where(
            lower,
            (
                result["test_primary_value"]
                - result["full_reference_primary_value"]
            )
            / result["full_reference_primary_value"].abs().clip(lower=1e-12),
            result["full_reference_primary_value"]
            - result["test_primary_value"],
        )
        return result

    def _finalize_artifacts(self, result: pd.DataFrame) -> None:
        super()._finalize_artifacts(result)
        if self.output_dir_ is None or result.empty:
            return
        paired = self._paired_results(result)
        paired.to_csv(self.output_dir_ / "bulk_spike_paired.csv", index=False)
        summary = self._summary(paired)
        summary.to_csv(self.output_dir_ / "bulk_spike_summary.csv", index=False)
        self._b1_allocation_table(result).to_csv(
            self.output_dir_ / "bulk_spike_b1_allocation.csv",
            index=False,
        )
        gate = self._gate(result, paired)
        (self.output_dir_ / "bulk_spike_gate.json").write_text(
            json.dumps(json_ready(gate), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        self._write_report(summary, gate)
        self._write_figure(summary)

    @staticmethod
    def _b1_allocation_table(result: pd.DataFrame) -> pd.DataFrame:
        keys = ["dataset", "seed", "budget_ratio", "model"]
        selected = result[
            (result["arm_name"] == "B3_validation_selected")
            & (result["selected_arm_name"] == "B1_bulk_single_spike")
        ][keys].assign(selected_by_b3=True)
        b1 = result[result["arm_name"] == "B1_bulk_single_spike"].copy()
        allocation_columns = [
            *keys,
            "problem_type",
            "n_train",
            "topology_total_budget",
            "bulk_train_rows",
            "spike_train_rows",
            "bulk_train_share",
            "spike_train_share",
            "bulk_routing_hard_rows",
            "spike_routing_hard_rows",
            "bulk_routing_hard_share",
            "spike_routing_hard_share",
            "bulk_routing_soft_share",
            "spike_routing_soft_share",
            "test_primary_metric",
            "test_primary_value",
        ]
        available = [column for column in allocation_columns if column in b1]
        return (
            b1[available]
            .merge(selected, on=keys, how="left", validate="one_to_one")
            .assign(selected_by_b3=lambda frame: frame["selected_by_b3"].fillna(False))
            .sort_values(keys)
            .reset_index(drop=True)
        )

    @staticmethod
    def _paired_results(result: pd.DataFrame) -> pd.DataFrame:
        keys = ["dataset", "seed", "budget_ratio", "model"]
        comparison_columns = [
            "test_primary_value",
            "test_tail_mean_absolute_error",
            "test_brier_score",
            "test_expected_calibration_error",
            "test_f1_macro",
            "test_worst_class_recall",
            "runtime_fit_seconds",
            "runtime_inference_seconds",
        ]
        available = [name for name in comparison_columns if name in result.columns]
        baseline = result[result["arm_name"] == "B0_standard_A9"][
            keys + available
        ].rename(columns={name: f"b0_{name}" for name in available})
        paired = result.merge(baseline, on=keys, how="left", validate="many_to_one")
        lower = paired["test_primary_metric"].isin({"rmse", "log_loss"})
        paired["gain_vs_b0"] = np.where(
            lower,
            (paired["b0_test_primary_value"] - paired["test_primary_value"])
            / paired["b0_test_primary_value"].clip(lower=1e-12),
            paired["test_primary_value"] - paired["b0_test_primary_value"],
        )
        for metric in ("tail_mean_absolute_error", "brier_score"):
            candidate = f"test_{metric}"
            reference = f"b0_test_{metric}"
            if candidate in paired and reference in paired:
                paired[f"{metric}_gain_vs_b0"] = (
                    paired[reference] - paired[candidate]
                ) / paired[reference].abs().clip(lower=1e-12)
        absolute_specs = {
            "expected_calibration_error": "lower",
            "f1_macro": "higher",
            "worst_class_recall": "higher",
        }
        for metric, direction in absolute_specs.items():
            candidate = f"test_{metric}"
            reference = f"b0_test_{metric}"
            if candidate in paired and reference in paired:
                paired[f"{metric}_gain_vs_b0"] = (
                    paired[reference] - paired[candidate]
                    if direction == "lower"
                    else paired[candidate] - paired[reference]
                )
        for metric in ("fit_seconds", "inference_seconds"):
            candidate = f"runtime_{metric}"
            reference = f"b0_runtime_{metric}"
            if candidate in paired and reference in paired:
                paired[f"{metric}_ratio_vs_b0"] = (
                    paired[candidate] / paired[reference].clip(lower=1e-12)
                )
        return paired

    @staticmethod
    def _summary(paired: pd.DataFrame) -> pd.DataFrame:
        aggregations: dict[str, tuple[str, Any]] = {
            "runs": ("seed", "size"),
            "mean_gain": ("gain_vs_b0", "mean"),
            "median_gain": ("gain_vs_b0", "median"),
            "worst_gain": ("gain_vs_b0", "min"),
            "win_rate": (
                "gain_vs_b0",
                lambda values: float((values > 0.0).mean()),
            ),
            "mean_primary_value": ("test_primary_value", "mean"),
            "mean_degradation_vs_full": (
                "degradation_vs_full",
                "mean",
            ),
            "mean_fit_seconds": ("runtime_fit_seconds", "mean"),
            "mean_inference_seconds": ("runtime_inference_seconds", "mean"),
            "mean_tree_count": ("complexity_tree_count", "mean"),
            "mean_max_depth": ("complexity_max_depth", "mean"),
        }
        optional = {
            "mean_tail_gain": "tail_mean_absolute_error_gain_vs_b0",
            "mean_brier_gain": "brier_score_gain_vs_b0",
            "mean_ece_gain": "expected_calibration_error_gain_vs_b0",
            "mean_f1_macro_gain": "f1_macro_gain_vs_b0",
            "mean_worst_class_recall_gain": "worst_class_recall_gain_vs_b0",
            "mean_shap_stability": (
                "complexity_mean_absolute_shap_cosine_stability"
            ),
        }
        for output, source in optional.items():
            if source in paired.columns:
                aggregations[output] = (source, "mean")
        return paired.groupby(
            ["dataset", "budget_ratio", "arm_name"],
            as_index=False,
        ).agg(**aggregations)

    def _gate(self, result: pd.DataFrame, paired: pd.DataFrame) -> dict[str, Any]:
        profile = self._gate_profile()
        b3 = paired[paired["arm_name"] == "B3_validation_selected"]
        selected = b3[b3["selected_arm_name"] != "B0_standard_A9"]
        specialized = result[
            result["arm_name"].isin(
                ["B1_bulk_single_spike", "B2_bulk_multi_spike"]
            )
        ]
        regression = b3[b3["problem_type"] == "regression"]
        classification = b3[b3["problem_type"] == "classification"]
        classification_guard_margins = profile.classification_harm_margins()
        classification_guards = {}
        for column, margin in classification_guard_margins.items():
            values = (
                classification[column].dropna()
                if column in classification.columns
                else pd.Series(dtype=float)
            )
            classification_guards[column] = bool(
                not values.empty and float(values.min()) >= -margin
            )
        tail_values = (
            regression["tail_mean_absolute_error_gain_vs_b0"].dropna()
            if "tail_mean_absolute_error_gain_vs_b0" in regression.columns
            else pd.Series(dtype=float)
        )
        checks = {
            "all_records_completed": bool(
                len(result) == self.config.expected_records
                and (result["status"] == "completed").all()
            ),
            "full_dataset_references_completed": bool(
                len(self.reference_records_)
                == self.config.expected_reference_records
                and all(
                    record.get("status") == "completed"
                    for record in self.reference_records_.values()
                )
            ),
            "specialized_topologies_preserve_exact_budget": bool(
                not specialized.empty
                and specialized["topology_exact_budget"].fillna(False).all()
                and (
                    specialized["topology_unique_rows"]
                    == specialized["topology_total_budget"]
                ).all()
            ),
            "median_gain_nonnegative": bool(not b3.empty and b3["gain_vs_b0"].median() >= 0.0),
            "worst_harm_within_margin": bool(
                not b3.empty
                and b3["gain_vs_b0"].min()
                >= -profile.primary_harm_margin
            ),
            "selected_specialization_win_rate_at_least_80pct": bool(
                not selected.empty
                and (selected["gain_vs_b0"] > 0.0).mean() >= 0.80
            ),
            "regression_tail_noninferiority": bool(
                regression.empty
                or (
                    not tail_values.empty
                    and float(tail_values.min())
                    >= -profile.tail_harm_margin
                )
            ),
            "classification_probability_and_balance_noninferiority": bool(
                classification.empty or all(classification_guards.values())
            ),
        }
        return {
            "status": "passed" if all(checks.values()) else "failed",
            "gate_profile": profile.to_dict(),
            "checks": checks,
            "expected_records": self.config.expected_records,
            "observed_records": len(result),
            "b3_median_gain": None if b3.empty else float(b3["gain_vs_b0"].median()),
            "b3_worst_gain": None if b3.empty else float(b3["gain_vs_b0"].min()),
            "selected_specialization_count": int(len(selected)),
            "selected_specialization_win_rate": (
                0.0
                if selected.empty
                else float((selected["gain_vs_b0"] > 0.0).mean())
            ),
            "regression_worst_tail_gain": (
                None if tail_values.empty else float(tail_values.min())
            ),
            "classification_guards": classification_guards,
        }

    def _write_report(self, summary: pd.DataFrame, gate: Mapping[str, Any]) -> None:
        lines = [
            "# Пилот топологий тела и сигнала на реальных данных",
            "",
            f"Статус критерия допуска: **{gate['status']}**.",
            f"Профиль критерия: `{gate['gate_profile']['name']}`.",
            "",
            "## Критерии",
            "",
            markdown_table(
                pd.DataFrame(
                    [
                        {"проверка": key, "результат": value}
                        for key, value in gate["checks"].items()
                    ]
                )
            ),
            "",
            "## Сводка",
            "",
            markdown_table(summary),
            "",
            "Подробное распределение обучающего бюджета и маршрутизации B1: "
            "`bulk_spike_b1_allocation.csv`.",
            "",
        ]
        (self.output_dir_ / "bulk_spike_report.md").write_text("\n".join(lines), encoding="utf-8")

    def _write_figure(self, summary: pd.DataFrame) -> None:
        datasets = tuple(summary["dataset"].drop_duplicates())
        n_columns = min(2, len(datasets))
        n_rows = int(np.ceil(len(datasets) / max(n_columns, 1)))
        figure, axes = plt.subplots(
            n_rows,
            n_columns,
            figsize=(7 * n_columns, 4.5 * n_rows),
            squeeze=False,
            sharex=True,
        )
        colors = {
            "B1_bulk_single_spike": "#4472C4",
            "B2_bulk_multi_spike": "#70AD47",
            "B3_validation_selected": "#ED7D31",
        }
        for axis, dataset in zip(axes.flat, datasets):
            dataset_summary = summary[
                (summary["dataset"] == dataset)
                & (summary["arm_name"] != "B0_standard_A9")
            ]
            for arm_name, group in dataset_summary.groupby("arm_name"):
                axis.plot(
                    group["budget_ratio"],
                    group["mean_gain"],
                    marker="o",
                    color=colors.get(arm_name),
                    label=arm_name,
                )
            axis.axhline(0.0, color="black", linewidth=0.8)
            axis.set_title(str(dataset))
            axis.set_xlabel("Доля бюджета")
            axis.set_ylabel("Средний выигрыш относительно B0")
            axis.grid(alpha=0.25)
        for axis in axes.flat[len(datasets):]:
            axis.set_visible(False)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        if handles:
            figure.legend(
                handles,
                labels,
                loc="lower center",
                bbox_to_anchor=(0.5, 0.01),
                ncol=len(handles),
            )
        figure.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
        figure.savefig(self.output_dir_ / "bulk_spike_gain_by_budget.png", dpi=180)
        plt.close(figure)


def run_rmt_bulk_spike_topology_real_experiment(
    *,
    regression_tasks: Optional[Sequence[str]] = None,
    classification_tasks: Optional[Sequence[str]] = None,
    models: Sequence[str] = ("lightgbm",),
    budget_ratios: Sequence[float] = DEFAULT_REAL_TOPOLOGY_BUDGETS,
    seeds: Sequence[int] = DEFAULT_REAL_TOPOLOGY_SEEDS,
    max_train_rows: Optional[int] = 100_000,
    gate_profile: str = STRICT_RESEARCH_GATE_PROFILE,
    output_dir: Optional[str | Path] = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    return BulkSpikeRealExperimentOrchestrator(
        BulkSpikeRealExperimentConfig(
            regression_tasks=(
                DEFAULT_REAL_TOPOLOGY_REGRESSION_TASKS
                if regression_tasks is None
                else tuple(regression_tasks)
            ),
            classification_tasks=(
                DEFAULT_REAL_TOPOLOGY_CLASSIFICATION_TASKS
                if classification_tasks is None
                else tuple(classification_tasks)
            ),
            models=models,
            budget_ratios=budget_ratios,
            seeds=seeds,
            max_train_rows=max_train_rows,
            gate_profile=gate_profile,
            output_dir=None if output_dir is None else Path(output_dir),
            show_progress=show_progress,
        )
    ).run()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--gate-profile",
        default=STRICT_RESEARCH_GATE_PROFILE,
        help="Профиль aggregate gate.",
    )
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    result = run_rmt_bulk_spike_topology_real_experiment(
        output_dir=args.output_dir,
        gate_profile=args.gate_profile,
        show_progress=not args.no_progress,
    )
    print(result.tail())


if __name__ == "__main__":
    main()
