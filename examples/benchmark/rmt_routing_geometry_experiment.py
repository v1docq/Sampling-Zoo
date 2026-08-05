"""Targeted Phase A benchmark for RMT partition routing geometry."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

ROOT_DIR = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = Path(__file__).resolve().parent
for module_path in (ROOT_DIR, BENCHMARK_DIR):
    if str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))

from benchmark_dataset_interfaces import cap_openml_dataset  # noqa: E402
from benchmark_datasets import (  # noqa: E402
    OpenMLRawDatasetBundle,
    RawDatasetBundle,
    load_suite_raw_datasets,
)
from benchmark_models import make_model_pool  # noqa: E402
from benchmark_repo import OPENML_CLASSIFICATION_SUITE, OPENML_REGRESSION_SUITE  # noqa: E402
from benchmark_sampling_strategies import make_chunking_strategy_configs  # noqa: E402
from rmt_experiment_utils import json_ready  # noqa: E402
from routing_geometry_replay import (  # noqa: E402
    FittedEnsembleRoutingReplay,
    RoutingGeometryArm,
    default_routing_geometry_arms,
)
from sampling_zoo.core.utils.sampling_ensemble import SamplingEnsemble  # noqa: E402


DEFAULT_ROUTING_REGRESSION_TASKS: tuple[str, ...] = (
    "diamonds",
    "elevators",
    "Brazilian_houses",
    "OnlineNewsPopularity",
)
DEFAULT_ROUTING_CLASSIFICATION_TASKS: tuple[str, ...] = (
    "adult",
    "bank-marketing",
    "covertype",
    "jannis",
)
DEFAULT_ROUTING_BUDGETS: tuple[float, ...] = (0.01, 0.05, 0.10, 0.20)
DEFAULT_ROUTING_SEEDS: tuple[int, ...] = (42, 43, 44, 45, 46)


@dataclass(frozen=True)
class RoutingGeometryExperimentConfig:
    regression_suite: Optional[int] = OPENML_REGRESSION_SUITE
    classification_suite: Optional[int] = OPENML_CLASSIFICATION_SUITE
    regression_tasks: Sequence[str] = DEFAULT_ROUTING_REGRESSION_TASKS
    classification_tasks: Sequence[str] = DEFAULT_ROUTING_CLASSIFICATION_TASKS
    models: Sequence[str] = ("lightgbm",)
    budget_ratios: Sequence[float] = DEFAULT_ROUTING_BUDGETS
    seeds: Sequence[int] = DEFAULT_ROUTING_SEEDS
    n_partitions: int = 5
    max_train_rows: Optional[int] = 100_000
    validation_fraction: float = 0.20
    output_dir: Optional[Path] = None
    show_progress: bool = True

    def __post_init__(self) -> None:
        if not self.models:
            raise ValueError("models must be non-empty")
        if not self.budget_ratios or any(not 0 < float(value) <= 1 for value in self.budget_ratios):
            raise ValueError("budget_ratios must contain values in (0, 1]")
        if not self.seeds:
            raise ValueError("seeds must be non-empty")
        if int(self.n_partitions) < 1:
            raise ValueError("n_partitions must be positive")
        if not 0 < float(self.validation_fraction) < 1:
            raise ValueError("validation_fraction must be in (0, 1)")


@dataclass(frozen=True)
class PreparedRoutingDataset:
    dataset: RawDatasetBundle
    X_train: pd.DataFrame
    X_validation: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_validation: pd.Series
    y_test: pd.Series
    seed: int


@dataclass(frozen=True)
class RoutingOuterSplit:
    X_train_full: Any
    X_test: Any
    y_train_full: Any
    y_test: Any


class RoutingGeometryExperimentOrchestrator:
    """Thin shell for load -> split -> fit once -> replay -> persist."""

    def __init__(
        self,
        config: RoutingGeometryExperimentConfig,
        *,
        arms: Optional[Sequence[RoutingGeometryArm]] = None,
    ) -> None:
        self.config = config
        self.arms = tuple(arms or default_routing_geometry_arms())
        self.output_dir_: Optional[Path] = None
        self.records_: list[dict[str, Any]] = []
        self.completed_arm_keys_: set[tuple[str, int, float, str, str]] = set()

    def run(self) -> pd.DataFrame:
        self._prepare_runtime()
        datasets = self._load_datasets()
        self._initialize_artifacts()
        self._execute_grid(datasets)
        result = self._build_result_table()
        self._finalize_artifacts(result)
        return result

    def _prepare_runtime(self) -> None:
        if not self.arms:
            raise ValueError("At least one routing arm is required")

    def _load_datasets(self) -> list[RawDatasetBundle]:
        with tqdm(
            total=2,
            desc="Load routing benchmark datasets",
            disable=not self.config.show_progress,
            leave=False,
            unit="stage",
        ) as progress:
            datasets = load_suite_raw_datasets(
                regression_suite=self.config.regression_suite,
                classification_suite=self.config.classification_suite,
                regression_tasks=tuple(self.config.regression_tasks),
                classification_tasks=tuple(self.config.classification_tasks),
                show_progress=self.config.show_progress,
            )
            progress.update(1)
            capped = [
                cap_openml_dataset(dataset, self.config.max_train_rows, int(self.config.seeds[0]))
                if isinstance(dataset, OpenMLRawDatasetBundle)
                else dataset
                for dataset in datasets
            ]
            progress.update(1)
        if not capped:
            raise RuntimeError("No datasets were resolved for routing geometry experiment")
        return capped

    def _initialize_artifacts(self) -> None:
        output_dir = self.config.output_dir
        if output_dir is None:
            run_id = f"run_rmt_routing_geometry_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            output_dir = BENCHMARK_DIR / "results" / run_id
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir_ = output_dir
        self.records_ = self._load_existing_records()
        self.completed_arm_keys_ = {
            self._arm_key(record)
            for record in self.records_
            if record.get("status") == "completed" and record.get("arm_name")
        }
        self._write_metadata(status="running")

    def _execute_grid(self, datasets: Sequence[RawDatasetBundle]) -> None:
        for dataset in tqdm(
            datasets,
            desc="Routing datasets",
            disable=not self.config.show_progress,
        ):
            outer_split = self._load_outer_split(dataset)
            for seed in tqdm(
                self.config.seeds,
                desc=f"Seeds ({dataset.name})",
                disable=not self.config.show_progress,
                leave=False,
            ):
                prepared = self._prepare_dataset(
                    dataset,
                    int(seed),
                    outer_split=outer_split,
                )
                self._run_dataset_seed(prepared)

    def _load_outer_split(self, dataset: RawDatasetBundle) -> RoutingOuterSplit:
        if isinstance(dataset, OpenMLRawDatasetBundle):
            split = dataset.load_split_data(show_progress=self.config.show_progress)
            X_train_full, y_train_full, X_test, y_test = split[:4]
        else:
            stratify = dataset.y if dataset.problem_type == "classification" else None
            X_train_full, X_test, y_train_full, y_test = train_test_split(
                dataset.X,
                dataset.y,
                test_size=0.20,
                random_state=int(self.config.seeds[0]),
                stratify=stratify,
            )
        return RoutingOuterSplit(
            X_train_full=X_train_full,
            X_test=X_test,
            y_train_full=y_train_full,
            y_test=y_test,
        )

    def _prepare_dataset(
        self,
        dataset: RawDatasetBundle,
        seed: int,
        *,
        outer_split: RoutingOuterSplit,
    ) -> PreparedRoutingDataset:
        X_train_full = outer_split.X_train_full
        y_train_full = outer_split.y_train_full
        stratify = y_train_full if dataset.problem_type == "classification" else None
        X_train, X_validation, y_train, y_validation = train_test_split(
            X_train_full,
            y_train_full,
            test_size=float(self.config.validation_fraction),
            random_state=seed,
            stratify=stratify,
        )
        return PreparedRoutingDataset(
            dataset=dataset,
            X_train=self._frame(X_train),
            X_validation=self._frame(X_validation),
            X_test=self._frame(outer_split.X_test),
            y_train=self._series(y_train),
            y_validation=self._series(y_validation),
            y_test=self._series(outer_split.y_test),
            seed=seed,
        )

    def _run_dataset_seed(self, prepared: PreparedRoutingDataset) -> None:
        model_pool = make_model_pool(
            seed=prepared.seed,
            model_names=self.config.models,
            problem_type=prepared.dataset.problem_type,
        )
        for budget in tqdm(
            self.config.budget_ratios,
            desc=f"Budgets ({prepared.dataset.name}/seed={prepared.seed})",
            disable=not self.config.show_progress,
            leave=False,
        ):
            for model_name, model_factory in tqdm(
                model_pool.items(),
                desc="Chunk models",
                disable=not self.config.show_progress,
                leave=False,
            ):
                self._run_leaf(
                    prepared=prepared,
                    budget=float(budget),
                    model_name=model_name,
                    model_factory=model_factory,
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
        pending_arms = tuple(
            arm
            for arm in self.arms
            if self._arm_key({**identity, "arm_name": arm.name})
            not in self.completed_arm_keys_
        )
        if not pending_arms:
            return
        try:
            ensemble = self._fit_ensemble(
                prepared=prepared,
                budget=budget,
                model_factory=model_factory,
            )
            replay = FittedEnsembleRoutingReplay(show_progress=self.config.show_progress)
            validation_results = replay.run_validation(
                ensemble=ensemble,
                X_val=prepared.X_validation,
                y_val=prepared.y_validation,
                arms=pending_arms,
                metadata=identity,
            )
            selected_temperatures = {
                result.arm_name: result.temperature for result in validation_results
            }
            test_results = replay.run_evaluation(
                ensemble=ensemble,
                features=prepared.X_test,
                target=prepared.y_test,
                arms=pending_arms,
                temperatures=selected_temperatures,
                metadata=identity,
            )
            validation_by_arm = {result.arm_name: result for result in validation_results}
            for test_result in test_results:
                validation_result = validation_by_arm[test_result.arm_name]
                self._append_record(
                    {
                        **identity,
                        "status": "completed",
                        "arm_name": test_result.arm_name,
                        "geometry_spec": next(
                            arm.spec.to_dict()
                            for arm in pending_arms
                            if arm.name == test_result.arm_name
                        ),
                        "selected_temperature": float(test_result.temperature),
                        "validation": validation_result.summary(),
                        "test": test_result.summary(),
                        "n_experts": len(ensemble.models),
                        "partition_sizes": {
                            name: int(len(indices))
                            for name, indices in (ensemble.partitioner.partitions or {}).items()
                        },
                        "sampler_diagnostics": dict(
                            getattr(ensemble.partitioner, "diagnostics_", {})
                        ),
                    }
                )
        except Exception as exc:
            self._append_record(
                {
                    **identity,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    def _fit_ensemble(
        self,
        *,
        prepared: PreparedRoutingDataset,
        budget: float,
        model_factory: Any,
    ) -> SamplingEnsemble:
        config = make_chunking_strategy_configs(
            problem_type=prepared.dataset.problem_type,
            strategy_names=("rmt_contraction",),
            n_partitions=self.config.n_partitions,
            seed=prepared.seed,
            ensemble_method="routed_weighted",
            budget_ratio=budget,
            force_chunking=True,
            extra_strategy_params={
                "rmt_contraction": {
                    "view_strategy": "gaussian",
                    "embedding_mode": "sv_scaled",
                    "router": "spectral",
                    "routing_representation": "source_centroid",
                    "routing_metric": "squared_euclidean",
                    "routing_kernel": "softmax",
                    "show_progress": self.config.show_progress,
                }
            },
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

    def _leaf_identity(
        self,
        prepared: PreparedRoutingDataset,
        budget: float,
        model_name: str,
    ) -> dict[str, Any]:
        return {
            "dataset": prepared.dataset.name,
            "task_id": getattr(prepared.dataset, "task_id", None),
            "task_name": getattr(prepared.dataset, "task_name", None),
            "suite_id": getattr(prepared.dataset, "suite_id", None),
            "problem_type": prepared.dataset.problem_type,
            "seed": prepared.seed,
            "budget_ratio": budget,
            "model": model_name,
            "n_train": len(prepared.X_train),
            "n_validation": len(prepared.X_validation),
            "n_test": len(prepared.X_test),
        }

    def _append_record(self, record: Mapping[str, Any]) -> None:
        payload = json_ready(dict(record))
        self.records_.append(payload)
        if payload.get("status") == "completed" and payload.get("arm_name"):
            self.completed_arm_keys_.add(self._arm_key(payload))
        if self.output_dir_ is None:
            raise RuntimeError("Artifacts are not initialized")
        with (self.output_dir_ / "routing_geometry_runs.jsonl").open(
            "a",
            encoding="utf-8",
        ) as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._build_result_table().to_csv(
            self.output_dir_ / "routing_geometry_replay.csv",
            index=False,
        )
        self._write_metadata(status="running")

    def _load_existing_records(self) -> list[dict[str, Any]]:
        if self.output_dir_ is None:
            return []
        path = self.output_dir_ / "routing_geometry_runs.jsonl"
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
                    f"Invalid routing replay record at line {line_number}"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(
                    f"Routing replay record at line {line_number} must be an object"
                )
            records.append(value)
        return records

    @staticmethod
    def _arm_key(record: Mapping[str, Any]) -> tuple[str, int, float, str, str]:
        return (
            str(record.get("dataset")),
            int(record.get("seed", -1)),
            round(float(record.get("budget_ratio", -1.0)), 12),
            str(record.get("model")),
            str(record.get("arm_name")),
        )

    def _build_result_table(self) -> pd.DataFrame:
        rows = []
        for record in self.records_:
            row = {
                key: record.get(key)
                for key in (
                    "dataset",
                    "task_id",
                    "task_name",
                    "suite_id",
                    "problem_type",
                    "seed",
                    "budget_ratio",
                    "model",
                    "arm_name",
                    "status",
                    "selected_temperature",
                    "n_train",
                    "n_validation",
                    "n_test",
                    "n_experts",
                    "error_type",
                    "error",
                )
            }
            for split_name in ("validation", "test"):
                split = record.get(split_name, {}) or {}
                row[f"{split_name}_primary_metric"] = split.get("primary_metric")
                row[f"{split_name}_primary_value"] = split.get("primary_value")
                for metric_name, value in (split.get("metrics", {}) or {}).items():
                    row[f"{split_name}_{metric_name}"] = value
                for name, value in (split.get("routing", {}) or {}).items():
                    row[f"{split_name}_routing_{name}"] = value
                for name, value in (split.get("diagnostics", {}) or {}).items():
                    row[f"{split_name}_{name}"] = value
            rows.append(row)
        return pd.DataFrame(rows)

    def _finalize_artifacts(self, result: pd.DataFrame) -> None:
        if self.output_dir_ is None:
            raise RuntimeError("Artifacts are not initialized")
        result.to_csv(self.output_dir_ / "routing_geometry_replay.csv", index=False)
        status = (
            "completed_with_failures"
            if any(record.get("status") == "failed" for record in self.records_)
            else "completed"
        )
        self._write_metadata(status=status)

    def _write_metadata(self, *, status: str) -> None:
        if self.output_dir_ is None:
            return
        payload = {
            "status": status,
            "updated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "record_count": len(self.records_),
            "config": json_ready(asdict(self.config)),
            "arms": [
                {
                    "name": arm.name,
                    "spec": arm.spec.to_dict(),
                    "temperature_candidates": list(arm.temperature_candidates),
                    "use_validation_priors": arm.use_validation_priors,
                }
                for arm in self.arms
            ],
        }
        (self.output_dir_ / "run_meta.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @staticmethod
    def _frame(value: Any) -> pd.DataFrame:
        return value.reset_index(drop=True) if isinstance(value, pd.DataFrame) else pd.DataFrame(value)

    @staticmethod
    def _series(value: Any) -> pd.Series:
        return value.reset_index(drop=True) if isinstance(value, pd.Series) else pd.Series(value)

    @staticmethod
    def _primary_metric(problem_type: str, target: pd.Series) -> str:
        if problem_type == "regression":
            return "rmse"
        return "roc_auc" if target.nunique(dropna=True) == 2 else "log_loss"

    @staticmethod
    def _class_representatives(
        features: pd.DataFrame,
        target: pd.Series,
    ) -> dict[Any, tuple[pd.Series, Any]]:
        representatives = {}
        values = target.to_numpy()
        for label in np.unique(values):
            index = int(np.flatnonzero(values == label)[0])
            representatives[label] = (features.iloc[index], target.iloc[index])
        return representatives


def run_rmt_routing_geometry_experiment(
    *,
    regression_tasks: Optional[Sequence[str]] = None,
    classification_tasks: Optional[Sequence[str]] = None,
    models: Sequence[str] = ("lightgbm",),
    budget_ratios: Sequence[float] = DEFAULT_ROUTING_BUDGETS,
    seeds: Sequence[int] = DEFAULT_ROUTING_SEEDS,
    max_train_rows: Optional[int] = 100_000,
    output_dir: Optional[str | Path] = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    config = RoutingGeometryExperimentConfig(
        regression_tasks=regression_tasks or DEFAULT_ROUTING_REGRESSION_TASKS,
        classification_tasks=classification_tasks or DEFAULT_ROUTING_CLASSIFICATION_TASKS,
        models=models,
        budget_ratios=budget_ratios,
        seeds=seeds,
        max_train_rows=max_train_rows,
        output_dir=None if output_dir is None else Path(output_dir),
        show_progress=show_progress,
    )
    return RoutingGeometryExperimentOrchestrator(config).run()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-train-rows", type=int, default=100_000)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_ROUTING_SEEDS),
    )
    parser.add_argument("--budgets", type=float, nargs="+", default=list(DEFAULT_ROUTING_BUDGETS))
    parser.add_argument("--models", nargs="+", default=["lightgbm"])
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    table = run_rmt_routing_geometry_experiment(
        models=tuple(args.models),
        budget_ratios=tuple(args.budgets),
        seeds=tuple(args.seeds),
        max_train_rows=args.max_train_rows,
        output_dir=args.output_dir,
        show_progress=not args.no_progress,
    )
    print(table.to_string(index=False))
