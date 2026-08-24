"""Pre-flight RMT classification benchmark for binary and multiclass tasks."""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from tqdm.auto import tqdm

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
BENCHMARK_DIR = Path(__file__).resolve().parent
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

from benchmark_dataset_interfaces import (  # noqa: E402
    cap_openml_dataset,
    make_synthetic_classification_smoke_dataset,
)
from benchmark_datasets import (  # noqa: E402
    OpenMLRawDatasetBundle,
    RawDatasetBundle,
    load_suite_raw_datasets,
)
from benchmark_incremental import IncrementalExperimentSaver  # noqa: E402
from benchmark_models import make_model_pool  # noqa: E402
from benchmark_repo import OPENML_CLASSIFICATION_SUITE  # noqa: E402
from benchmark_sampling_strategies import (  # noqa: E402
    make_chunking_strategy_configs,
)
from rmt_experiment_utils import json_ready  # noqa: E402
from rmt_regression_medium_datasets import (  # noqa: E402
    RMTRegressionExperimentOrchestrator,
)
from sampling_zoo.core.experiment.contracts import (  # noqa: E402
    StrategyGridContract,
)
from sampling_zoo.core.experiment.morphisms import (  # noqa: E402
    normalize_strategy_grid,
)
from sampling_zoo.core.experiment.resume import (  # noqa: E402
    ResumePolicy,
)


DEFAULT_CLASSIFICATION_GATE_TASKS: tuple[str, ...] = (
    "adult",
    "credit-g",
    "segment",
)
DEFAULT_CLASSIFICATION_GATE_BUDGETS: tuple[float, ...] = (0.05, 0.20)


@dataclass(frozen=True)
class RMTClassificationGateConfig:
    classification_suite: int | None = OPENML_CLASSIFICATION_SUITE
    classification_tasks: Sequence[str] | None = (
        DEFAULT_CLASSIFICATION_GATE_TASKS
    )
    models: Sequence[str] = ("lightgbm",)
    budget_ratios: Sequence[float] = DEFAULT_CLASSIFICATION_GATE_BUDGETS
    ensemble_methods: Sequence[str] = ("voting", "routed_weighted")
    n_partitions: int = 5
    max_train_rows: int | None = 50_000
    seed: int = 42
    show_progress: bool = True
    synthetic_smoke: bool = False
    resume_from: str | Path | None = None
    resume_policy: str = ResumePolicy.RETRY_FAILED.value

    def __post_init__(self) -> None:
        ResumePolicy.parse(self.resume_policy)
        if not self.budget_ratios:
            raise ValueError("budget_ratios must not be empty")
        if any(not 0 < float(value) <= 1 for value in self.budget_ratios):
            raise ValueError("budget_ratios must be in (0, 1]")


def make_rmt_classification_gate_strategy_configs(
    *,
    budget_ratios: Sequence[float],
    ensemble_methods: Sequence[str],
    n_partitions: int,
    seed: int,
) -> dict[str, dict[str, Any]]:
    """Build a small grid that isolates classification sampling behaviour."""

    configs: dict[str, dict[str, Any]] = {
        "full_dataset": {
            "strategy": "full_dataset",
            "force_direct_model": True,
            "ensemble_method": "full_dataset",
            "budget_ratio": 1.0,
        }
    }
    for budget_ratio in budget_ratios:
        budget = float(budget_ratio)
        budget_tag = f"{int(round(100 * budget)):02d}"
        random_config = make_chunking_strategy_configs(
            problem_type="classification",
            strategy_names=("random",),
            n_partitions=n_partitions,
            seed=seed,
            ensemble_method="voting",
            budget_ratio=budget,
            force_chunking=True,
        )["random"]
        configs[f"random__voting__budget_{budget_tag}"] = random_config

        for ensemble_method in ensemble_methods:
            rmt_config = make_chunking_strategy_configs(
                problem_type="classification",
                strategy_names=("rmt_contraction",),
                n_partitions=n_partitions,
                seed=seed,
                ensemble_method=str(ensemble_method),
                budget_ratio=budget,
                force_chunking=True,
            )["rmt_contraction"]
            rmt_config.update(
                {
                    "selection_method": "capped_leverage",
                    "class_coverage_policy": "preserve_local_classes",
                    "min_samples_per_class": 1,
                    "cluster_target_type": "classification",
                    "budget_feasibility_mode": "hard",
                    "min_sampled_rows_per_partition": 3,
                    "router": "spectral",
                }
            )
            configs[
                f"rmt_contraction__capped_class_aware__{ensemble_method}"
                f"__budget_{budget_tag}"
            ] = rmt_config
    return configs


class RMTClassificationGateOrchestrator(
    RMTRegressionExperimentOrchestrator
):
    config: RMTClassificationGateConfig

    def __init__(self, config: RMTClassificationGateConfig) -> None:
        super().__init__(config)  # type: ignore[arg-type]

    def _run_id_prefix(self) -> str:
        return "run_rmt_classification_sampling_gate"

    def _create_incremental_saver(
        self,
        logger,
    ) -> IncrementalExperimentSaver:
        def _build_ensemble_tables(
            records: Sequence[Mapping[str, Any]],
        ) -> None:
            self.report_builder.build_tables(records, logger.paths.metrics)

        def _build_markdown_report(
            records: Sequence[Mapping[str, Any]],
        ) -> None:
            logger.create_markdown_report(records)

        return IncrementalExperimentSaver(
            records_path=(
                logger.paths.metrics / "rmt_classification_gate_runs.jsonl"
            ),
            metadata_path=logger.paths.root / "run_meta.json",
            snapshot_hooks=(
                _build_ensemble_tables,
                _build_markdown_report,
            ),
            metadata_builder=lambda records, status: self._build_run_meta(
                logger,
                records,
                status,
            ),
            json_ready=json_ready,
        )

    def _load_datasets(self) -> list[RawDatasetBundle]:
        if self.config.synthetic_smoke:
            return [
                make_synthetic_classification_smoke_dataset(
                    self.config.seed,
                    n_classes=2,
                ),
                make_synthetic_classification_smoke_dataset(
                    self.config.seed + 1,
                    n_classes=3,
                ),
            ]

        datasets = load_suite_raw_datasets(
            classification_suite=self.config.classification_suite,
            regression_suite=None,
            classification_tasks=self.config.classification_tasks,
            regression_tasks=None,
            show_progress=self.config.show_progress,
        )
        prepared: list[RawDatasetBundle] = []
        for dataset in tqdm(
            datasets,
            desc="Prepare classification datasets",
            disable=not self.config.show_progress,
            leave=False,
        ):
            prepared.append(
                cap_openml_dataset(
                    dataset,
                    self.config.max_train_rows,
                    self.config.seed,
                )
                if isinstance(dataset, OpenMLRawDatasetBundle)
                else dataset
            )
        return prepared

    def _load_available_datasets(self) -> list[RawDatasetBundle]:
        datasets = self._load_datasets()
        if not datasets:
            raise RuntimeError(
                "No classification datasets are available for the RMT gate."
            )
        return datasets

    def _build_strategy_grid(self) -> StrategyGridContract:
        return normalize_strategy_grid(
            make_rmt_classification_gate_strategy_configs(
                budget_ratios=self.config.budget_ratios,
                ensemble_methods=self.config.ensemble_methods,
                n_partitions=self.config.n_partitions,
                seed=self.config.seed,
            )
        )

    def _make_model_pool(self) -> dict[str, Any]:
        return make_model_pool(
            seed=self.config.seed,
            model_names=self.config.models,
            problem_type="classification",
        )

    def _build_run_meta(
        self,
        logger,
        run_records: Sequence[Mapping[str, Any]],
        status: str = "completed",
    ) -> dict[str, Any]:
        return {
            "run_id": logger.run_id,
            "output_dir": str(logger.paths.root),
            "classification_suite": self.config.classification_suite,
            "classification_tasks": list(
                self.config.classification_tasks or []
            ),
            "models": list(self.config.models),
            "ensemble_methods": list(self.config.ensemble_methods),
            "budget_ratios": list(self.config.budget_ratios),
            "n_partitions": int(self.config.n_partitions),
            "max_train_rows": self.config.max_train_rows,
            "synthetic_smoke": bool(self.config.synthetic_smoke),
            "resume": (
                None
                if self.resume_session is None
                else self.resume_session.to_dict()
            ),
            "status": status,
            "records": len(run_records),
            "experiment_plan": (
                None
                if self.experiment_plan is None
                else self.experiment_plan.to_dict()
            ),
            "effective_config": asdict(self.config),
        }

    def _announce_completion(self, logger) -> Path:
        print(
            "RMT classification sampling gate completed. "
            f"Artifacts: {logger.paths.root}"
        )
        return logger.paths.root


def run_rmt_classification_sampling_gate(
    classification_tasks: Sequence[str] | None = None,
    models: Sequence[str] = ("lightgbm",),
    max_train_rows: int | None = 50_000,
    show_progress: bool = True,
    synthetic_smoke: bool = False,
) -> Path:
    config = RMTClassificationGateConfig(
        classification_tasks=(
            classification_tasks or DEFAULT_CLASSIFICATION_GATE_TASKS
        ),
        models=models,
        max_train_rows=max_train_rows,
        show_progress=show_progress,
        synthetic_smoke=synthetic_smoke,
    )
    return RMTClassificationGateOrchestrator(config).run()


if __name__ == "__main__":
    run_rmt_classification_sampling_gate()
