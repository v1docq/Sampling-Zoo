"""Run the seven-stage RMT classification and bulk/spike research program."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Optional, Sequence

import pandas as pd
from tqdm.auto import tqdm


ROOT_DIR = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = Path(__file__).resolve().parent
for module_path in (ROOT_DIR, BENCHMARK_DIR):
    if str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))

from rmt_bulk_spike_topology_real_experiment import (  # noqa: E402
    REAL_TOPOLOGY_ARMS,
    run_rmt_bulk_spike_topology_real_experiment,
)
from rmt_classification_geometry_guard_analysis import (  # noqa: E402
    build_classification_geometry_gate,
)
from rmt_conditional_row_policy_experiment import (  # noqa: E402
    ConditionalRowPolicyExperimentConfig,
    run_rmt_conditional_row_policy_experiment,
)
from rmt_cross_fitted_geometry_selector_experiment import (  # noqa: E402
    DEFAULT_CLASSIFICATION_GUARDED_SELECTOR_ARM_NAME,
    run_rmt_classification_guarded_geometry_selector_experiment,
)
from rmt_experiment_utils import json_ready, markdown_table  # noqa: E402
from rmt_scaling_law_analysis import run_scaling_law_analysis  # noqa: E402
from routing_geometry_replay import default_validation_selector_arms  # noqa: E402
from sampling_zoo.core.experiment.research_program import (  # noqa: E402
    ResearchProgramPlan,
    ResearchStageResult,
    ResearchStageSpec,
    ResearchStageStatus,
    deserialize_stage_result,
    selected_stage_closure,
)


PROGRAM_STAGE_CLASSIFICATION = "classification_geometry_guard"
PROGRAM_STAGE_ROW_POLICY = "conditional_row_policy"
PROGRAM_STAGE_PILOT = "bulk_spike_pilot"
PROGRAM_STAGE_CONFIRMATION = "bulk_spike_confirmation"
PROGRAM_STAGE_FULL_GRID = "bulk_spike_full_grid"
PROGRAM_STAGE_SCALING = "dense_scaling_grid"
PROGRAM_STAGE_REPORT = "final_analysis"

DEFAULT_PROGRAM_BUDGETS: tuple[float, ...] = (0.01, 0.05, 0.10, 0.20)
DEFAULT_SCALING_BUDGETS: tuple[float, ...] = (
    0.01,
    0.02,
    0.03,
    0.05,
    0.075,
    0.10,
    0.15,
    0.20,
    0.30,
    0.50,
)
DEFAULT_PROGRAM_SEEDS: tuple[int, ...] = (42, 43, 44, 45, 46)

CLASSIFICATION_GUARD_TASKS: tuple[str, ...] = (
    "adult",
    "bank-marketing",
    "PhishingWebsites",
    "car",
    "segment",
    "vehicle",
)
ROW_POLICY_REGRESSION_TASKS: tuple[str, ...] = (
    "diamonds",
    "elevators",
    "Brazilian_houses",
    "OnlineNewsPopularity",
)
ROW_POLICY_CLASSIFICATION_TASKS: tuple[str, ...] = (
    "adult",
    "bank-marketing",
    "covertype",
    "jannis",
)
PILOT_REGRESSION_TASKS: tuple[str, ...] = (
    "diamonds",
    "OnlineNewsPopularity",
)
PILOT_CLASSIFICATION_TASKS: tuple[str, ...] = (
    "bank-marketing",
    "covertype",
)
CONFIRMATION_REGRESSION_TASKS: tuple[str, ...] = (
    "elevators",
    "Brazilian_houses",
    "house_16H",
    "pol",
)
CONFIRMATION_CLASSIFICATION_TASKS: tuple[str, ...] = (
    "adult",
    "PhishingWebsites",
    "segment",
    "vehicle",
)
FULL_REGRESSION_TASKS: tuple[str, ...] = (
    "diamonds",
    "house_16H",
    "house_sales",
    "elevators",
    "pol",
    "Brazilian_houses",
    "OnlineNewsPopularity",
)
FULL_CLASSIFICATION_TASKS: tuple[str, ...] = (
    "adult",
    "bank-marketing",
    "PhishingWebsites",
    "car",
    "segment",
    "vehicle",
    "covertype",
    "jannis",
)
SCALING_REGRESSION_TASKS: tuple[str, ...] = (
    "diamonds",
    "house_16H",
    "pol",
    "OnlineNewsPopularity",
)
SCALING_CLASSIFICATION_TASKS: tuple[str, ...] = (
    "adult",
    "bank-marketing",
    "covertype",
    "segment",
)


@dataclass(frozen=True)
class RMTBulkSpikeResearchProgramConfig:
    output_root: Optional[Path] = None
    models: Sequence[str] = ("lightgbm",)
    budgets: Sequence[float] = DEFAULT_PROGRAM_BUDGETS
    scaling_budgets: Sequence[float] = DEFAULT_SCALING_BUDGETS
    seeds: Sequence[int] = DEFAULT_PROGRAM_SEEDS
    max_train_rows: Optional[int] = 100_000
    show_progress: bool = True
    bootstrap_iterations: int = 500

    def __post_init__(self) -> None:
        object.__setattr__(self, "models", tuple(str(value) for value in self.models))
        object.__setattr__(
            self,
            "budgets",
            tuple(float(value) for value in self.budgets),
        )
        object.__setattr__(
            self,
            "scaling_budgets",
            tuple(float(value) for value in self.scaling_budgets),
        )
        object.__setattr__(self, "seeds", tuple(int(value) for value in self.seeds))
        if self.output_root is not None:
            object.__setattr__(self, "output_root", Path(self.output_root))
        if not self.models or not self.budgets or not self.scaling_budgets:
            raise ValueError("models and budget grids must be non-empty")
        if not self.seeds:
            raise ValueError("seeds must be non-empty")
        for grid_name in ("budgets", "scaling_budgets"):
            values = getattr(self, grid_name)
            if any(not 0.0 < value < 1.0 for value in values):
                raise ValueError(f"{grid_name} must contain values in (0, 1)")
            if len(set(values)) != len(values):
                raise ValueError(f"{grid_name} must contain unique values")
        if not set(self.budgets) <= set(self.scaling_budgets):
            raise ValueError("scaling_budgets must include the main budget grid")
        if int(self.bootstrap_iterations) < 0:
            raise ValueError("bootstrap_iterations must be non-negative")


def build_rmt_bulk_spike_research_plan(
    config: RMTBulkSpikeResearchProgramConfig,
) -> ResearchProgramPlan:
    models = len(config.models)
    seeds = len(config.seeds)
    budgets = len(config.budgets)
    topology_arms = len(REAL_TOPOLOGY_ARMS)
    geometry_arms = len(default_validation_selector_arms()) + 1
    stage_specs = (
        ResearchStageSpec(
            stage_id=PROGRAM_STAGE_CLASSIFICATION,
            title="Защищённый выбор геометрии для классификации",
            artifact_directory="01_classification_geometry_guard",
            expected_records=(
                len(CLASSIFICATION_GUARD_TASKS)
                * models
                * seeds
                * budgets
                * geometry_arms
            ),
        ),
        ResearchStageSpec(
            stage_id=PROGRAM_STAGE_ROW_POLICY,
            title="Условный выбор uniform или capped leverage",
            artifact_directory="02_conditional_row_policy",
            expected_records=(
                (
                    len(ROW_POLICY_REGRESSION_TASKS)
                    + len(ROW_POLICY_CLASSIFICATION_TASKS)
                )
                * models
                * seeds
                * budgets
                * 3
            ),
        ),
        ResearchStageSpec(
            stage_id=PROGRAM_STAGE_PILOT,
            title="Пилот bulk/spike-топологии",
            artifact_directory="03_bulk_spike_pilot",
            expected_records=(
                (len(PILOT_REGRESSION_TASKS) + len(PILOT_CLASSIFICATION_TASKS))
                * models
                * seeds
                * budgets
                * topology_arms
            ),
            dependencies=(
                PROGRAM_STAGE_CLASSIFICATION,
                PROGRAM_STAGE_ROW_POLICY,
            ),
            gate_dependencies=(PROGRAM_STAGE_CLASSIFICATION,),
        ),
        ResearchStageSpec(
            stage_id=PROGRAM_STAGE_CONFIRMATION,
            title="Независимое подтверждение bulk/spike-топологии",
            artifact_directory="04_bulk_spike_confirmation",
            expected_records=(
                (
                    len(CONFIRMATION_REGRESSION_TASKS)
                    + len(CONFIRMATION_CLASSIFICATION_TASKS)
                )
                * models
                * seeds
                * budgets
                * topology_arms
            ),
            dependencies=(PROGRAM_STAGE_PILOT,),
            gate_dependencies=(PROGRAM_STAGE_PILOT,),
        ),
        ResearchStageSpec(
            stage_id=PROGRAM_STAGE_FULL_GRID,
            title="Полная AMLB-сетка bulk/spike",
            artifact_directory="05_bulk_spike_full_grid",
            expected_records=(
                (len(FULL_REGRESSION_TASKS) + len(FULL_CLASSIFICATION_TASKS))
                * models
                * seeds
                * budgets
                * topology_arms
            ),
            dependencies=(PROGRAM_STAGE_CONFIRMATION,),
            gate_dependencies=(PROGRAM_STAGE_CONFIRMATION,),
        ),
        ResearchStageSpec(
            stage_id=PROGRAM_STAGE_SCALING,
            title="Плотная бюджетная сетка законов масштабирования",
            artifact_directory="06_dense_scaling_grid",
            expected_records=(
                (
                    len(SCALING_REGRESSION_TASKS)
                    + len(SCALING_CLASSIFICATION_TASKS)
                )
                * models
                * seeds
                * len(config.scaling_budgets)
                * topology_arms
            ),
            dependencies=(PROGRAM_STAGE_FULL_GRID,),
            gate_dependencies=(PROGRAM_STAGE_FULL_GRID,),
        ),
        ResearchStageSpec(
            stage_id=PROGRAM_STAGE_REPORT,
            title="Итоговый анализ и законы масштабирования",
            artifact_directory="07_final_analysis",
            expected_records=(
                (
                    len(SCALING_REGRESSION_TASKS)
                    + len(SCALING_CLASSIFICATION_TASKS)
                )
                * models
                * 2
            ),
            dependencies=(PROGRAM_STAGE_SCALING,),
        ),
    )
    return ResearchProgramPlan(stages=stage_specs)


class RMTBulkSpikeResearchProgramOrchestrator:
    def __init__(
        self,
        config: RMTBulkSpikeResearchProgramConfig,
        *,
        handlers: Optional[
            Mapping[str, Callable[[ResearchStageSpec], ResearchStageResult]]
        ] = None,
    ) -> None:
        self.config = config
        self.output_root = self._resolve_output_root()
        self.plan = build_rmt_bulk_spike_research_plan(config)
        self._handlers = dict(handlers or self._default_handlers())
        self.results: dict[str, ResearchStageResult] = {}

    def run(
        self,
        *,
        selected_stages: Optional[Sequence[str]] = None,
    ) -> Path:
        self.output_root.mkdir(parents=True, exist_ok=True)
        self._write_plan()
        self.results = self._load_state()
        selected = (
            self.plan.stage_ids()
            if selected_stages is None
            else selected_stage_closure(self.plan, selected_stages)
        )
        for stage_id in tqdm(
            selected,
            desc="RMT research program",
            unit="stage",
            disable=not self.config.show_progress,
        ):
            stage = self.plan.stage(stage_id)
            existing = self.results.get(stage_id)
            if existing is not None and existing.complete:
                continue
            blocked_reason = self.plan.blocked_reason(stage_id, self.results)
            if blocked_reason is not None:
                self.results[stage_id] = ResearchStageResult(
                    stage_id=stage_id,
                    status=ResearchStageStatus.BLOCKED,
                    record_count=0,
                    expected_records=stage.expected_records,
                    artifact_directory=str(self._stage_dir(stage)),
                    message=blocked_reason,
                )
                self._write_state()
                continue
            self._execute_stage(stage)
        self._write_state()
        return self.output_root

    def _execute_stage(self, stage: ResearchStageSpec) -> None:
        handler = self._handlers.get(stage.stage_id)
        if handler is None:
            raise KeyError(f"No research handler for {stage.stage_id}")
        self.results[stage.stage_id] = ResearchStageResult(
            stage_id=stage.stage_id,
            status=ResearchStageStatus.RUNNING,
            record_count=0,
            expected_records=stage.expected_records,
            artifact_directory=str(self._stage_dir(stage)),
        )
        self._append_event(stage.stage_id, "started")
        self._write_state()
        try:
            result = handler(stage)
        except Exception as exc:
            self.results[stage.stage_id] = ResearchStageResult(
                stage_id=stage.stage_id,
                status=ResearchStageStatus.FAILED,
                record_count=0,
                expected_records=stage.expected_records,
                artifact_directory=str(self._stage_dir(stage)),
                message=f"{type(exc).__name__}: {exc}",
            )
            self._append_event(stage.stage_id, "failed", str(exc))
            self._write_state()
            raise
        if not result.complete:
            error = RuntimeError(
                f"Stage {stage.stage_id} produced {result.record_count}/"
                f"{result.expected_records} records"
            )
            self.results[stage.stage_id] = ResearchStageResult(
                stage_id=stage.stage_id,
                status=ResearchStageStatus.FAILED,
                record_count=result.record_count,
                expected_records=result.expected_records,
                artifact_directory=result.artifact_directory,
                gate_status=result.gate_status,
                message=str(error),
            )
            self._append_event(stage.stage_id, "failed", str(error))
            self._write_state()
            raise error
        self.results[stage.stage_id] = result
        self._append_event(stage.stage_id, "completed", result.gate_status)
        self._write_state()

    def _default_handlers(
        self,
    ) -> dict[str, Callable[[ResearchStageSpec], ResearchStageResult]]:
        return {
            PROGRAM_STAGE_CLASSIFICATION: self._run_classification_guard,
            PROGRAM_STAGE_ROW_POLICY: self._run_row_policy,
            PROGRAM_STAGE_PILOT: self._run_topology_pilot,
            PROGRAM_STAGE_CONFIRMATION: self._run_topology_confirmation,
            PROGRAM_STAGE_FULL_GRID: self._run_topology_full_grid,
            PROGRAM_STAGE_SCALING: self._run_dense_scaling,
            PROGRAM_STAGE_REPORT: self._run_final_analysis,
        }

    def _run_classification_guard(
        self,
        stage: ResearchStageSpec,
    ) -> ResearchStageResult:
        output_dir = self._stage_dir(stage)
        result = run_rmt_classification_guarded_geometry_selector_experiment(
            classification_tasks=CLASSIFICATION_GUARD_TASKS,
            models=self.config.models,
            budget_ratios=self.config.budgets,
            seeds=self.config.seeds,
            max_train_rows=self.config.max_train_rows,
            output_dir=output_dir,
            show_progress=self.config.show_progress,
        )
        selector_count = (
            len(CLASSIFICATION_GUARD_TASKS)
            * len(self.config.models)
            * len(self.config.seeds)
            * len(self.config.budgets)
        )
        gate = build_classification_geometry_gate(
            output_dir,
            expected_records=stage.expected_records,
            expected_selector_records=selector_count,
        )
        return self._completed_result(stage, len(result), gate["status"])

    def _run_row_policy(self, stage: ResearchStageSpec) -> ResearchStageResult:
        output_dir = run_rmt_conditional_row_policy_experiment(
            ConditionalRowPolicyExperimentConfig(
                regression_tasks=ROW_POLICY_REGRESSION_TASKS,
                classification_tasks=ROW_POLICY_CLASSIFICATION_TASKS,
                models=self.config.models,
                budget_ratios=self.config.budgets,
                seeds=self.config.seeds,
                max_train_rows=self.config.max_train_rows,
                output_dir=self._stage_dir(stage),
                show_progress=self.config.show_progress,
            )
        )
        metadata = self._read_json(output_dir / "run_meta.json")
        gate = self._read_json(output_dir / "phase_s1_gate.json")
        return self._completed_result(
            stage,
            int(metadata["record_count"]),
            str(gate["status"]),
        )

    def _run_topology_pilot(
        self,
        stage: ResearchStageSpec,
    ) -> ResearchStageResult:
        return self._run_topology_stage(
            stage,
            regression_tasks=PILOT_REGRESSION_TASKS,
            classification_tasks=PILOT_CLASSIFICATION_TASKS,
        )

    def _run_topology_confirmation(
        self,
        stage: ResearchStageSpec,
    ) -> ResearchStageResult:
        return self._run_topology_stage(
            stage,
            regression_tasks=CONFIRMATION_REGRESSION_TASKS,
            classification_tasks=CONFIRMATION_CLASSIFICATION_TASKS,
        )

    def _run_topology_full_grid(
        self,
        stage: ResearchStageSpec,
    ) -> ResearchStageResult:
        return self._run_topology_stage(
            stage,
            regression_tasks=FULL_REGRESSION_TASKS,
            classification_tasks=FULL_CLASSIFICATION_TASKS,
        )

    def _run_topology_stage(
        self,
        stage: ResearchStageSpec,
        *,
        regression_tasks: Sequence[str],
        classification_tasks: Sequence[str],
        budgets: Optional[Sequence[float]] = None,
    ) -> ResearchStageResult:
        output_dir = self._stage_dir(stage)
        result = run_rmt_bulk_spike_topology_real_experiment(
            regression_tasks=regression_tasks,
            classification_tasks=classification_tasks,
            models=self.config.models,
            budget_ratios=(self.config.budgets if budgets is None else budgets),
            seeds=self.config.seeds,
            max_train_rows=self.config.max_train_rows,
            output_dir=output_dir,
            show_progress=self.config.show_progress,
        )
        gate = self._read_json(output_dir / "bulk_spike_gate.json")
        return self._completed_result(stage, len(result), str(gate["status"]))

    def _run_dense_scaling(
        self,
        stage: ResearchStageSpec,
    ) -> ResearchStageResult:
        output_dir = self._stage_dir(stage)
        self._reuse_full_grid_records(output_dir)
        return self._run_topology_stage(
            stage,
            regression_tasks=SCALING_REGRESSION_TASKS,
            classification_tasks=SCALING_CLASSIFICATION_TASKS,
            budgets=self.config.scaling_budgets,
        )

    def _run_final_analysis(
        self,
        stage: ResearchStageSpec,
    ) -> ResearchStageResult:
        output_dir = self._stage_dir(stage)
        source = self._stage_dir(self.plan.stage(PROGRAM_STAGE_SCALING))
        fits = run_scaling_law_analysis(
            source / "routing_geometry_replay.csv",
            output_dir,
            bootstrap_iterations=int(self.config.bootstrap_iterations),
        )
        result = self._completed_result(stage, len(fits), "not_applicable")
        temporary = dict(self.results)
        temporary[stage.stage_id] = result
        self._write_final_report(output_dir, temporary, fits)
        return result

    def _reuse_full_grid_records(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        target_records = output_dir / "routing_geometry_runs.jsonl"
        target_references = output_dir / "bulk_spike_full_references.jsonl"
        if target_records.exists() or target_references.exists():
            return
        full_dir = self._stage_dir(self.plan.stage(PROGRAM_STAGE_FULL_GRID))
        requested = {
            *SCALING_REGRESSION_TASKS,
            *SCALING_CLASSIFICATION_TASKS,
        }
        selected_records = self._filter_jsonl_by_task(
            full_dir / "routing_geometry_runs.jsonl",
            requested,
        )
        selected_references = self._filter_jsonl_by_task(
            full_dir / "bulk_spike_full_references.jsonl",
            requested,
        )
        self._write_jsonl(target_records, selected_records)
        self._write_jsonl(target_references, selected_references)
        (output_dir / "reuse_manifest.json").write_text(
            json.dumps(
                {
                    "source": str(full_dir),
                    "reused_arm_records": len(selected_records),
                    "reused_reference_records": len(selected_references),
                    "tasks": sorted(requested),
                    "budgets": list(self.config.budgets),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    @classmethod
    def _filter_jsonl_by_task(
        cls,
        path: Path,
        requested: set[str],
    ) -> tuple[dict[str, Any], ...]:
        if not path.exists():
            raise FileNotFoundError(path)
        selected = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            task_name = str(record.get("task_name") or "")
            dataset = str(record.get("dataset") or "")
            canonical = task_name or dataset.split("__task_", maxsplit=1)[0]
            if canonical in requested:
                selected.append(record)
        return tuple(selected)

    @staticmethod
    def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
        path.write_text(
            "".join(
                json.dumps(json_ready(dict(record)), ensure_ascii=False) + "\n"
                for record in records
            ),
            encoding="utf-8",
        )

    def _completed_result(
        self,
        stage: ResearchStageSpec,
        record_count: int,
        gate_status: str,
    ) -> ResearchStageResult:
        return ResearchStageResult(
            stage_id=stage.stage_id,
            status=ResearchStageStatus.COMPLETED,
            record_count=int(record_count),
            expected_records=int(stage.expected_records),
            artifact_directory=str(self._stage_dir(stage)),
            gate_status=gate_status,
        )

    def _resolve_output_root(self) -> Path:
        if self.config.output_root is not None:
            return Path(self.config.output_root)
        suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
        return BENCHMARK_DIR / "results" / f"run_rmt_research_program_{suffix}"

    def _stage_dir(self, stage: ResearchStageSpec) -> Path:
        return self.output_root / stage.artifact_directory

    def _write_plan(self) -> None:
        path = self.output_root / "research_program_plan.json"
        config = asdict(replace(self.config, output_root=self.output_root))
        config.pop("show_progress", None)
        payload = {
            "program": "rmt_classification_bulk_spike_scaling",
            "created_at": self._timestamp(),
            "updated_at": self._timestamp(),
            "config": json_ready(config),
            "plan": self.plan.to_dict(),
        }
        if path.exists():
            existing = self._read_json(path)
            if existing.get("config") != payload["config"] or existing.get(
                "plan"
            ) != payload["plan"]:
                raise ValueError(
                    "research program resume config does not match the "
                    "existing plan"
                )
            payload["created_at"] = existing.get(
                "created_at",
                existing.get("created_or_updated_at", payload["created_at"]),
            )
        self._atomic_write_json(path, payload)

    def _load_state(self) -> dict[str, ResearchStageResult]:
        path = self.output_root / "research_program_state.json"
        if not path.exists():
            return {}
        payload = self._read_json(path)
        results = {
            stage_id: deserialize_stage_result(value)
            for stage_id, value in (payload.get("stages", {}) or {}).items()
        }
        unknown = set(results) - set(self.plan.stage_ids())
        if unknown:
            raise ValueError(f"unknown stages in research state: {sorted(unknown)}")
        for stage_id, result in results.items():
            expected = self.plan.stage(stage_id).expected_records
            if int(result.expected_records) != int(expected):
                raise ValueError(
                    f"research state count contract changed for {stage_id}"
                )
        return results

    def _write_state(self) -> None:
        statuses = {result.status for result in self.results.values()}
        status = (
            "failed"
            if ResearchStageStatus.FAILED in statuses
            else "blocked"
            if ResearchStageStatus.BLOCKED in statuses
            else "completed"
            if len(self.results) == len(self.plan.stages)
            and all(result.complete for result in self.results.values())
            else "running"
        )
        payload = {
            "status": status,
            "updated_at": self._timestamp(),
            "stages": {
                stage_id: result.to_dict()
                for stage_id, result in self.results.items()
            },
        }
        self._atomic_write_json(
            self.output_root / "research_program_state.json",
            payload,
        )

    def _append_event(self, stage_id: str, event: str, message: str = "") -> None:
        path = self.output_root / "research_program_events.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "timestamp": self._timestamp(),
                        "stage_id": stage_id,
                        "event": event,
                        "message": message,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    def _write_final_report(
        self,
        output_dir: Path,
        results: Mapping[str, ResearchStageResult],
        fits: pd.DataFrame,
    ) -> None:
        stage_rows = [
            {
                "stage": stage.stage_id,
                "status": results.get(stage.stage_id).status.value,
                "records": results.get(stage.stage_id).record_count,
                "expected": stage.expected_records,
                "gate": results.get(stage.stage_id).gate_status,
                "artifacts": stage.artifact_directory,
            }
            for stage in self.plan.stages
            if results.get(stage.stage_id) is not None
        ]
        fit_columns = [
            "dataset",
            "model",
            "arm_name",
            "primary_metric",
            "exponent",
            "exponent_confidence_interval",
            "r_squared",
            "evidence_status",
            "required_budget_by_degradation",
        ]
        lines = [
            "# Итог исследовательской программы RMT",
            "",
            "## Выполнение этапов",
            "",
            markdown_table(pd.DataFrame(stage_rows)),
            "",
            "## Законы масштабирования",
            "",
            markdown_table(fits[fit_columns]),
            "",
            "Статус `supported` означает, что показатель степени не лежит "
            "на границе, использовано не менее шести бюджетных точек и "
            "коэффициент детерминации не ниже 0,80. Остальные оценки остаются "
            "исследовательскими.",
            "",
        ]
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "research_program_report.md").write_text(
            "\n".join(lines),
            encoding="utf-8",
        )

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"Expected a JSON object in {path}")
        return value

    @staticmethod
    def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(json_ready(dict(payload)), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        temporary.replace(path)

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_rmt_bulk_spike_research_program(
    config: Optional[RMTBulkSpikeResearchProgramConfig] = None,
    *,
    selected_stages: Optional[Sequence[str]] = None,
) -> Path:
    return RMTBulkSpikeResearchProgramOrchestrator(
        config or RMTBulkSpikeResearchProgramConfig()
    ).run(selected_stages=selected_stages)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--stages", nargs="+")
    parser.add_argument("--max-train-rows", type=int, default=100_000)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    config = RMTBulkSpikeResearchProgramConfig(
        output_root=args.output_root,
        max_train_rows=args.max_train_rows,
        show_progress=not args.no_progress,
    )
    orchestrator = RMTBulkSpikeResearchProgramOrchestrator(config)
    if args.plan_only:
        orchestrator.output_root.mkdir(parents=True, exist_ok=True)
        orchestrator._write_plan()
        print(orchestrator.output_root / "research_program_plan.json")
        return
    print(orchestrator.run(selected_stages=args.stages))


if __name__ == "__main__":
    main()
