"""Build the AMLB-compatible safety gate for classification routing geometry."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = Path(__file__).resolve().parent
for module_path in (ROOT_DIR, BENCHMARK_DIR):
    if str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))

from rmt_cross_fitted_geometry_selector_experiment import (  # noqa: E402
    DEFAULT_CLASSIFICATION_GUARDED_SELECTOR_ARM_NAME,
)
from rmt_experiment_utils import json_ready, markdown_table  # noqa: E402


CLASSIFICATION_REFERENCE_ARM = "A2_median_scaled_euclidean"


@dataclass(frozen=True)
class ClassificationGeometryGateSpec:
    expected_records: int
    expected_selector_records: int
    roc_auc_absolute_margin: float = 0.005
    log_loss_relative_margin: float = 0.01
    min_nonbaseline_win_rate: float = 0.70

    def __post_init__(self) -> None:
        if int(self.expected_records) < 1:
            raise ValueError("expected_records must be positive")
        if int(self.expected_selector_records) < 1:
            raise ValueError("expected_selector_records must be positive")
        if float(self.roc_auc_absolute_margin) < 0.0:
            raise ValueError("roc_auc_absolute_margin must be non-negative")
        if float(self.log_loss_relative_margin) < 0.0:
            raise ValueError("log_loss_relative_margin must be non-negative")
        if not 0.0 <= float(self.min_nonbaseline_win_rate) <= 1.0:
            raise ValueError("min_nonbaseline_win_rate must be in [0, 1]")


class ClassificationGeometryGateAnalyzer:
    def __init__(self, spec: ClassificationGeometryGateSpec) -> None:
        self.spec = spec

    def build(self, run_dir: Path) -> dict[str, Any]:
        run_dir = Path(run_dir)
        runs = pd.read_csv(run_dir / "routing_geometry_replay.csv")
        records = self._load_jsonl(run_dir / "routing_geometry_runs.jsonl")
        paired = self._pair_selector_with_reference(runs)
        gate = self._evaluate(runs, records, paired)
        paired.to_csv(
            run_dir / "classification_geometry_guard_paired.csv",
            index=False,
        )
        (run_dir / "classification_geometry_gate.json").write_text(
            json.dumps(json_ready(gate), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        self._write_report(run_dir, paired, gate)
        return gate

    @staticmethod
    def _load_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
        if not path.exists():
            raise FileNotFoundError(path)
        records = []
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL record {line_number} must be an object")
            records.append(value)
        return tuple(records)

    def _pair_selector_with_reference(self, runs: pd.DataFrame) -> pd.DataFrame:
        keys = ["dataset", "seed", "budget_ratio", "model"]
        reference = runs[
            runs["arm_name"] == CLASSIFICATION_REFERENCE_ARM
        ][keys + ["test_primary_metric", "test_primary_value"]].rename(
            columns={
                "test_primary_metric": "reference_primary_metric",
                "test_primary_value": "reference_primary_value",
            }
        )
        selected = runs[
            runs["arm_name"]
            == DEFAULT_CLASSIFICATION_GUARDED_SELECTOR_ARM_NAME
        ].copy()
        paired = selected.merge(
            reference,
            on=keys,
            how="left",
            validate="one_to_one",
        )
        metric_matches = (
            paired["test_primary_metric"]
            == paired["reference_primary_metric"]
        )
        paired["metric_contract_valid"] = metric_matches
        lower_is_better = paired["test_primary_metric"].eq("log_loss")
        paired["gain_vs_a2"] = np.where(
            lower_is_better,
            (
                paired["reference_primary_value"]
                - paired["test_primary_value"]
            )
            / paired["reference_primary_value"].clip(lower=1e-12),
            paired["test_primary_value"]
            - paired["reference_primary_value"],
        )
        paired["allowed_harm"] = np.where(
            lower_is_better,
            -float(self.spec.log_loss_relative_margin),
            -float(self.spec.roc_auc_absolute_margin),
        )
        paired["within_harm_margin"] = (
            paired["gain_vs_a2"] >= paired["allowed_harm"]
        )
        paired["selected_nonbaseline"] = (
            paired["selected_arm_name"] != CLASSIFICATION_REFERENCE_ARM
        )
        return paired

    def _evaluate(
        self,
        runs: pd.DataFrame,
        records: Sequence[Mapping[str, Any]],
        paired: pd.DataFrame,
    ) -> dict[str, Any]:
        completed = runs[runs["status"] == "completed"]
        failed_count = int((runs["status"] == "failed").sum())
        metric_contract = self._metric_contract(completed)
        coverage_values = [
            bool((record.get("sampler_diagnostics", {}) or {}).get(
                "class_coverage_guaranteed",
                False,
            ))
            for record in records
            if record.get("status") == "completed"
        ]
        nonbaseline = paired[paired["selected_nonbaseline"]]
        win_rate = (
            float((nonbaseline["gain_vs_a2"] > 0.0).mean())
            if not nonbaseline.empty
            else None
        )
        safety_checks = {
            "expected_record_count": len(runs) == self.spec.expected_records,
            "all_records_completed": (
                len(completed) == self.spec.expected_records
                and failed_count == 0
            ),
            "expected_selector_record_count": (
                len(paired) == self.spec.expected_selector_records
            ),
            "primary_metric_contract": metric_contract,
            "class_coverage_guaranteed": (
                bool(coverage_values) and all(coverage_values)
            ),
            "metric_alignment": bool(paired["metric_contract_valid"].all()),
            "selected_policy_noninferior": bool(
                paired["within_harm_margin"].all()
            ),
        }
        effectiveness_passed = (
            win_rate is None
            or win_rate >= float(self.spec.min_nonbaseline_win_rate)
        )
        safety_passed = all(safety_checks.values())
        return {
            "status": "passed" if safety_passed else "failed",
            "evidence_status": (
                "safe_and_effective"
                if safety_passed and effectiveness_passed
                else "safe_fallback_only"
                if safety_passed
                else "unsafe"
            ),
            "spec": asdict(self.spec),
            "checks": safety_checks,
            "record_count": int(len(runs)),
            "failed_count": failed_count,
            "selector_record_count": int(len(paired)),
            "selected_nonbaseline_count": int(len(nonbaseline)),
            "selected_nonbaseline_win_rate": win_rate,
            "selector_effectiveness_passed": effectiveness_passed,
            "worst_gain_vs_a2": (
                None if paired.empty else float(paired["gain_vs_a2"].min())
            ),
        }

    @staticmethod
    def _metric_contract(completed: pd.DataFrame) -> bool:
        if completed.empty:
            return False
        binary = completed[completed["test_primary_metric"] == "roc_auc"]
        multiclass = completed[
            completed["test_primary_metric"] == "log_loss"
        ]
        valid_names = completed["test_primary_metric"].isin(
            {"roc_auc", "log_loss"}
        ).all()
        binary_probabilities = (
            binary.empty
            or (
                binary["test_roc_auc"].notna().all()
                and binary["test_log_loss"].notna().all()
            )
        )
        multiclass_probabilities = (
            multiclass.empty or multiclass["test_log_loss"].notna().all()
        )
        return bool(valid_names and binary_probabilities and multiclass_probabilities)

    @staticmethod
    def _write_report(
        run_dir: Path,
        paired: pd.DataFrame,
        gate: Mapping[str, Any],
    ) -> None:
        summary = (
            paired.groupby(
                ["dataset", "test_primary_metric"],
                dropna=False,
            )
            .agg(
                runs=("seed", "size"),
                nonbaseline_selections=("selected_nonbaseline", "sum"),
                mean_gain_vs_a2=("gain_vs_a2", "mean"),
                median_gain_vs_a2=("gain_vs_a2", "median"),
                worst_gain_vs_a2=("gain_vs_a2", "min"),
            )
            .reset_index()
        )
        lines = [
            "# Критерий безопасности геометрии роутинга для классификации",
            "",
            f"Статус: **{gate['status']}**.",
            f"Характер свидетельства: `{gate['evidence_status']}`.",
            "",
            "Проверка использует ROC AUC для бинарных задач и log loss для "
            "многоклассовых задач. Безопасность и эффективность небазового "
            "выбора оцениваются раздельно.",
            "",
            "## Проверки",
            "",
            *[
                f"- `{name}`: `{value}`."
                for name, value in gate["checks"].items()
            ],
            "",
            "## Результаты по датасетам",
            "",
            markdown_table(summary),
            "",
        ]
        (run_dir / "classification_geometry_gate_report.md").write_text(
            "\n".join(lines),
            encoding="utf-8",
        )


def build_classification_geometry_gate(
    run_dir: Path,
    *,
    expected_records: int,
    expected_selector_records: int,
) -> dict[str, Any]:
    return ClassificationGeometryGateAnalyzer(
        ClassificationGeometryGateSpec(
            expected_records=expected_records,
            expected_selector_records=expected_selector_records,
        )
    ).build(run_dir)
