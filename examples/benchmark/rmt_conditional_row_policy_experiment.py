"""Phase S.1: cross-fitted selection between uniform and capped leverage rows."""

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
from tqdm.auto import tqdm


ROOT_DIR = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = Path(__file__).resolve().parent
for module_path in (ROOT_DIR, BENCHMARK_DIR):
    if str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))

from rmt_cross_fitted_geometry_selector_experiment import (  # noqa: E402
    DEFAULT_TAIL_GUARDED_SELECTOR_ARM_NAME,
    run_rmt_tail_guarded_geometry_selector_experiment,
)
from rmt_experiment_utils import json_ready, markdown_table  # noqa: E402
from routing_geometry_replay import (  # noqa: E402
    DEFAULT_VALIDATION_SELECTOR_ARM_NAMES,
)
from sampling_zoo.core.experiment.row_policy_selection import (  # noqa: E402
    CrossFittedRowPolicySelectionSpec,
    CrossFittedRowPolicySelector,
    RowPolicyFoldScore,
    RowSamplingPolicyRequest,
    RowSamplingPolicyResult,
    RowSamplingPolicySpec,
)


DEFAULT_PHASE_S1_REGRESSION_TASKS: tuple[str, ...] = (
    "diamonds",
    "elevators",
    "Brazilian_houses",
    "OnlineNewsPopularity",
)
DEFAULT_PHASE_S1_CLASSIFICATION_TASKS: tuple[str, ...] = (
    "adult",
    "bank-marketing",
    "covertype",
    "jannis",
)
DEFAULT_PHASE_S1_BUDGETS: tuple[float, ...] = (0.01, 0.05, 0.10, 0.20)
DEFAULT_PHASE_S1_SEEDS: tuple[int, ...] = (42, 43, 44, 45, 46)
PHASE_S1_A2_ARM = "A2_median_scaled_euclidean"
PHASE_S1_A9_ARM = DEFAULT_TAIL_GUARDED_SELECTOR_ARM_NAME

U0_POLICY = RowSamplingPolicySpec("U0_uniform", "uniform")
C1_POLICY = RowSamplingPolicySpec("C1_capped_leverage", "capped_leverage")


@dataclass(frozen=True)
class ConditionalRowPolicyExperimentConfig:
    """Frozen real-data protocol for the conditional row-policy pilot."""

    regression_tasks: Sequence[str] = DEFAULT_PHASE_S1_REGRESSION_TASKS
    classification_tasks: Sequence[str] = DEFAULT_PHASE_S1_CLASSIFICATION_TASKS
    models: Sequence[str] = ("lightgbm",)
    budget_ratios: Sequence[float] = DEFAULT_PHASE_S1_BUDGETS
    seeds: Sequence[int] = DEFAULT_PHASE_S1_SEEDS
    n_partitions: int = 5
    max_train_rows: Optional[int] = 100_000
    selection_folds: int = 5
    primary_noninferiority_margin: float = 0.005
    min_positive_fold_fraction: float = 2.0 / 3.0
    tail_noninferiority_margin: float = 0.01
    tail_risk_quantile: float = 0.90
    regression_stratification_bins: int = 10
    output_dir: Optional[Path] = None
    show_progress: bool = True

    def __post_init__(self) -> None:
        for field_name in ("regression_tasks", "classification_tasks", "models"):
            object.__setattr__(
                self,
                field_name,
                tuple(str(value) for value in getattr(self, field_name)),
            )
        object.__setattr__(
            self,
            "budget_ratios",
            tuple(float(value) for value in self.budget_ratios),
        )
        object.__setattr__(self, "seeds", tuple(int(value) for value in self.seeds))
        if self.output_dir is not None:
            object.__setattr__(self, "output_dir", Path(self.output_dir))
        if not self.regression_tasks and not self.classification_tasks:
            raise ValueError("at least one Phase S.1 task is required")
        if not self.models or not self.seeds:
            raise ValueError("models and seeds must be non-empty")
        if not self.budget_ratios or any(
            not 0.0 < value <= 1.0 for value in self.budget_ratios
        ):
            raise ValueError("budget_ratios must contain values in (0, 1]")
        if int(self.selection_folds) < 3:
            raise ValueError("selection_folds must be at least 3")

    @property
    def leaf_count(self) -> int:
        return (
            (len(self.regression_tasks) + len(self.classification_tasks))
            * len(self.models)
            * len(self.budget_ratios)
            * len(self.seeds)
        )

    @property
    def expected_official_records(self) -> int:
        return self.leaf_count * 3


class ConditionalRowPolicyExperimentOrchestrator:
    """Run two fitted policies, then derive S1 decisions from inner folds."""

    def __init__(
        self,
        config: ConditionalRowPolicyExperimentConfig,
        *,
        selector: Optional[CrossFittedRowPolicySelector] = None,
    ) -> None:
        self.config = config
        self.output_dir = self._resolve_output_dir()
        self.selector = selector or CrossFittedRowPolicySelector(
            CrossFittedRowPolicySelectionSpec(
                min_folds=int(config.selection_folds),
                primary_noninferiority_margin=float(
                    config.primary_noninferiority_margin
                ),
                min_positive_fold_fraction=float(
                    config.min_positive_fold_fraction
                ),
                tail_noninferiority_margin=float(
                    config.tail_noninferiority_margin
                ),
                random_state=int(config.seeds[0]),
            )
        )

    def run(self) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        completed_policies = []
        self._write_meta("running", completed_policies)
        try:
            for policy in tqdm(
                (U0_POLICY, C1_POLICY),
                desc="Phase S.1 fitted row policies",
                unit="policy",
                disable=not self.config.show_progress,
            ):
                self._run_policy(policy)
                completed_policies.append(policy.name)
                self._write_meta("running", completed_policies)
            official, decisions, counterfactual = self._derive_results()
            self._build_artifacts(official, decisions, counterfactual)
            completed = (
                len(official) == self.config.expected_official_records
                and bool((official["status"] == "completed").all())
            )
            self._write_meta(
                "completed" if completed else "completed_with_failures",
                completed_policies,
                record_count=len(official),
            )
        except Exception as exc:
            self._write_meta(
                "failed",
                completed_policies,
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        return self.output_dir

    def _resolve_output_dir(self) -> Path:
        if self.config.output_dir is not None:
            return Path(self.config.output_dir)
        run_id = "run_rmt_conditional_row_policy_" + datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
        return BENCHMARK_DIR / "results" / run_id

    def _run_policy(self, policy: RowSamplingPolicySpec) -> None:
        child_dir = self.output_dir / policy.name
        metadata_path = child_dir / "run_meta.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("status") == "completed":
                return
        run_rmt_tail_guarded_geometry_selector_experiment(
            regression_tasks=self.config.regression_tasks,
            classification_tasks=self.config.classification_tasks,
            models=self.config.models,
            budget_ratios=self.config.budget_ratios,
            seeds=self.config.seeds,
            n_partitions=self.config.n_partitions,
            max_train_rows=self.config.max_train_rows,
            sampler_extra_params={
                "selection_method": policy.selection_method,
                "class_allocation_policy": "proportional",
            },
            selection_folds=self.config.selection_folds,
            tail_risk_quantile=self.config.tail_risk_quantile,
            tail_noninferiority_margin=self.config.tail_noninferiority_margin,
            regression_stratification_bins=(
                self.config.regression_stratification_bins
            ),
            output_dir=child_dir,
            show_progress=self.config.show_progress,
        )

    def _derive_results(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        policy_artifacts = {
            policy.name: self._load_policy_artifacts(policy)
            for policy in (U0_POLICY, C1_POLICY)
        }
        keys = sorted(
            set(policy_artifacts[U0_POLICY.name]["a9_records"])
            & set(policy_artifacts[C1_POLICY.name]["a9_records"])
        )
        official_rows = []
        decision_rows = []
        counterfactual_rows = []
        for key in tqdm(
            keys,
            desc="Phase S.1 cross-fitted decisions",
            unit="leaf",
            disable=not self.config.show_progress,
        ):
            uniform_record = policy_artifacts[U0_POLICY.name]["a9_records"][key]
            capped_record = policy_artifacts[C1_POLICY.name]["a9_records"][key]
            for policy, record in (
                (U0_POLICY, uniform_record),
                (C1_POLICY, capped_record),
            ):
                official_rows.append(
                    self._flatten_record(record, policy.name, policy.selection_method)
                )

            a9_decision = self._select_policy(
                key=key,
                topology="S1b_fixed_A9",
                artifacts=policy_artifacts,
            )
            chosen_record = (
                capped_record
                if a9_decision.selected_policy == C1_POLICY
                else uniform_record
            )
            official_rows.append(
                {
                    **self._flatten_record(
                        chosen_record,
                        "S1_conditional",
                        a9_decision.selected_policy.selection_method,
                    ),
                    "selection": a9_decision.summary(),
                }
            )
            decision_rows.append(
                {**self._identity_from_key(key), **a9_decision.summary()}
            )

            a2_decision = self._select_policy(
                key=key,
                topology="S1a_fixed_A2",
                artifacts=policy_artifacts,
            )
            counterfactual_rows.append(
                {**self._identity_from_key(key), **a2_decision.summary()}
            )
        return (
            pd.DataFrame(official_rows),
            pd.DataFrame(decision_rows),
            pd.DataFrame(counterfactual_rows),
        )

    def _select_policy(
        self,
        *,
        key: tuple[str, int, float, str],
        topology: str,
        artifacts: Mapping[str, Mapping[str, Any]],
    ) -> RowSamplingPolicyResult:
        scores = []
        for policy in (U0_POLICY, C1_POLICY):
            policy_data = artifacts[policy.name]
            record = policy_data["a9_records"][key]
            selected_arm = (
                PHASE_S1_A2_ARM
                if topology == "S1a_fixed_A2"
                else str(record.get("selected_arm_name"))
            )
            fold_rows = policy_data["fold_rows"].get((key, selected_arm), ())
            coverage = self._class_coverage_preserved(record)
            scores.extend(
                self._row_policy_fold_score(
                    policy=policy,
                    row=row,
                    class_coverage_preserved=coverage,
                )
                for row in fold_rows
            )
        primary_metric = scores[0].primary_metric if scores else ""
        return self.selector.select(
            RowSamplingPolicyRequest(
                reference_policy=U0_POLICY,
                candidate_policy=C1_POLICY,
                fold_scores=tuple(scores),
                primary_gain_scale=(
                    "absolute" if primary_metric == "roc_auc" else "relative"
                ),
            )
        )

    def _load_policy_artifacts(
        self,
        policy: RowSamplingPolicySpec,
    ) -> dict[str, Any]:
        child_dir = self.output_dir / policy.name
        records = self._load_jsonl(child_dir / "routing_geometry_runs.jsonl")
        a9_records = {
            self._record_key(record): record
            for record in records
            if record.get("arm_name") == PHASE_S1_A9_ARM
            and record.get("status") == "completed"
        }
        fold_table = pd.read_csv(child_dir / "geometry_selection_fold_losses.csv")
        fold_rows: dict[
            tuple[tuple[str, int, float, str], str],
            list[dict[str, Any]],
        ] = {}
        for row in fold_table.to_dict(orient="records"):
            row_key = self._record_key(row)
            fold_rows.setdefault((row_key, str(row.get("arm_name"))), []).append(row)
        return {"a9_records": a9_records, "fold_rows": fold_rows}

    @staticmethod
    def _row_policy_fold_score(
        *,
        policy: RowSamplingPolicySpec,
        row: Mapping[str, Any],
        class_coverage_preserved: Optional[bool],
    ) -> RowPolicyFoldScore:
        primary_metric = str(row["primary_metric"]).lower()
        tail_value = row.get("tail_value")
        has_tail = tail_value is not None and pd.notna(tail_value)
        return RowPolicyFoldScore(
            policy_name=policy.name,
            fold_id=str(row["fold_id"]),
            primary_metric=primary_metric,
            primary_value=float(row["primary_value"]),
            primary_direction=(
                "higher" if primary_metric == "roc_auc" else "lower"
            ),
            tail_metric=str(row["tail_metric"]) if has_tail else None,
            tail_value=float(tail_value) if has_tail else None,
            tail_direction=str(row["tail_direction"]) if has_tail else None,
            class_coverage_preserved=class_coverage_preserved,
            evaluation_rows=int(row["evaluation_rows"]),
        )

    @staticmethod
    def _class_coverage_preserved(record: Mapping[str, Any]) -> Optional[bool]:
        if record.get("problem_type") != "classification":
            return None
        diagnostics = record.get("sampler_diagnostics", {}) or {}
        return bool(diagnostics.get("class_coverage_guaranteed", False))

    @staticmethod
    def _record_key(record: Mapping[str, Any]) -> tuple[str, int, float, str]:
        return (
            str(record.get("dataset")),
            int(record.get("seed")),
            round(float(record.get("budget_ratio")), 12),
            str(record.get("model")),
        )

    @staticmethod
    def _identity_from_key(
        key: tuple[str, int, float, str],
    ) -> dict[str, Any]:
        return {
            "dataset": key[0],
            "seed": key[1],
            "budget_ratio": key[2],
            "model": key[3],
        }

    @staticmethod
    def _flatten_record(
        record: Mapping[str, Any],
        row_policy_arm: str,
        selected_method: str,
    ) -> dict[str, Any]:
        test = record.get("test", {}) or {}
        metrics = test.get("metrics", {}) or {}
        return {
            "dataset": record.get("dataset"),
            "problem_type": record.get("problem_type"),
            "task_id": record.get("task_id"),
            "suite_id": record.get("suite_id"),
            "seed": record.get("seed"),
            "budget_ratio": record.get("budget_ratio"),
            "model": record.get("model"),
            "row_policy_arm": row_policy_arm,
            "selected_method": selected_method,
            "status": record.get("status"),
            "selected_geometry_arm": record.get("selected_arm_name"),
            "primary_metric": test.get("primary_metric"),
            "primary_value": test.get("primary_value"),
            "rmse": metrics.get("rmse"),
            "mae": metrics.get("mae"),
            "tail_mae": metrics.get("tail_mean_absolute_error"),
            "roc_auc": metrics.get("roc_auc"),
            "log_loss": metrics.get("log_loss"),
            "brier_score": metrics.get("brier_score"),
            "ece": metrics.get("expected_calibration_error"),
            "f1_macro": metrics.get("f1_macro"),
            "worst_class_recall": metrics.get("worst_class_recall"),
        }

    def _build_artifacts(
        self,
        official: pd.DataFrame,
        decisions: pd.DataFrame,
        counterfactual: pd.DataFrame,
    ) -> None:
        official.to_csv(self.output_dir / "phase_s1_raw.csv", index=False)
        decisions.to_csv(self.output_dir / "phase_s1_decisions.csv", index=False)
        counterfactual.to_csv(
            self.output_dir / "phase_s1_a2_counterfactual.csv",
            index=False,
        )
        paired = self._pair_with_uniform(official)
        paired.to_csv(self.output_dir / "phase_s1_paired.csv", index=False)
        summary = self._summarize(paired)
        summary.to_csv(self.output_dir / "phase_s1_summary.csv", index=False)
        gate = self._evaluate_gate(paired, decisions)
        (self.output_dir / "phase_s1_gate.json").write_text(
            json.dumps(json_ready(gate), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        self._write_report(summary, gate)
        self._write_figure(summary)

    @staticmethod
    def _pair_with_uniform(raw: pd.DataFrame) -> pd.DataFrame:
        keys = ["dataset", "problem_type", "seed", "budget_ratio", "model"]
        baseline = raw[raw["row_policy_arm"] == U0_POLICY.name][
            keys + ["primary_value", "tail_mae"]
        ].rename(
            columns={
                "primary_value": "uniform_primary_value",
                "tail_mae": "uniform_tail_mae",
            }
        )
        paired = raw.merge(baseline, on=keys, how="left", validate="many_to_one")
        lower = paired["primary_metric"].isin({"rmse", "log_loss"})
        paired["primary_gain_vs_uniform"] = np.where(
            lower,
            (
                paired["uniform_primary_value"] - paired["primary_value"]
            )
            / paired["uniform_primary_value"].clip(lower=1e-12),
            paired["primary_value"] - paired["uniform_primary_value"],
        )
        paired["tail_gain_vs_uniform"] = np.where(
            paired["tail_mae"].notna(),
            (paired["uniform_tail_mae"] - paired["tail_mae"])
            / paired["uniform_tail_mae"].clip(lower=1e-12),
            np.nan,
        )
        return paired

    @staticmethod
    def _summarize(paired: pd.DataFrame) -> pd.DataFrame:
        return (
            paired.groupby(
                ["dataset", "problem_type", "budget_ratio", "row_policy_arm"],
                dropna=False,
            )
            .agg(
                runs=("seed", "size"),
                primary_gain_mean=("primary_gain_vs_uniform", "mean"),
                primary_gain_median=("primary_gain_vs_uniform", "median"),
                win_rate=(
                    "primary_gain_vs_uniform",
                    lambda values: float((values > 0.0).mean()),
                ),
                tail_gain_mean=("tail_gain_vs_uniform", "mean"),
            )
            .reset_index()
        )

    @staticmethod
    def _evaluate_gate(
        paired: pd.DataFrame,
        decisions: pd.DataFrame,
    ) -> dict[str, Any]:
        selected_capped = decisions[decisions["selected_policy"] == C1_POLICY.name]
        selection_rate = float(len(selected_capped) / max(len(decisions), 1))
        selected_keys = {
            (
                row.dataset,
                int(row.seed),
                round(float(row.budget_ratio), 12),
                row.model,
            )
            for row in selected_capped.itertuples()
        }
        capped = paired[paired["row_policy_arm"] == C1_POLICY.name]
        capped_selected = capped[
            capped.apply(
                lambda row: (
                    row["dataset"],
                    int(row["seed"]),
                    round(float(row["budget_ratio"]), 12),
                    row["model"],
                )
                in selected_keys,
                axis=1,
            )
        ]
        win_rate = (
            float((capped_selected["primary_gain_vs_uniform"] > 0.0).mean())
            if not capped_selected.empty
            else 0.0
        )
        passed = selection_rate >= 0.10 and win_rate >= 0.70
        return {
            "status": "passed" if passed else "failed",
            "capped_selection_rate": selection_rate,
            "minimum_capped_selection_rate": 0.10,
            "capped_win_rate_when_selected": win_rate,
            "minimum_capped_win_rate_when_selected": 0.70,
            "selected_capped_runs": int(len(capped_selected)),
        }

    def _write_report(
        self,
        summary: pd.DataFrame,
        gate: Mapping[str, Any],
    ) -> None:
        lines = [
            "# Phase S.1: условный выбор политики строк",
            "",
            f"- Статус критерия допуска: `{gate['status']}`.",
            (
                "- Доля решений в пользу capped leverage: "
                f"`{gate['capped_selection_rate']:.3f}`."
            ),
            (
                "- Доля выигрышей capped leverage среди выбранных случаев: "
                f"`{gate['capped_win_rate_when_selected']:.3f}`."
            ),
            "- Официальный S1 использует A9; решения для фиксированного A2 сохранены отдельно.",
            "",
            "## Результаты по датасетам и бюджетам",
            "",
            markdown_table(summary),
        ]
        (self.output_dir / "phase_s1_report.md").write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )

    def _write_figure(self, summary: pd.DataFrame) -> None:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return
        selected = summary[summary["row_policy_arm"] == "S1_conditional"]
        figure, axis = plt.subplots(figsize=(9, 5.5))
        for dataset, values in selected.groupby("dataset"):
            axis.plot(
                values["budget_ratio"],
                values["primary_gain_mean"],
                marker="o",
                linewidth=1.5,
                label=dataset,
            )
        axis.axhline(0.0, color="#555555", linewidth=0.8)
        axis.set_xlabel("Доля бюджета")
        axis.set_ylabel("Средний выигрыш S1 относительно uniform")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7, ncol=2)
        figure.tight_layout()
        figure.savefig(self.output_dir / "phase_s1_gain_by_budget.png", dpi=180)
        plt.close(figure)

    def _write_meta(
        self,
        status: str,
        completed_policies: Sequence[str],
        *,
        record_count: int = 0,
        error: Optional[str] = None,
    ) -> None:
        config = asdict(self.config)
        config["output_dir"] = str(self.output_dir)
        payload = {
            "experiment": "rmt_conditional_row_policy_phase_s1",
            "status": status,
            "updated_at": datetime.now().isoformat(),
            "completed_policies": list(completed_policies),
            "record_count": int(record_count),
            "expected_official_records": self.config.expected_official_records,
            "config": json_ready(config),
            "selector_spec": json_ready(asdict(self.selector.spec)),
            "error": error,
        }
        (self.output_dir / "run_meta.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @staticmethod
    def _load_jsonl(path: Path) -> list[dict[str, Any]]:
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
        return records


def run_rmt_conditional_row_policy_experiment(
    config: Optional[ConditionalRowPolicyExperimentConfig] = None,
) -> Path:
    return ConditionalRowPolicyExperimentOrchestrator(
        config or ConditionalRowPolicyExperimentConfig()
    ).run()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-train-rows", type=int, default=100_000)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = ConditionalRowPolicyExperimentConfig(
        regression_tasks=("diamonds",) if args.smoke else DEFAULT_PHASE_S1_REGRESSION_TASKS,
        classification_tasks=() if args.smoke else DEFAULT_PHASE_S1_CLASSIFICATION_TASKS,
        budget_ratios=(0.10,) if args.smoke else DEFAULT_PHASE_S1_BUDGETS,
        seeds=(42,) if args.smoke else DEFAULT_PHASE_S1_SEEDS,
        max_train_rows=min(args.max_train_rows, 10_000) if args.smoke else args.max_train_rows,
        output_dir=args.output_dir,
        show_progress=not args.no_progress,
    )
    print(run_rmt_conditional_row_policy_experiment(config))


if __name__ == "__main__":
    main()
