"""Real-data Phase S with frozen A9 routing and exact row-sketch policies."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


DEFAULT_REAL_REGRESSION_TASKS: tuple[str, ...] = (
    "diamonds",
    "OnlineNewsPopularity",
)
DEFAULT_REAL_CLASSIFICATION_TASKS: tuple[str, ...] = (
    "bank-marketing",
    "covertype",
)
DEFAULT_REAL_BUDGETS: tuple[float, ...] = (0.05, 0.10, 0.20)
DEFAULT_REAL_SEEDS: tuple[int, ...] = (11, 29, 47)
A9_ARM_NAME = "A9_tail_guarded_cross_fitted_selected"


@dataclass(frozen=True)
class RealLeverageSketchArm:
    arm_id: str
    selection_method: str
    training_reweighting: str = "none"
    leverage_mixture_alpha: float = 0.25
    leverage_cap_quantile: float = 0.95

    def sampler_params(self) -> dict[str, Any]:
        return {
            "selection_method": self.selection_method,
            "training_reweighting": self.training_reweighting,
            "leverage_mixture_alpha": float(self.leverage_mixture_alpha),
            "leverage_cap_quantile": float(self.leverage_cap_quantile),
        }


DEFAULT_REAL_ARMS: tuple[RealLeverageSketchArm, ...] = (
    RealLeverageSketchArm("R0_uniform", "uniform"),
    RealLeverageSketchArm("R1_capped_incumbent", "capped_leverage"),
    RealLeverageSketchArm("R2_robust_mixture", "robust_leverage_mixture"),
    RealLeverageSketchArm(
        "R3_robust_mixture_ipw",
        "robust_leverage_mixture",
        training_reweighting="inverse_probability",
    ),
    RealLeverageSketchArm(
        "R4_ridge_leverage_ipw",
        "saturated_ridge_leverage",
        training_reweighting="inverse_probability",
    ),
)


@dataclass(frozen=True)
class RealLeverageSketchConfig:
    regression_tasks: Sequence[str] = DEFAULT_REAL_REGRESSION_TASKS
    classification_tasks: Sequence[str] = DEFAULT_REAL_CLASSIFICATION_TASKS
    models: Sequence[str] = ("lightgbm",)
    budget_ratios: Sequence[float] = DEFAULT_REAL_BUDGETS
    seeds: Sequence[int] = DEFAULT_REAL_SEEDS
    arms: Sequence[RealLeverageSketchArm] = DEFAULT_REAL_ARMS
    max_train_rows: int | None = 50_000
    selection_folds: int = 5
    n_partitions: int = 5
    tail_risk_quantile: float = 0.90
    tail_noninferiority_margin: float = 0.005
    regression_stratification_bins: int = 10
    output_dir: Path | None = None
    show_progress: bool = True

    def __post_init__(self) -> None:
        for name in ("regression_tasks", "classification_tasks", "models"):
            object.__setattr__(
                self,
                name,
                tuple(str(value) for value in getattr(self, name)),
            )
        object.__setattr__(
            self,
            "budget_ratios",
            tuple(float(value) for value in self.budget_ratios),
        )
        object.__setattr__(self, "seeds", tuple(int(value) for value in self.seeds))
        object.__setattr__(self, "arms", tuple(self.arms))
        if self.output_dir is not None:
            object.__setattr__(self, "output_dir", Path(self.output_dir))
        if not self.regression_tasks and not self.classification_tasks:
            raise ValueError("at least one real-data task is required")
        if not self.models or not self.seeds or not self.arms:
            raise ValueError("models, seeds, and arms must not be empty")
        if any(not 0.0 < value <= 1.0 for value in self.budget_ratios):
            raise ValueError("budget_ratios must contain values in (0, 1]")
        if len({arm.arm_id for arm in self.arms}) != len(self.arms):
            raise ValueError("arm_id values must be unique")
        if int(self.selection_folds) < 3:
            raise ValueError("selection_folds must be at least 3")
        if int(self.n_partitions) < 1:
            raise ValueError("n_partitions must be positive")

    @property
    def expected_a9_records(self) -> int:
        task_count = len(self.regression_tasks) + len(self.classification_tasks)
        return (
            task_count
            * len(self.models)
            * len(self.budget_ratios)
            * len(self.seeds)
            * len(self.arms)
        )


class RealLeverageSketchOrchestrator:
    """Run one resumable A9 child experiment per row-sketch policy."""

    def __init__(self, config: RealLeverageSketchConfig) -> None:
        self.config = config
        self.output_dir = self._resolve_output_dir()

    def run(self) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        completed_arms: list[str] = []
        self._write_meta("running", completed_arms)
        try:
            for arm in tqdm(
                self.config.arms,
                desc="Real-data leverage Phase S",
                disable=not self.config.show_progress,
                unit="arm",
            ):
                self._run_arm(arm)
                completed_arms.append(arm.arm_id)
                self._write_meta("running", completed_arms)
            raw = self._collect_a9_records()
            self._build_artifacts(raw)
            status = (
                "completed"
                if len(raw) == self.config.expected_a9_records
                and (raw["status"] == "completed").all()
                else "completed_with_failures"
            )
            self._write_meta(status, completed_arms, record_count=len(raw))
        except Exception as exc:
            self._write_meta(
                "failed",
                completed_arms,
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        return self.output_dir

    def _resolve_output_dir(self) -> Path:
        if self.config.output_dir is not None:
            return Path(self.config.output_dir)
        run_id = "run_rmt_leverage_sketch_real_" + datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
        return Path(__file__).resolve().parent / "results" / run_id

    def _run_arm(self, arm: RealLeverageSketchArm) -> None:
        child_dir = self.output_dir / arm.arm_id
        metadata_path = child_dir / "run_meta.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("status") == "completed":
                return
        from rmt_cross_fitted_geometry_selector_experiment import (
            run_rmt_tail_guarded_geometry_selector_experiment,
        )

        run_rmt_tail_guarded_geometry_selector_experiment(
            regression_tasks=self.config.regression_tasks,
            classification_tasks=self.config.classification_tasks,
            models=self.config.models,
            budget_ratios=self.config.budget_ratios,
            seeds=self.config.seeds,
            n_partitions=self.config.n_partitions,
            max_train_rows=self.config.max_train_rows,
            selection_folds=self.config.selection_folds,
            tail_risk_quantile=self.config.tail_risk_quantile,
            tail_noninferiority_margin=self.config.tail_noninferiority_margin,
            regression_stratification_bins=(
                self.config.regression_stratification_bins
            ),
            sampler_extra_params=arm.sampler_params(),
            output_dir=child_dir,
            show_progress=self.config.show_progress,
        )

    def _collect_a9_records(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for arm in self.config.arms:
            path = self.output_dir / arm.arm_id / "routing_geometry_runs.jsonl"
            if not path.exists():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("arm_name") != A9_ARM_NAME:
                    continue
                rows.append(self._flatten_a9_record(record, arm))
        return pd.DataFrame(rows)

    @staticmethod
    def _flatten_a9_record(
        record: Mapping[str, Any],
        arm: RealLeverageSketchArm,
    ) -> dict[str, Any]:
        test = record.get("test", {}) or {}
        metrics = test.get("metrics", {}) or {}
        diagnostics = record.get("sampler_diagnostics", {}) or {}
        subspace_values = tuple(
            (diagnostics.get("partition_subspace_preservation", {}) or {}).values()
        )
        sketch_values = tuple(
            (diagnostics.get("partition_sketch_plans", {}) or {}).values()
        )

        def mean_nested(values: Sequence[Mapping[str, Any]], key: str) -> float | None:
            observed = [
                float(value[key])
                for value in values
                if value.get(key) is not None
            ]
            return float(np.mean(observed)) if observed else None

        inclusion_values = [
            value.get("inclusion", {}) or {} for value in sketch_values
        ]
        return {
            "dataset": record.get("dataset"),
            "problem_type": record.get("problem_type"),
            "seed": record.get("seed"),
            "budget_ratio": record.get("budget_ratio"),
            "model": record.get("model"),
            "row_arm_id": arm.arm_id,
            "selection_method": arm.selection_method,
            "training_reweighting": arm.training_reweighting,
            "status": record.get("status"),
            "selected_geometry_arm": record.get("selected_arm_name"),
            "primary_metric": test.get("primary_metric"),
            "primary_value": test.get("primary_value"),
            "rmse": metrics.get("rmse"),
            "mae": metrics.get("mae"),
            "tail_mae": metrics.get("tail_mae"),
            "roc_auc": metrics.get("roc_auc"),
            "log_loss": metrics.get("log_loss"),
            "n_experts": record.get("n_experts"),
            "total_train_rows": int(sum((record.get("partition_sizes", {}) or {}).values())),
            "ensemble_fit_seconds": diagnostics.get("ensemble_fit_seconds"),
            "sampler_fit_seconds": (
                (diagnostics.get("runtime", {}) or {}).get("total_seconds")
            ),
            "gram_relative_error_mean": mean_nested(
                subspace_values,
                "gram_relative_error",
            ),
            "projection_cost_error_mean": mean_nested(
                subspace_values,
                "projection_cost_relative_error",
            ),
            "principal_angle_max_mean": mean_nested(
                subspace_values,
                "max_principal_angle_degrees",
            ),
            "effective_sample_size_mean": mean_nested(
                sketch_values,
                "effective_sample_size",
            ),
            "inclusion_probability_min_mean": mean_nested(
                inclusion_values,
                "min_probability",
            ),
            "inclusion_probability_max_mean": mean_nested(
                inclusion_values,
                "max_probability",
            ),
        }

    def _build_artifacts(self, raw: pd.DataFrame) -> None:
        raw.to_csv(self.output_dir / "phase_s_real_raw.csv", index=False)
        paired = self._pair_with_uniform(raw)
        paired.to_csv(self.output_dir / "phase_s_real_paired.csv", index=False)
        summary = self._summarize(paired)
        summary.to_csv(self.output_dir / "phase_s_real_summary.csv", index=False)
        gate = self._evaluate_gate(paired)
        (self.output_dir / "phase_s_real_gate.json").write_text(
            json.dumps(gate, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        self._write_report(raw, summary, gate)
        self._write_figures(summary)

    @staticmethod
    def _pair_with_uniform(raw: pd.DataFrame) -> pd.DataFrame:
        keys = ["dataset", "problem_type", "seed", "budget_ratio", "model"]
        baseline_columns = keys + [
            "primary_value",
            "tail_mae",
            "ensemble_fit_seconds",
            "gram_relative_error_mean",
        ]
        baseline = raw[raw["row_arm_id"] == "R0_uniform"][baseline_columns].rename(
            columns={
                "primary_value": "uniform_primary_value",
                "tail_mae": "uniform_tail_mae",
                "ensemble_fit_seconds": "uniform_fit_seconds",
                "gram_relative_error_mean": "uniform_gram_error",
            }
        )
        paired = raw.merge(baseline, on=keys, how="left", validate="many_to_one")
        lower_is_better = paired["primary_metric"].isin({"rmse", "log_loss"})
        paired["primary_gain_vs_uniform"] = np.where(
            lower_is_better,
            (
                paired["uniform_primary_value"] - paired["primary_value"]
            ) / paired["uniform_primary_value"].clip(lower=1e-12),
            paired["primary_value"] - paired["uniform_primary_value"],
        )
        paired["tail_gain_vs_uniform"] = (
            paired["uniform_tail_mae"] - paired["tail_mae"]
        ) / paired["uniform_tail_mae"].clip(lower=1e-12)
        paired["fit_time_ratio_vs_uniform"] = (
            paired["ensemble_fit_seconds"]
            / paired["uniform_fit_seconds"].clip(lower=1e-12)
        )
        paired["gram_error_delta_vs_uniform"] = (
            paired["gram_relative_error_mean"] - paired["uniform_gram_error"]
        )
        return paired

    @staticmethod
    def _summarize(paired: pd.DataFrame) -> pd.DataFrame:
        return (
            paired.groupby(
                [
                    "dataset",
                    "problem_type",
                    "budget_ratio",
                    "row_arm_id",
                    "selection_method",
                    "training_reweighting",
                ],
                dropna=False,
            )
            .agg(
                runs=("seed", "size"),
                primary_value_mean=("primary_value", "mean"),
                primary_gain_mean=("primary_gain_vs_uniform", "mean"),
                primary_gain_median=("primary_gain_vs_uniform", "median"),
                tail_gain_mean=("tail_gain_vs_uniform", "mean"),
                gram_error_mean=("gram_relative_error_mean", "mean"),
                projection_cost_error_mean=("projection_cost_error_mean", "mean"),
                effective_sample_size_mean=("effective_sample_size_mean", "mean"),
                fit_time_seconds_mean=("ensemble_fit_seconds", "mean"),
                fit_time_ratio_mean=("fit_time_ratio_vs_uniform", "mean"),
            )
            .reset_index()
        )

    @staticmethod
    def _evaluate_gate(paired: pd.DataFrame) -> dict[str, Any]:
        candidate = paired[paired["row_arm_id"] == "R3_robust_mixture_ipw"]
        by_dataset = (
            candidate.groupby("dataset")
            .agg(
                mean_gain=("primary_gain_vs_uniform", "mean"),
                median_gain=("primary_gain_vs_uniform", "median"),
                worst_gain=("primary_gain_vs_uniform", "min"),
                mean_tail_gain=("tail_gain_vs_uniform", "mean"),
            )
            .reset_index()
        )
        nonnegative_fraction = (
            float(np.mean(by_dataset["mean_gain"] >= 0.0))
            if not by_dataset.empty
            else 0.0
        )
        median_gain = (
            float(candidate["primary_gain_vs_uniform"].median())
            if not candidate.empty
            else float("nan")
        )
        passed = (
            not candidate.empty
            and nonnegative_fraction >= 0.75
            and median_gain >= -0.005
            and bool((candidate["status"] == "completed").all())
        )
        return {
            "status": "passed" if passed else "failed",
            "candidate_arm": "R3_robust_mixture_ipw",
            "dataset_nonnegative_fraction": nonnegative_fraction,
            "required_dataset_nonnegative_fraction": 0.75,
            "median_primary_gain": median_gain,
            "minimum_median_primary_gain": -0.005,
            "by_dataset": by_dataset.to_dict("records"),
        }

    def _write_report(
        self,
        raw: pd.DataFrame,
        summary: pd.DataFrame,
        gate: Mapping[str, Any],
    ) -> None:
        ranking = (
            summary.groupby("row_arm_id")
            .agg(
                mean_gain=("primary_gain_mean", "mean"),
                median_gain=("primary_gain_median", "median"),
                mean_tail_gain=("tail_gain_mean", "mean"),
                mean_gram_error=("gram_error_mean", "mean"),
                mean_fit_time_ratio=("fit_time_ratio_mean", "mean"),
            )
            .sort_values("mean_gain", ascending=False)
            .reset_index()
        )
        lines = [
            "# Phase S на реальных данных",
            "",
            f"- Статус gate: `{gate['status']}`",
            f"- A9-записей: `{len(raw)}/{self.config.expected_a9_records}`",
            "- Геометрия роутинга выбирается неизменённым A9-селектором.",
            "- Между arms меняются только политика выбора строк и режим весов.",
            "",
            "## Общий рейтинг",
            "",
            ranking.to_markdown(index=False, floatfmt=".6f"),
            "",
            "## Gate для R3_robust_mixture_ipw",
            "",
            pd.DataFrame(gate["by_dataset"]).to_markdown(index=False, floatfmt=".6f"),
            "",
            "Положительный gain означает улучшение относительно `R0_uniform` при одинаковых датасете, бюджете, модели и начальном значении генератора.",
        ]
        (self.output_dir / "phase_s_real_report.md").write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )

    def _write_figures(self, summary: pd.DataFrame) -> None:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return
        plot_data = (
            summary.groupby(["budget_ratio", "row_arm_id"])
            .agg(
                gain=("primary_gain_mean", "mean"),
                gram=("gram_error_mean", "mean"),
                fit_ratio=("fit_time_ratio_mean", "mean"),
            )
            .reset_index()
        )
        for metric, ylabel, filename in (
            ("gain", "Средний gain относительно uniform", "phase_s_real_gain.png"),
            ("gram", "Средняя ошибка Gram-матрицы", "phase_s_real_gram.png"),
            ("fit_ratio", "Отношение времени обучения к uniform", "phase_s_real_fit_time.png"),
        ):
            figure, axis = plt.subplots(figsize=(10, 5.5))
            for arm_id, values in plot_data.groupby("row_arm_id"):
                axis.plot(
                    values["budget_ratio"],
                    values[metric],
                    marker="o",
                    linewidth=1.8,
                    label=arm_id,
                )
            axis.axhline(0.0 if metric != "fit_ratio" else 1.0, color="#555555", linewidth=0.8)
            axis.set_xlabel("Доля бюджета")
            axis.set_ylabel(ylabel)
            axis.grid(alpha=0.25)
            axis.legend(fontsize=8, ncol=2)
            figure.tight_layout()
            figure.savefig(self.output_dir / filename, dpi=180)
            plt.close(figure)

    def _write_meta(
        self,
        status: str,
        completed_arms: Sequence[str],
        *,
        record_count: int = 0,
        error: str | None = None,
    ) -> None:
        config_payload = asdict(self.config)
        config_payload["output_dir"] = str(self.output_dir)
        config_payload["arms"] = [asdict(arm) for arm in self.config.arms]
        payload = {
            "experiment": "rmt_leverage_sketch_real_phase_s",
            "status": status,
            "completed_arms": list(completed_arms),
            "record_count": int(record_count),
            "expected_a9_records": int(self.config.expected_a9_records),
            "updated_at": datetime.now().isoformat(),
            "config": config_payload,
            "error": error,
        }
        (self.output_dir / "run_meta.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def run_rmt_leverage_sketch_real(
    config: RealLeverageSketchConfig | None = None,
) -> Path:
    return RealLeverageSketchOrchestrator(
        config or RealLeverageSketchConfig()
    ).run()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-train-rows", type=int, default=50_000)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = RealLeverageSketchConfig(
        regression_tasks=("diamonds",) if args.smoke else DEFAULT_REAL_REGRESSION_TASKS,
        classification_tasks=() if args.smoke else DEFAULT_REAL_CLASSIFICATION_TASKS,
        budget_ratios=(0.10,) if args.smoke else DEFAULT_REAL_BUDGETS,
        seeds=(11,) if args.smoke else DEFAULT_REAL_SEEDS,
        max_train_rows=min(args.max_train_rows, 10_000) if args.smoke else args.max_train_rows,
        output_dir=args.output_dir,
        show_progress=not args.no_progress,
    )
    output_dir = run_rmt_leverage_sketch_real(config)
    print(f"Real-data Phase S completed: {output_dir}")


if __name__ == "__main__":
    main()
