"""Independent regression confirmation for the A9 routing selector."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Sequence

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = Path(__file__).resolve().parent
for module_path in (ROOT_DIR, BENCHMARK_DIR):
    if str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))

from rmt_experiment_utils import markdown_table  # noqa: E402

REFERENCE_ARM = "A2_median_scaled_euclidean"
DEFAULT_TAIL_GUARDED_SELECTOR_ARM_NAME = "A9_tail_guarded_cross_fitted_selected"
DEFAULT_HOLDOUT_TASKS: tuple[str, ...] = (
    "house_16H",
    "house_sales",
    "pol",
    "topo_2_1",
)
DEFAULT_HOLDOUT_BUDGETS: tuple[float, ...] = (0.01, 0.05, 0.10, 0.20)
DEFAULT_HOLDOUT_SEEDS: tuple[int, ...] = (42, 43, 44, 45, 46)


@dataclass(frozen=True)
class TailGuardHoldoutGateSpec:
    """Pre-registered acceptance criteria for the independent A9 check."""

    min_nonnegative_dataset_fraction: float = 0.75
    min_nonbaseline_win_rate: float = 0.80
    max_primary_relative_degradation: float = 0.005
    max_tail_relative_degradation: float = 0.01

    def __post_init__(self) -> None:
        for name in (
            "min_nonnegative_dataset_fraction",
            "min_nonbaseline_win_rate",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        for name in (
            "max_primary_relative_degradation",
            "max_tail_relative_degradation",
        ):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True)
class TailGuardHoldoutGateResult:
    """Typed decision produced from completed external-test records."""

    passed: bool
    expected_records: int
    completed_records: int
    failed_records: int
    paired_runs: int
    median_gain_vs_a2: float
    mean_gain_vs_a2: float
    nonnegative_dataset_fraction: float
    nonbaseline_selection_count: int
    nonbaseline_win_rate: float
    worst_primary_gain: float
    worst_tail_gain: float
    violations: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _relative_gain(reference: float, candidate: float, metric: str) -> float:
    scale = max(abs(float(reference)), np.finfo(float).eps)
    if str(metric) in {"roc_auc", "accuracy", "f1_macro", "f1_weighted", "r2"}:
        return (float(candidate) - float(reference)) / scale
    return (float(reference) - float(candidate)) / scale


def build_tail_guard_holdout_comparison(replay: pd.DataFrame) -> pd.DataFrame:
    """Pair A9 with A2 for every dataset, seed, budget and model."""

    required = {
        "task_name",
        "seed",
        "budget_ratio",
        "model",
        "arm_name",
        "selected_arm_name",
        "status",
        "test_primary_metric",
        "test_primary_value",
        "test_tail_mean_absolute_error",
    }
    missing = sorted(required - set(replay.columns))
    if missing:
        raise ValueError(f"Routing replay is missing columns: {missing}")

    identity = ["task_name", "seed", "budget_ratio", "model"]
    reference = replay.loc[replay.arm_name == REFERENCE_ARM].copy()
    selected = replay.loc[
        replay.arm_name == DEFAULT_TAIL_GUARDED_SELECTOR_ARM_NAME
    ].copy()
    pairs = reference.merge(
        selected,
        on=identity,
        suffixes=("_a2", "_a9"),
        validate="one_to_one",
    )
    pairs["primary_metric"] = pairs.test_primary_metric_a2
    pairs["primary_gain_vs_a2"] = [
        _relative_gain(reference_value, candidate_value, metric)
        for reference_value, candidate_value, metric in zip(
            pairs.test_primary_value_a2,
            pairs.test_primary_value_a9,
            pairs.primary_metric,
        )
    ]
    pairs["tail_gain_vs_a2"] = [
        _relative_gain(reference_value, candidate_value, "tail_mae")
        for reference_value, candidate_value in zip(
            pairs.test_tail_mean_absolute_error_a2,
            pairs.test_tail_mean_absolute_error_a9,
        )
    ]
    pairs["selected_nonbaseline"] = pairs.selected_arm_name_a9 != REFERENCE_ARM
    return pairs


def evaluate_tail_guard_holdout(
    replay: pd.DataFrame,
    *,
    expected_records: int,
    spec: TailGuardHoldoutGateSpec,
) -> tuple[TailGuardHoldoutGateResult, pd.DataFrame, pd.DataFrame]:
    """Evaluate completeness and all pre-registered acceptance criteria."""

    comparison = build_tail_guard_holdout_comparison(replay)
    completed = int((replay.status == "completed").sum())
    failed = int((replay.status != "completed").sum())
    by_dataset = (
        comparison.groupby("task_name", as_index=False)
        .agg(
            paired_runs=("primary_gain_vs_a2", "size"),
            median_gain_vs_a2=("primary_gain_vs_a2", "median"),
            mean_gain_vs_a2=("primary_gain_vs_a2", "mean"),
            worst_primary_gain=("primary_gain_vs_a2", "min"),
            median_tail_gain_vs_a2=("tail_gain_vs_a2", "median"),
            worst_tail_gain=("tail_gain_vs_a2", "min"),
        )
        .sort_values("task_name")
        .reset_index(drop=True)
    )
    nonbaseline = comparison.loc[comparison.selected_nonbaseline]
    nonnegative_fraction = float(
        (by_dataset.median_gain_vs_a2 >= 0.0).mean()
    )
    nonbaseline_win_rate = (
        1.0
        if nonbaseline.empty
        else float((nonbaseline.primary_gain_vs_a2 > 0.0).mean())
    )
    worst_tail_gain = float(comparison.tail_gain_vs_a2.min())
    violations: list[str] = []
    if len(replay) != int(expected_records) or completed != int(expected_records):
        violations.append("incomplete_record_grid")
    if failed:
        violations.append("failed_records_present")
    if float(comparison.primary_gain_vs_a2.median()) < 0.0:
        violations.append("negative_overall_median_gain")
    if nonnegative_fraction < spec.min_nonnegative_dataset_fraction:
        violations.append("insufficient_nonnegative_dataset_fraction")
    if nonbaseline_win_rate < spec.min_nonbaseline_win_rate:
        violations.append("insufficient_nonbaseline_win_rate")
    if float(comparison.primary_gain_vs_a2.min()) < -spec.max_primary_relative_degradation:
        violations.append("primary_worst_case_exceeds_limit")
    if worst_tail_gain < -spec.max_tail_relative_degradation:
        violations.append("tail_worst_case_exceeds_limit")

    result = TailGuardHoldoutGateResult(
        passed=not violations,
        expected_records=int(expected_records),
        completed_records=completed,
        failed_records=failed,
        paired_runs=int(len(comparison)),
        median_gain_vs_a2=float(comparison.primary_gain_vs_a2.median()),
        mean_gain_vs_a2=float(comparison.primary_gain_vs_a2.mean()),
        nonnegative_dataset_fraction=nonnegative_fraction,
        nonbaseline_selection_count=int(len(nonbaseline)),
        nonbaseline_win_rate=nonbaseline_win_rate,
        worst_primary_gain=float(comparison.primary_gain_vs_a2.min()),
        worst_tail_gain=worst_tail_gain,
        violations=tuple(violations),
    )
    return result, comparison, by_dataset


def build_holdout_budget_summary(comparison: pd.DataFrame) -> pd.DataFrame:
    """Aggregate paired holdout gains without mixing budget levels."""

    return (
        comparison.groupby(["task_name", "budget_ratio"], as_index=False)
        .agg(
            paired_runs=("primary_gain_vs_a2", "size"),
            median_gain_vs_a2=("primary_gain_vs_a2", "median"),
            mean_gain_vs_a2=("primary_gain_vs_a2", "mean"),
            median_tail_gain_vs_a2=("tail_gain_vs_a2", "median"),
            nonbaseline_selection_rate=("selected_nonbaseline", "mean"),
        )
        .sort_values(["task_name", "budget_ratio"])
        .reset_index(drop=True)
    )


def write_tail_guard_holdout_figures(
    output_dir: Path,
    comparison: pd.DataFrame,
) -> None:
    """Persist compact diagnostics for gain curves and selector behavior."""

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    budget_summary = build_holdout_budget_summary(comparison)
    grouped = list(budget_summary.groupby("task_name"))
    n_columns = 2
    n_rows = max(1, int(np.ceil(len(grouped) / n_columns)))
    figure, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(12, 4 * n_rows),
        squeeze=False,
        sharex=True,
    )
    for axis, (task_name, values) in zip(axes.flat, grouped):
        axis.plot(
            values["budget_ratio"],
            values["mean_gain_vs_a2"],
            marker="o",
            linewidth=1.8,
            color="#3274a1",
            label="Средний выигрыш",
        )
        axis.plot(
            values["budget_ratio"],
            values["median_gain_vs_a2"],
            marker="s",
            linewidth=1.4,
            linestyle="--",
            color="#e1812c",
            label="Медианный выигрыш",
        )
        axis.axhline(0.0, color="#555555", linewidth=0.8)
        axis.set_title(task_name)
        axis.set_xlabel("Доля бюджета")
        axis.set_ylabel("Относительный выигрыш A9 к A2")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    for axis in axes.flat[len(grouped):]:
        axis.set_visible(False)
    figure.tight_layout()
    figure.savefig(output_dir / "tail_guard_holdout_gain_by_budget.png", dpi=180)
    plt.close(figure)

    selection_counts = (
        comparison["selected_arm_name_a9"]
        .value_counts()
        .sort_values(ascending=True)
    )
    figure, axis = plt.subplots(figsize=(10, 5.5))
    axis.barh(selection_counts.index, selection_counts.values, color="#3274a1")
    axis.set_xlabel("Число выборов")
    axis.set_ylabel("Выбранная геометрия")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "tail_guard_holdout_selection_counts.png", dpi=180)
    plt.close(figure)


def write_tail_guard_holdout_artifacts(
    output_dir: Path,
    *,
    spec: TailGuardHoldoutGateSpec,
    result: TailGuardHoldoutGateResult,
    comparison: pd.DataFrame,
    by_dataset: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_dir / "tail_guard_holdout_pairs.csv", index=False)
    by_dataset.to_csv(output_dir / "tail_guard_holdout_by_dataset.csv", index=False)
    budget_summary = build_holdout_budget_summary(comparison)
    budget_summary.to_csv(
        output_dir / "tail_guard_holdout_by_budget.csv",
        index=False,
    )
    payload = {"spec": asdict(spec), "result": result.to_dict()}
    (output_dir / "tail_guard_holdout_gate.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    status = "ПРОЙДЕН" if result.passed else "НЕ ПРОЙДЕН"
    rows = [
        "# Независимая проверка хвостового ограничителя A9",
        "",
        f"Статус критерия допуска: **{status}**.",
        "",
        f"Завершено записей: **{result.completed_records}/{result.expected_records}**, "
        f"ошибок: **{result.failed_records}**.",
        f"Медианный выигрыш A9 относительно A2: **{result.median_gain_vs_a2:+.3%}**; "
        f"средний: **{result.mean_gain_vs_a2:+.3%}**.",
        f"Доля датасетов с неотрицательной медианой: "
        f"**{result.nonnegative_dataset_fraction:.1%}**.",
        f"Доля побед при выборе небазовой геометрии: "
        f"**{result.nonbaseline_win_rate:.1%}** "
        f"({result.nonbaseline_selection_count} небазовых решений).",
        f"Худший выигрыш основной метрики: **{result.worst_primary_gain:+.3%}**; "
        f"худший выигрыш TailMAE: **{result.worst_tail_gain:+.3%}**.",
        "",
        "## Результаты по датасетам",
        "",
        markdown_table(by_dataset),
        "",
        "## Результаты по бюджетам",
        "",
        markdown_table(budget_summary),
        "",
        "## Нарушения",
        "",
        *(f"- `{value}`" for value in result.violations),
    ]
    if not result.violations:
        rows.append("- Нарушений нет.")
    (output_dir / "tail_guard_holdout_report.md").write_text(
        "\n".join(rows) + "\n",
        encoding="utf-8",
    )
    write_tail_guard_holdout_figures(output_dir, comparison)


def run_rmt_tail_guard_holdout(
    *,
    regression_tasks: Sequence[str] = DEFAULT_HOLDOUT_TASKS,
    models: Sequence[str] = ("lightgbm",),
    budget_ratios: Sequence[float] = DEFAULT_HOLDOUT_BUDGETS,
    seeds: Sequence[int] = DEFAULT_HOLDOUT_SEEDS,
    max_train_rows: int | None = 100_000,
    output_dir: str | Path | None = None,
    show_progress: bool = True,
) -> Path:
    """Run the frozen A9 holdout protocol and materialize its decision."""

    from rmt_cross_fitted_geometry_selector_experiment import (
        run_rmt_tail_guarded_geometry_selector_experiment,
    )

    resolved_output = Path(output_dir) if output_dir is not None else (
        BENCHMARK_DIR
        / "results"
        / f"run_rmt_tail_guard_holdout_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    run_rmt_tail_guarded_geometry_selector_experiment(
        regression_tasks=tuple(regression_tasks),
        classification_tasks=(),
        models=tuple(models),
        budget_ratios=tuple(budget_ratios),
        seeds=tuple(seeds),
        max_train_rows=max_train_rows,
        selection_folds=5,
        tail_risk_quantile=0.90,
        tail_noninferiority_margin=0.005,
        regression_stratification_bins=10,
        output_dir=resolved_output,
        show_progress=show_progress,
    )
    replay = pd.read_csv(resolved_output / "routing_geometry_replay.csv")
    expected = (
        len(tuple(regression_tasks))
        * len(tuple(models))
        * len(tuple(budget_ratios))
        * len(tuple(seeds))
        * 4
    )
    spec = TailGuardHoldoutGateSpec()
    result, comparison, by_dataset = evaluate_tail_guard_holdout(
        replay,
        expected_records=expected,
        spec=spec,
    )
    write_tail_guard_holdout_artifacts(
        resolved_output,
        spec=spec,
        result=result,
        comparison=comparison,
        by_dataset=by_dataset,
    )
    return resolved_output


def _run_cli() -> Path:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_HOLDOUT_TASKS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_HOLDOUT_SEEDS))
    parser.add_argument("--budgets", type=float, nargs="+", default=list(DEFAULT_HOLDOUT_BUDGETS))
    parser.add_argument("--max-train-rows", type=int, default=100_000)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    return run_rmt_tail_guard_holdout(
        regression_tasks=tuple(args.tasks),
        seeds=tuple(args.seeds),
        budget_ratios=tuple(args.budgets),
        max_train_rows=args.max_train_rows,
        output_dir=args.output_dir,
        show_progress=not args.no_progress,
    )


if __name__ == "__main__":
    _run_cli()
