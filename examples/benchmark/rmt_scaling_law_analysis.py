"""Fit and visualize budget-quality scaling laws from RMT experiment artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = Path(__file__).resolve().parent
for module_path in (ROOT_DIR, BENCHMARK_DIR):
    if str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))

from rmt_experiment_utils import json_ready, markdown_table  # noqa: E402
from sampling_zoo.core.experiment.scaling_laws import (  # noqa: E402
    BudgetScalingLawFitter,
    ScalingLawFitSpec,
    ScalingLawObservation,
)


DEFAULT_SCALING_ARMS: tuple[str, ...] = (
    "B0_standard_A9",
    "B3_validation_selected",
)


class RMTScalingLawArtifactBuilder:
    def __init__(
        self,
        *,
        arms: Sequence[str] = DEFAULT_SCALING_ARMS,
        bootstrap_iterations: int = 500,
        random_state: int = 42,
    ) -> None:
        self.arms = tuple(str(value) for value in arms)
        self.fitter = BudgetScalingLawFitter(
            ScalingLawFitSpec(
                bootstrap_iterations=int(bootstrap_iterations),
                random_state=int(random_state),
            )
        )

    def build(self, runs_path: Path, output_dir: Path) -> pd.DataFrame:
        runs = self._load_runs(runs_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        points, fits = self._fit_groups(runs)
        points.to_csv(output_dir / "scaling_law_points.csv", index=False)
        fits.to_csv(output_dir / "scaling_law_fits.csv", index=False)
        (output_dir / "scaling_law_fits.json").write_text(
            json.dumps(json_ready(fits.to_dict(orient="records")), indent=2),
            encoding="utf-8",
        )
        self._write_report(fits, output_dir)
        self._write_figure(points, fits, output_dir)
        return fits

    def _load_runs(self, path: Path) -> pd.DataFrame:
        runs = pd.read_csv(path)
        required = {
            "dataset",
            "seed",
            "budget_ratio",
            "model",
            "arm_name",
            "status",
            "test_primary_metric",
            "test_primary_value",
            "full_reference_primary_value",
        }
        missing = sorted(required - set(runs.columns))
        if missing:
            raise ValueError(f"Scaling-law input is missing columns: {missing}")
        selected = runs[
            (runs["status"] == "completed")
            & runs["arm_name"].isin(self.arms)
        ].copy()
        if selected.empty:
            raise ValueError("No completed scaling-law rows were found")
        return selected

    def _fit_groups(
        self,
        runs: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        point_rows = []
        fit_rows = []
        group_columns = ["dataset", "model", "arm_name"]
        for identity, group in runs.groupby(group_columns, sort=True):
            observations = self._observations(group)
            fit = self.fitter.fit(observations)
            dataset, model, arm_name = identity
            fit_rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "arm_name": arm_name,
                    **fit.to_dict(),
                }
            )
            for budget in sorted({item.budget_ratio for item in observations}):
                budget_observations = [
                    item for item in observations if item.budget_ratio == budget
                ]
                point_rows.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "arm_name": arm_name,
                        "budget_ratio": budget,
                        "median_loss": float(
                            np.median([item.loss for item in budget_observations])
                        ),
                        "predicted_loss": fit.predict_loss(budget),
                        "n_observations": len(budget_observations),
                        "is_full_reference": bool(budget == 1.0),
                    }
                )
        return pd.DataFrame(point_rows), pd.DataFrame(fit_rows)

    @staticmethod
    def _observations(group: pd.DataFrame) -> tuple[ScalingLawObservation, ...]:
        metrics = set(group["test_primary_metric"].dropna().astype(str))
        if len(metrics) != 1:
            raise ValueError("One scaling-law group must use exactly one metric")
        metric = next(iter(metrics))
        observations = [
            ScalingLawObservation(
                budget_ratio=float(row.budget_ratio),
                primary_value=float(row.test_primary_value),
                primary_metric=metric,
                replicate_id=str(row.seed),
            )
            for row in group.itertuples(index=False)
        ]
        reference_by_seed = group.drop_duplicates(subset=["seed"])
        observations.extend(
            ScalingLawObservation(
                budget_ratio=1.0,
                primary_value=float(row.full_reference_primary_value),
                primary_metric=metric,
                replicate_id=str(row.seed),
                is_full_reference=True,
            )
            for row in reference_by_seed.itertuples(index=False)
        )
        return tuple(observations)

    @staticmethod
    def _write_report(fits: pd.DataFrame, output_dir: Path) -> None:
        display = fits.copy()
        if "required_budget_by_degradation" in display:
            display["required_budget_by_degradation"] = display[
                "required_budget_by_degradation"
            ].map(json.dumps)
        lines = [
            "# Масштабирование качества RMT по бюджету",
            "",
            "Аппроксимация использует функцию "
            "`loss(b) = L_full + A * (b^(-alpha) - 1)` с точным якорем "
            "опорной модели при `b=1`. Для ROC AUC "
            "используется потеря `1 - ROC AUC`.",
            "",
            "Параметр `alpha` описывает скорость ухудшения при уменьшении бюджета. "
            "Высокий `R^2` на четырёх бюджетах и одной опорной точке является "
            "предварительным свидетельством, а не универсальным законом.",
            "",
            markdown_table(display),
            "",
        ]
        (output_dir / "scaling_law_report.md").write_text(
            "\n".join(lines),
            encoding="utf-8",
        )

    @staticmethod
    def _write_figure(
        points: pd.DataFrame,
        fits: pd.DataFrame,
        output_dir: Path,
    ) -> None:
        datasets = tuple(points["dataset"].drop_duplicates())
        n_columns = min(2, len(datasets))
        n_rows = int(np.ceil(len(datasets) / max(n_columns, 1)))
        figure, axes = plt.subplots(
            n_rows,
            n_columns,
            figsize=(7 * n_columns, 4.5 * n_rows),
            squeeze=False,
        )
        for axis, dataset in zip(axes.flat, datasets):
            selected = points[points["dataset"] == dataset]
            for arm_name, group in selected.groupby("arm_name"):
                axis.scatter(
                    group["budget_ratio"],
                    group["median_loss"],
                    label=f"{arm_name}: observed",
                )
                fit_row = fits[
                    (fits["dataset"] == dataset)
                    & (fits["arm_name"] == arm_name)
                ].iloc[0]
                budget_grid = np.geomspace(
                    max(0.005, float(group["budget_ratio"].min())),
                    1.0,
                    160,
                )
                predicted = float(fit_row["asymptotic_loss"]) + float(
                    fit_row["scale"]
                ) * budget_grid ** (-float(fit_row["exponent"]))
                axis.plot(budget_grid, predicted, label=f"{arm_name}: fit")
            axis.set_xscale("log")
            axis.set_title(str(dataset))
            axis.set_xlabel("Доля бюджета")
            axis.set_ylabel("Функция потерь")
            axis.grid(alpha=0.25)
            axis.legend(fontsize=8)
        for axis in axes.flat[len(datasets):]:
            axis.set_visible(False)
        figure.tight_layout()
        figure.savefig(output_dir / "scaling_law_curves.png", dpi=180)
        plt.close(figure)


def run_scaling_law_analysis(
    runs_path: Path,
    output_dir: Path,
    *,
    arms: Sequence[str] = DEFAULT_SCALING_ARMS,
    bootstrap_iterations: int = 500,
) -> pd.DataFrame:
    return RMTScalingLawArtifactBuilder(
        arms=arms,
        bootstrap_iterations=bootstrap_iterations,
    ).build(runs_path, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs_path", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--arms", nargs="+", default=list(DEFAULT_SCALING_ARMS))
    parser.add_argument("--bootstrap-iterations", type=int, default=500)
    args = parser.parse_args()
    result = run_scaling_law_analysis(
        args.runs_path,
        args.output_dir,
        arms=tuple(args.arms),
        bootstrap_iterations=args.bootstrap_iterations,
    )
    print(result)


if __name__ == "__main__":
    main()
