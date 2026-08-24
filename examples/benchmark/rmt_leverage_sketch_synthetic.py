"""Synthetic Phase S for exact-budget leverage-preserving row sketches."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss, mean_absolute_error, mean_squared_error, roc_auc_score
from tqdm.auto import tqdm


ROOT_DIR = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = Path(__file__).resolve().parent
for module_path in (ROOT_DIR, BENCHMARK_DIR):
    if str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))

from sampling_zoo.core.sampling_strategies.spectral.backend.matrix_backend import (  # noqa: E402
    MatrixRMTBackend,
    compute_leverage_scores,
)
from sampling_zoo.core.sampling_strategies.spectral.leverage_sketch import (  # noqa: E402
    build_exact_budget_sketch_plan,
    evaluate_subspace_preservation,
)
from rmt_experiment_utils import markdown_table  # noqa: E402


DEFAULT_SCENARIOS: tuple[str, ...] = (
    "low_rank_regression",
    "high_leverage_regression",
    "heavy_tail_regression",
    "rare_class_classification",
)
DEFAULT_BUDGET_RATIOS: tuple[float, ...] = (0.05, 0.10, 0.20)
DEFAULT_SEEDS: tuple[int, ...] = (11, 29, 47)


@dataclass(frozen=True)
class LeverageSketchArm:
    arm_id: str
    policy: str
    training_reweighting: str = "none"
    leverage_cap_quantile: float = 0.95
    leverage_mixture_alpha: float = 0.25


DEFAULT_ARMS: tuple[LeverageSketchArm, ...] = (
    LeverageSketchArm("S0_uniform", "uniform"),
    LeverageSketchArm("S1_top_leverage", "leverage"),
    LeverageSketchArm("S2_capped_leverage", "capped_leverage"),
    LeverageSketchArm("S3_saturated_leverage", "saturated_leverage"),
    LeverageSketchArm("S4_robust_mixture", "robust_leverage_mixture"),
    LeverageSketchArm(
        "S5_robust_mixture_ipw",
        "robust_leverage_mixture",
        training_reweighting="inverse_probability",
    ),
    LeverageSketchArm(
        "S6_ridge_leverage_ipw",
        "saturated_ridge_leverage",
        training_reweighting="inverse_probability",
    ),
)


@dataclass(frozen=True)
class LeverageSketchSyntheticConfig:
    scenarios: Sequence[str] = DEFAULT_SCENARIOS
    budget_ratios: Sequence[float] = DEFAULT_BUDGET_RATIOS
    seeds: Sequence[int] = DEFAULT_SEEDS
    arms: Sequence[LeverageSketchArm] = DEFAULT_ARMS
    n_train: int = 600
    n_test: int = 400
    n_features: int = 32
    latent_rank: int = 4
    tail_quantile: float = 0.90
    snapshot_every: int = 10
    show_progress: bool = True
    output_root: str | Path = Path(__file__).resolve().parent / "results"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "scenarios",
            tuple(str(value).strip().lower() for value in self.scenarios),
        )
        object.__setattr__(
            self,
            "budget_ratios",
            tuple(float(value) for value in self.budget_ratios),
        )
        object.__setattr__(self, "seeds", tuple(int(value) for value in self.seeds))
        object.__setattr__(self, "arms", tuple(self.arms))
        object.__setattr__(self, "output_root", Path(self.output_root))
        if not self.scenarios or any(value not in DEFAULT_SCENARIOS for value in self.scenarios):
            raise ValueError(f"scenarios must be selected from {DEFAULT_SCENARIOS}")
        if not self.budget_ratios or any(
            not 0.0 < value <= 1.0 for value in self.budget_ratios
        ):
            raise ValueError("budget_ratios must contain values in (0, 1]")
        if not self.seeds or not self.arms:
            raise ValueError("seeds and arms must not be empty")
        if len({arm.arm_id for arm in self.arms}) != len(self.arms):
            raise ValueError("arm_id values must be unique")
        if int(self.n_train) < 40 or int(self.n_test) < 40:
            raise ValueError("n_train and n_test must be at least 40")
        if int(self.n_features) < int(self.latent_rank) or int(self.latent_rank) < 1:
            raise ValueError("latent_rank must be positive and no larger than n_features")
        if not 0.5 < float(self.tail_quantile) < 1.0:
            raise ValueError("tail_quantile must be in (0.5, 1)")
        if int(self.snapshot_every) < 1:
            raise ValueError("snapshot_every must be positive")

    @property
    def expected_records(self) -> int:
        return (
            len(self.scenarios)
            * len(self.budget_ratios)
            * len(self.seeds)
            * len(self.arms)
        )


@dataclass(frozen=True)
class SyntheticSketchDataset:
    scenario: str
    problem: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


def build_phase_s_grid(
    config: LeverageSketchSyntheticConfig,
) -> tuple[tuple[str, float, int, LeverageSketchArm], ...]:
    return tuple(
        (scenario, budget, seed, arm)
        for scenario in config.scenarios
        for budget in config.budget_ratios
        for seed in config.seeds
        for arm in config.arms
    )


def generate_synthetic_sketch_dataset(
    scenario: str,
    *,
    n_train: int,
    n_test: int,
    n_features: int,
    latent_rank: int,
    random_state: int,
) -> SyntheticSketchDataset:
    rng = np.random.default_rng(random_state)
    loadings = rng.normal(size=(latent_rank, n_features)) / np.sqrt(latent_rank)

    def latent_rows(size: int) -> np.ndarray:
        return rng.normal(size=(size, latent_rank))

    train_latent = latent_rows(n_train)
    test_latent = latent_rows(n_test)
    noise_scale = 0.20
    if scenario == "high_leverage_regression":
        train_tail = rng.random(n_train) < 0.05
        test_tail = rng.random(n_test) < 0.05
        train_latent[train_tail] *= 6.0
        test_latent[test_tail] *= 6.0
    else:
        train_tail = np.zeros(n_train, dtype=bool)
        test_tail = np.zeros(n_test, dtype=bool)

    if scenario == "heavy_tail_regression":
        train_noise = rng.standard_t(df=3.0, size=(n_train, n_features)) * noise_scale
        test_noise = rng.standard_t(df=3.0, size=(n_test, n_features)) * noise_scale
    else:
        train_noise = rng.normal(scale=noise_scale, size=(n_train, n_features))
        test_noise = rng.normal(scale=noise_scale, size=(n_test, n_features))
    X_train = train_latent @ loadings + train_noise
    X_test = test_latent @ loadings + test_noise

    if scenario == "rare_class_classification":
        train_score = train_latent[:, 0] + 0.5 * train_latent[:, 1]
        test_score = test_latent[:, 0] + 0.5 * test_latent[:, 1]
        threshold = float(np.quantile(train_score, 0.90))
        y_train = (train_score + rng.normal(scale=0.25, size=n_train) > threshold).astype(int)
        y_test = (test_score + rng.normal(scale=0.25, size=n_test) > threshold).astype(int)
        problem = "classification"
    else:
        y_train = (
            2.0 * train_latent[:, 0]
            - 1.2 * train_latent[:, 1]
            + 0.4 * np.square(train_latent[:, 2])
            + 3.0 * train_tail.astype(float)
            + rng.normal(scale=0.25, size=n_train)
        )
        y_test = (
            2.0 * test_latent[:, 0]
            - 1.2 * test_latent[:, 1]
            + 0.4 * np.square(test_latent[:, 2])
            + 3.0 * test_tail.astype(float)
            + rng.normal(scale=0.25, size=n_test)
        )
        problem = "regression"

    mean = np.mean(X_train, axis=0)
    scale = np.std(X_train, axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return SyntheticSketchDataset(
        scenario=scenario,
        problem=problem,
        X_train=(X_train - mean) / scale,
        y_train=np.asarray(y_train),
        X_test=(X_test - mean) / scale,
        y_test=np.asarray(y_test),
    )


class LeverageSketchSyntheticOrchestrator:
    """Effectful shell for the deterministic Phase S grid."""

    def __init__(self, config: LeverageSketchSyntheticConfig) -> None:
        self.config = config

    def run(self) -> Path:
        output_dir = self._create_output_dir()
        grid = build_phase_s_grid(self.config)
        self._write_meta(output_dir, status="running", record_count=0)
        records: list[dict[str, Any]] = []
        jsonl_path = output_dir / "phase_s_runs.jsonl"
        try:
            with jsonl_path.open("a", encoding="utf-8") as stream:
                for index, point in enumerate(
                    tqdm(
                        grid,
                        total=len(grid),
                        desc="Synthetic leverage Phase S",
                        disable=not self.config.show_progress,
                        unit="run",
                    ),
                    start=1,
                ):
                    record = self._run_grid_point(*point)
                    records.append(record)
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    stream.flush()
                    if index % self.config.snapshot_every == 0:
                        self._write_meta(
                            output_dir,
                            status="running",
                            record_count=index,
                        )
            self._build_artifacts(output_dir, records)
            self._write_meta(
                output_dir,
                status="completed",
                record_count=len(records),
            )
        except Exception as exc:
            self._write_meta(
                output_dir,
                status="failed",
                record_count=len(records),
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        return output_dir

    def _create_output_dir(self) -> Path:
        run_id = "run_rmt_leverage_sketch_synthetic_" + datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
        output_dir = Path(self.config.output_root) / run_id
        output_dir.mkdir(parents=True, exist_ok=False)
        return output_dir

    def _run_grid_point(
        self,
        scenario: str,
        budget_ratio: float,
        seed: int,
        arm: LeverageSketchArm,
    ) -> dict[str, Any]:
        started = perf_counter()
        dataset = generate_synthetic_sketch_dataset(
            scenario,
            n_train=self.config.n_train,
            n_test=self.config.n_test,
            n_features=self.config.n_features,
            latent_rank=self.config.latent_rank,
            random_state=seed,
        )
        target_size = max(2, int(round(dataset.X_train.shape[0] * budget_ratio)))
        U, _S, _Vt = np.linalg.svd(dataset.X_train, full_matrices=False)
        rank = min(self.config.latent_rank, U.shape[1])
        leverage_scores = compute_leverage_scores(U[:, :rank])
        ridge_result = MatrixRMTBackend.compute_ridge_leverage_scores(
            dataset.X_train,
            "auto",
        )
        plan = build_exact_budget_sketch_plan(
            np.arange(dataset.X_train.shape[0]),
            target_size=target_size,
            policy=arm.policy,
            scores=leverage_scores,
            ridge_scores=ridge_result.scores,
            ridge_lambda=ridge_result.ridge_lambda,
            random_state=seed,
            leverage_cap_quantile=arm.leverage_cap_quantile,
            leverage_mixture_alpha=arm.leverage_mixture_alpha,
            reweighting=arm.training_reweighting,
            metadata={"arm_id": arm.arm_id},
        )
        subspace = evaluate_subspace_preservation(
            dataset.X_train,
            plan,
            rank=rank,
        )
        metrics = self._fit_and_evaluate(dataset, plan.selected_indices, plan.training_weights)
        return {
            "status": "completed",
            "scenario": scenario,
            "problem": dataset.problem,
            "budget_ratio": float(budget_ratio),
            "seed": int(seed),
            "arm_id": arm.arm_id,
            "policy": arm.policy,
            "training_reweighting": arm.training_reweighting,
            "target_size": int(target_size),
            "selected_size": int(plan.selected_indices.size),
            "unique_selected_size": int(np.unique(plan.selected_indices).size),
            "exact_budget": bool(plan.exact_budget),
            "inclusion_saturated_count": plan.inclusion.saturated_count,
            "inclusion_probability_min": float(np.min(plan.inclusion.probabilities)),
            "inclusion_probability_max": float(np.max(plan.inclusion.probabilities)),
            "training_weight_min": float(np.min(plan.training_weights)),
            "training_weight_max": float(np.max(plan.training_weights)),
            "effective_sample_size": plan.to_dict()["effective_sample_size"],
            **subspace.to_dict(),
            **metrics,
            "runtime_seconds": float(perf_counter() - started),
        }

    def _fit_and_evaluate(
        self,
        dataset: SyntheticSketchDataset,
        selected: np.ndarray,
        weights: np.ndarray,
    ) -> dict[str, Any]:
        X_selected = dataset.X_train[selected]
        y_selected = dataset.y_train[selected]
        if dataset.problem == "classification":
            if np.unique(y_selected).size < 2:
                return {
                    "primary_metric": "roc_auc",
                    "primary_score": 0.5,
                    "roc_auc": 0.5,
                    "log_loss": float(np.log(2.0)),
                    "single_class_sample": True,
                }
            model = LogisticRegression(max_iter=500, random_state=0)
            model.fit(X_selected, y_selected, sample_weight=weights)
            probabilities = model.predict_proba(dataset.X_test)[:, 1]
            return {
                "primary_metric": "roc_auc",
                "primary_score": float(roc_auc_score(dataset.y_test, probabilities)),
                "roc_auc": float(roc_auc_score(dataset.y_test, probabilities)),
                "log_loss": float(log_loss(dataset.y_test, probabilities, labels=[0, 1])),
                "single_class_sample": False,
            }

        model = Ridge(alpha=1.0)
        model.fit(X_selected, y_selected, sample_weight=weights)
        predictions = model.predict(dataset.X_test)
        absolute_errors = np.abs(dataset.y_test - predictions)
        tail_threshold = float(
            np.quantile(np.abs(dataset.y_test), self.config.tail_quantile)
        )
        tail_mask = np.abs(dataset.y_test) >= tail_threshold
        rmse = float(np.sqrt(mean_squared_error(dataset.y_test, predictions)))
        return {
            "primary_metric": "rmse",
            "primary_score": rmse,
            "rmse": rmse,
            "mae": float(mean_absolute_error(dataset.y_test, predictions)),
            "tail_mae": float(np.mean(absolute_errors[tail_mask])),
            "single_class_sample": False,
        }

    def _build_artifacts(
        self,
        output_dir: Path,
        records: Iterable[dict[str, Any]],
    ) -> None:
        raw = pd.DataFrame(records)
        raw.to_csv(output_dir / "phase_s_raw_runs.csv", index=False)
        paired = self._pair_with_uniform(raw)
        paired.to_csv(output_dir / "phase_s_paired_runs.csv", index=False)
        summary = self._build_summary(paired)
        summary.to_csv(output_dir / "phase_s_summary.csv", index=False)
        gate = self._evaluate_gate(raw, summary)
        (output_dir / "phase_s_gate.json").write_text(
            json.dumps(gate, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        self._write_report(output_dir, raw, summary, gate)
        self._write_figures(output_dir, summary)

    @staticmethod
    def _pair_with_uniform(raw: pd.DataFrame) -> pd.DataFrame:
        keys = ["scenario", "problem", "budget_ratio", "seed"]
        baseline = raw[raw["arm_id"] == "S0_uniform"][
            keys + ["primary_score", "gram_relative_error"]
        ].rename(
            columns={
                "primary_score": "uniform_primary_score",
                "gram_relative_error": "uniform_gram_relative_error",
            }
        )
        paired = raw.merge(baseline, on=keys, how="left", validate="many_to_one")
        regression = paired["problem"] == "regression"
        paired["primary_gain_vs_uniform"] = np.where(
            regression,
            (
                paired["uniform_primary_score"] - paired["primary_score"]
            ) / paired["uniform_primary_score"].clip(lower=1e-12),
            paired["primary_score"] - paired["uniform_primary_score"],
        )
        paired["gram_error_delta_vs_uniform"] = (
            paired["gram_relative_error"] - paired["uniform_gram_relative_error"]
        )
        return paired

    @staticmethod
    def _build_summary(paired: pd.DataFrame) -> pd.DataFrame:
        return (
            paired.groupby(
                ["scenario", "problem", "budget_ratio", "arm_id", "policy", "training_reweighting"],
                dropna=False,
            )
            .agg(
                runs=("seed", "size"),
                primary_score_mean=("primary_score", "mean"),
                primary_gain_mean=("primary_gain_vs_uniform", "mean"),
                primary_gain_median=("primary_gain_vs_uniform", "median"),
                gram_relative_error_mean=("gram_relative_error", "mean"),
                gram_error_delta_mean=("gram_error_delta_vs_uniform", "mean"),
                projection_cost_error_mean=("projection_cost_relative_error", "mean"),
                max_principal_angle_mean=("max_principal_angle_degrees", "mean"),
                effective_sample_size_mean=("effective_sample_size", "mean"),
                runtime_seconds_mean=("runtime_seconds", "mean"),
            )
            .reset_index()
        )

    @staticmethod
    def _evaluate_gate(raw: pd.DataFrame, summary: pd.DataFrame) -> dict[str, Any]:
        nonbaseline = summary[summary["arm_id"] != "S0_uniform"]
        by_scenario = (
            nonbaseline.groupby(["scenario", "arm_id"])
            .agg(scenario_gain=("primary_gain_mean", "mean"))
            .reset_index()
        )
        worst_scenario = (
            by_scenario.groupby("arm_id")
            .agg(worst_scenario_gain=("scenario_gain", "min"))
        )
        policy_ranking = (
            nonbaseline.groupby("arm_id")
            .agg(
                primary_gain_mean=("primary_gain_mean", "mean"),
                primary_gain_median=("primary_gain_median", "median"),
                gram_error_delta_mean=("gram_error_delta_mean", "mean"),
            )
            .join(worst_scenario)
            .sort_values(
                ["primary_gain_mean", "gram_error_delta_mean"],
                ascending=[False, True],
            )
        )
        eligible = policy_ranking[
            (policy_ranking["primary_gain_mean"] >= -0.005)
            & (policy_ranking["worst_scenario_gain"] >= -0.05)
            & (policy_ranking["gram_error_delta_mean"] <= 0.0)
        ]
        shortlisted = eligible.head(2).reset_index().to_dict("records")
        diagnostic_arms = (
            policy_ranking.loc[
                ~policy_ranking.index.isin(eligible.index)
            ]
            .head(2)
            .reset_index()
            .to_dict("records")
        )
        exact_budget = bool(raw["exact_budget"].all())
        unique_budget = bool((raw["selected_size"] == raw["unique_selected_size"]).all())
        best_gain = (
            float(policy_ranking["primary_gain_mean"].iloc[0])
            if not policy_ranking.empty
            else float("nan")
        )
        return {
            "status": (
                "passed"
                if exact_budget and unique_budget and not eligible.empty
                else "failed"
            ),
            "exact_budget_all_runs": exact_budget,
            "unique_rows_all_runs": unique_budget,
            "minimum_acceptable_mean_gain": -0.005,
            "minimum_acceptable_worst_scenario_gain": -0.05,
            "require_nonpositive_gram_error_delta": True,
            "best_mean_primary_gain": best_gain,
            "shortlisted_arms": shortlisted,
            "diagnostic_arms": diagnostic_arms,
        }

    def _write_report(
        self,
        output_dir: Path,
        raw: pd.DataFrame,
        summary: pd.DataFrame,
        gate: dict[str, Any],
    ) -> None:
        ranking = (
            summary[summary["arm_id"] != "S0_uniform"]
            .groupby("arm_id")
            .agg(
                mean_gain=("primary_gain_mean", "mean"),
                median_gain=("primary_gain_median", "median"),
                mean_gram_error=("gram_relative_error_mean", "mean"),
                mean_projection_error=("projection_cost_error_mean", "mean"),
            )
            .sort_values("mean_gain", ascending=False)
            .reset_index()
        )
        by_scenario = (
            summary[summary["arm_id"] != "S0_uniform"]
            .groupby(["scenario", "arm_id"])
            .agg(
                mean_gain=("primary_gain_mean", "mean"),
                mean_gram_error=("gram_relative_error_mean", "mean"),
            )
            .reset_index()
        )
        lines = [
            "# Синтетический Phase S: leverage-preserving семплирование",
            "",
            f"- Статус: `{gate['status']}`",
            f"- Завершено запусков: `{len(raw)}/{self.config.expected_records}`",
            f"- Точный бюджет во всех запусках: `{gate['exact_budget_all_runs']}`",
            f"- Повторяющиеся строки отсутствуют: `{gate['unique_rows_all_runs']}`",
            "",
            "## Общий рейтинг политик",
            "",
            markdown_table(ranking),
            "",
            "## Кандидаты для Phase S на реальных данных",
            "",
            markdown_table(pd.DataFrame(gate["shortlisted_arms"])),
            "",
            "## Эффекты по сценариям",
            "",
            markdown_table(by_scenario),
            "",
            "## Диагностические контроли для реальных данных",
            "",
            markdown_table(pd.DataFrame(gate["diagnostic_arms"])),
            "",
            "## Интерпретация",
            "",
            "Положительный gain означает меньший RMSE для регрессии или больший ROC AUC для классификации относительно равномерного семплирования при том же начальном значении генератора и бюджете.",
            "Ошибки Gram-матрицы и стоимости проекции оценивают сохранение полной геометрии обучающей выборки до обучения итоговой модели.",
            "Результат `S5_robust_mixture_ipw` следует интерпретировать как эффект согласованной пары: смешанная leverage-вероятность включения и обратное взвешивание при обучении. Политика `S4_robust_mixture` использует те же выбранные строки, но без коррекции весов.",
        ]
        (output_dir / "phase_s_report.md").write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _write_figures(output_dir: Path, summary: pd.DataFrame) -> None:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return
        plot_data = (
            summary.groupby(["budget_ratio", "arm_id"])
            .agg(
                primary_gain=("primary_gain_mean", "mean"),
                gram_error=("gram_relative_error_mean", "mean"),
            )
            .reset_index()
        )
        for metric, ylabel, filename in (
            (
                "primary_gain",
                "Средний gain относительно uniform",
                "phase_s_primary_gain.png",
            ),
            (
                "gram_error",
                "Средняя относительная ошибка Gram-матрицы",
                "phase_s_gram_error.png",
            ),
        ):
            figure, axis = plt.subplots(figsize=(10, 5.5))
            for arm_id, arm_data in plot_data.groupby("arm_id"):
                axis.plot(
                    arm_data["budget_ratio"],
                    arm_data[metric],
                    marker="o",
                    linewidth=1.8,
                    label=arm_id,
                )
            axis.axhline(0.0, color="#555555", linewidth=0.8)
            axis.set_xlabel("Доля бюджета")
            axis.set_ylabel(ylabel)
            axis.grid(alpha=0.25)
            axis.legend(fontsize=8, ncol=2)
            figure.tight_layout()
            figure.savefig(output_dir / filename, dpi=180)
            plt.close(figure)

    def _write_meta(
        self,
        output_dir: Path,
        *,
        status: str,
        record_count: int,
        error: str | None = None,
    ) -> None:
        payload = {
            "experiment": "rmt_leverage_sketch_synthetic_phase_s",
            "status": status,
            "record_count": int(record_count),
            "expected_records": int(self.config.expected_records),
            "updated_at": datetime.now().isoformat(),
            "config": {
                **asdict(self.config),
                "output_root": str(self.config.output_root),
                "arms": [asdict(arm) for arm in self.config.arms],
            },
            "error": error,
        }
        (output_dir / "run_meta.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def run_rmt_leverage_sketch_synthetic(
    config: LeverageSketchSyntheticConfig | None = None,
) -> Path:
    return LeverageSketchSyntheticOrchestrator(
        config or LeverageSketchSyntheticConfig()
    ).run()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = LeverageSketchSyntheticConfig(
        scenarios=("low_rank_regression", "rare_class_classification") if args.smoke else DEFAULT_SCENARIOS,
        budget_ratios=(0.10,) if args.smoke else DEFAULT_BUDGET_RATIOS,
        seeds=(11,) if args.smoke else DEFAULT_SEEDS,
        n_train=240 if args.smoke else 600,
        n_test=160 if args.smoke else 400,
        show_progress=not args.no_progress,
    )
    output_dir = run_rmt_leverage_sketch_synthetic(config)
    print(f"Synthetic Phase S completed: {output_dir}")


if __name__ == "__main__":
    main()
