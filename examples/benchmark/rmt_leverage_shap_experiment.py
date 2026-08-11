"""Factorial validation of row leverage sampling and Leverage SHAP coalitions."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
from time import perf_counter
import sys
from typing import Any, Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm


ROOT_DIR = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = Path(__file__).resolve().parent
for module_path in (ROOT_DIR, BENCHMARK_DIR):
    if str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))

from rmt_experiment_utils import markdown_table
from sampling_zoo.core.sampling_strategies.spectral.leverage_sketch import (
    build_exact_budget_sketch_plan,
    evaluate_subspace_preservation,
)
from sampling_zoo.core.validation.leverage_shap import (
    CoalitionSamplingPolicy,
    estimate_projected_shapley,
    exact_shapley_values,
    full_kernel_regression_loss,
    make_masked_model_game,
)


DEFAULT_SHAP_BUDGET_MULTIPLIERS: tuple[float, ...] = (2.0, 4.0, 8.0, 16.0)
DEFAULT_SHAP_SEEDS: tuple[int, ...] = (
    11,
    29,
    47,
    71,
    101,
    127,
    151,
    181,
    211,
    241,
    271,
    307,
    337,
    367,
    397,
    431,
    461,
    491,
    521,
    557,
)


@dataclass(frozen=True)
class LeverageShapExperimentConfig:
    n_players: int = 10
    budget_multipliers: Sequence[float] = DEFAULT_SHAP_BUDGET_MULTIPLIERS
    seeds: Sequence[int] = DEFAULT_SHAP_SEEDS
    row_budget_ratio: float = 0.25
    output_dir: Path | None = None
    show_progress: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "budget_multipliers",
            tuple(map(float, self.budget_multipliers)),
        )
        object.__setattr__(self, "seeds", tuple(map(int, self.seeds)))
        if self.output_dir is not None:
            object.__setattr__(self, "output_dir", Path(self.output_dir))
        if not 4 <= int(self.n_players) <= 16:
            raise ValueError("n_players must be in [4, 16]")
        if not self.budget_multipliers or any(
            value <= 0.0 for value in self.budget_multipliers
        ):
            raise ValueError("budget_multipliers must be positive")
        if not self.seeds:
            raise ValueError("seeds must not be empty")
        if not 0.0 < float(self.row_budget_ratio) <= 1.0:
            raise ValueError("row_budget_ratio must be in (0, 1]")


@dataclass(frozen=True)
class ExplanationCase:
    scenario: str
    row_arm: str
    game: Callable[[np.ndarray], np.ndarray]
    exact_values: np.ndarray
    full_reference_values: np.ndarray
    downstream_rmse: float | None
    row_gram_error: float | None
    row_budget_exact: bool


def _synthetic_games(n_players: int) -> tuple[tuple[str, Callable[[np.ndarray], np.ndarray]], ...]:
    weights = np.linspace(-1.5, 2.0, n_players)

    def interaction(coalitions: np.ndarray) -> np.ndarray:
        z = coalitions.astype(float)
        return (
            z @ weights
            + 3.0 * z[:, 0] * z[:, 1]
            - 2.5 * z[:, 2] * z[:, 5]
            + 1.75 * z[:, 3] * z[:, 4] * z[:, 7]
        )

    def threshold(coalitions: np.ndarray) -> np.ndarray:
        z = coalitions.astype(float)
        score = z @ weights / np.sqrt(n_players)
        return np.tanh(1.8 * score) + 0.8 * (np.sum(z, axis=1) >= n_players / 2)

    return (("synthetic_interaction", interaction), ("synthetic_threshold", threshold))


def _row_leverage_scores(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    standardized = StandardScaler().fit_transform(X)
    U, _singular, _Vt = np.linalg.svd(standardized, full_matrices=False)
    scores = np.sum(np.square(U), axis=1)
    return standardized, scores


def _fit_diabetes_cases(
    config: LeverageShapExperimentConfig,
    seed: int,
) -> tuple[ExplanationCase, ...]:
    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=seed,
    )
    baseline = np.mean(X_train, axis=0)
    distances = np.linalg.norm(X_test - baseline, axis=1)
    explicand = X_test[int(np.argmax(distances))]
    standardized, leverage_scores = _row_leverage_scores(X_train)
    candidates = np.arange(X_train.shape[0], dtype=int)
    row_budget = max(config.n_players * 2, int(round(len(X_train) * config.row_budget_ratio)))
    row_budget = min(row_budget, len(X_train))
    arm_specs = (
        ("D0_full_rows", None, "none"),
        ("D1_uniform_rows", "uniform", "none"),
        (
            "D2_robust_leverage_ipw",
            "robust_leverage_mixture",
            "inverse_probability",
        ),
    )
    cases: list[ExplanationCase] = []
    full_reference: np.ndarray | None = None
    prepared: list[tuple[str, Any, Any, bool, float | None]] = []
    for arm_name, policy, reweighting in arm_specs:
        if policy is None:
            selected = candidates
            sample_weight = None
            exact_budget = True
            gram_error = 0.0
        else:
            plan = build_exact_budget_sketch_plan(
                candidates,
                target_size=row_budget,
                policy=policy,
                scores=leverage_scores,
                random_state=seed,
                leverage_mixture_alpha=0.25,
                reweighting=reweighting,
            )
            selected = plan.selected_indices
            sample_weight = (
                plan.training_weights if reweighting == "inverse_probability" else None
            )
            exact_budget = bool(selected.size == row_budget)
            preservation = evaluate_subspace_preservation(
                standardized,
                plan,
                rank=min(standardized.shape[1], selected.size),
            )
            gram_error = preservation.gram_relative_error
        model = HistGradientBoostingRegressor(
            max_iter=120,
            max_depth=4,
            learning_rate=0.06,
            l2_regularization=0.1,
            random_state=seed,
        )
        model.fit(X_train[selected], y_train[selected], sample_weight=sample_weight)
        rmse = float(np.sqrt(mean_squared_error(y_test, model.predict(X_test))))
        game = make_masked_model_game(baseline, explicand, model.predict)
        exact = exact_shapley_values(game, X.shape[1])
        if arm_name == "D0_full_rows":
            full_reference = exact.copy()
        prepared.append((arm_name, game, exact, exact_budget, gram_error, rmse))
    if full_reference is None:
        raise RuntimeError("full-data explanation reference was not built")
    for arm_name, game, exact, exact_budget, gram_error, rmse in prepared:
        cases.append(
            ExplanationCase(
                scenario="diabetes_hist_gradient_boosting",
                row_arm=arm_name,
                game=game,
                exact_values=exact,
                full_reference_values=full_reference,
                downstream_rmse=rmse,
                row_gram_error=gram_error,
                row_budget_exact=exact_budget,
            )
        )
    return tuple(cases)


def _build_cases(
    config: LeverageShapExperimentConfig,
    seed: int,
) -> tuple[ExplanationCase, ...]:
    cases: list[ExplanationCase] = []
    for scenario, game in _synthetic_games(config.n_players):
        exact = exact_shapley_values(game, config.n_players)
        cases.append(
            ExplanationCase(
                scenario=scenario,
                row_arm="not_applicable",
                game=game,
                exact_values=exact,
                full_reference_values=exact,
                downstream_rmse=None,
                row_gram_error=None,
                row_budget_exact=True,
            )
        )
    cases.extend(_fit_diabetes_cases(config, seed))
    return tuple(cases)


def _evaluate_estimator(
    case: ExplanationCase,
    *,
    seed: int,
    budget_multiplier: float,
    policy: CoalitionSamplingPolicy,
) -> dict[str, Any]:
    n_players = int(case.exact_values.size)
    budget = max(6, int(round(n_players * budget_multiplier)))
    started = perf_counter()
    estimate = estimate_projected_shapley(
        case.game,
        n_players,
        budget=budget,
        policy=policy,
        random_state=seed,
    )
    elapsed = perf_counter() - started
    difference = estimate.values - case.exact_values
    relative_error = float(
        np.linalg.norm(difference) / max(np.linalg.norm(case.exact_values), 1e-12)
    )
    full_drift = float(
        np.linalg.norm(case.exact_values - case.full_reference_values)
        / max(np.linalg.norm(case.full_reference_values), 1e-12)
    )
    exact_loss = full_kernel_regression_loss(case.game, case.exact_values)
    estimated_loss = full_kernel_regression_loss(case.game, estimate.values)
    return {
        "run_key": (
            f"{case.scenario}|{case.row_arm}|seed={seed}"
            f"|budget={budget_multiplier:.12g}|{policy.value}"
        ),
        "status": "completed",
        "scenario": case.scenario,
        "row_arm": case.row_arm,
        "seed": int(seed),
        "budget_multiplier": float(budget_multiplier),
        "requested_model_evaluations": int(budget),
        "model_evaluations": int(estimate.model_evaluations),
        "coalition_policy": policy.value,
        "shap_mse": float(np.mean(np.square(difference))),
        "shap_relative_l2_error": relative_error,
        "full_kernel_loss": estimated_loss,
        "full_kernel_excess_loss": float(max(estimated_loss - exact_loss, 0.0)),
        "efficiency_residual": estimate.efficiency_residual,
        "condition_number": estimate.condition_number,
        "explanation_drift_vs_full_rows": full_drift,
        "downstream_rmse": case.downstream_rmse,
        "row_gram_relative_error": case.row_gram_error,
        "row_budget_exact": case.row_budget_exact,
        "runtime_seconds": float(elapsed),
        "estimate": estimate.to_dict(include_values=True),
        "exact_values": case.exact_values.tolist(),
    }


def _flatten_records(records: Sequence[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {key: value for key, value in record.items() if key not in {"estimate", "exact_values"}}
            for record in records
        ]
    )


def _build_paired(raw: pd.DataFrame) -> pd.DataFrame:
    keys = ["scenario", "row_arm", "seed", "budget_multiplier"]
    pivot = raw.pivot(index=keys, columns="coalition_policy", values="shap_relative_l2_error").reset_index()
    pivot["leverage_gain"] = pivot["kernel_weight"] - pivot["leverage"]
    pivot["leverage_relative_gain"] = pivot["leverage_gain"] / pivot["kernel_weight"].clip(lower=1e-12)
    return pivot


def _build_summary(raw: pd.DataFrame) -> pd.DataFrame:
    return (
        raw.groupby(
            ["scenario", "row_arm", "coalition_policy", "budget_multiplier"],
            as_index=False,
        )
        .agg(
            median_relative_error=("shap_relative_l2_error", "median"),
            mean_relative_error=("shap_relative_l2_error", "mean"),
            worst_relative_error=("shap_relative_l2_error", "max"),
            median_efficiency_residual=("efficiency_residual", "median"),
            median_runtime_seconds=("runtime_seconds", "median"),
            median_downstream_rmse=("downstream_rmse", "median"),
            median_explanation_drift=("explanation_drift_vs_full_rows", "median"),
            median_row_gram_error=("row_gram_relative_error", "median"),
        )
        .sort_values(["scenario", "row_arm", "budget_multiplier", "coalition_policy"])
        .reset_index(drop=True)
    )


def _build_gate(raw: pd.DataFrame, paired: pd.DataFrame, expected: int) -> dict[str, Any]:
    informative = paired[
        (paired["kernel_weight"] > 1e-10) | (paired["leverage"] > 1e-10)
    ]
    median_gain = float(informative["leverage_gain"].median())
    mean_gain = float(informative["leverage_gain"].mean())
    win_rate = float(np.mean(informative["leverage_gain"] > 0.0))
    checks = {
        "all_records_completed": bool(len(raw) == expected and (raw["status"] == "completed").all()),
        "row_budgets_exact": bool(raw["row_budget_exact"].all()),
        "projected_efficiency_holds": bool(float(raw["efficiency_residual"].max()) <= 1e-8),
        "informative_median_leverage_gain_positive": bool(median_gain > 0.0),
        "informative_mean_leverage_gain_positive": bool(mean_gain > 0.0),
        "informative_leverage_win_rate_at_least_60pct": bool(win_rate >= 0.60),
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "informative_pair_count": int(len(informative)),
        "median_leverage_gain": median_gain,
        "mean_leverage_gain": mean_gain,
        "leverage_win_rate": win_rate,
    }


def _plot_error_curves(summary: pd.DataFrame, output_dir: Path) -> None:
    display = summary.copy()
    display["case"] = display["scenario"] + " / " + display["row_arm"]
    cases = list(display["case"].drop_duplicates())
    columns = 2
    rows = int(np.ceil(len(cases) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(11, 3.8 * rows), squeeze=False)
    for axis, case_name in zip(axes.flat, cases):
        case = display[display["case"] == case_name]
        for policy, group in case.groupby("coalition_policy"):
            axis.plot(
                group["budget_multiplier"],
                group["median_relative_error"],
                marker="o",
                label=policy,
            )
        axis.set_title(case_name, fontsize=9)
        axis.set_xlabel("Model-evaluation budget / features")
        axis.set_ylabel("Median relative SHAP error")
        axis.set_yscale("log")
        axis.grid(alpha=0.25)
    for axis in axes.flat[len(cases):]:
        axis.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(output_dir / "shap_error_budget_curves.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_row_tradeoff(summary: pd.DataFrame, output_dir: Path) -> None:
    rows = summary[
        (summary["scenario"] == "diabetes_hist_gradient_boosting")
        & (summary["coalition_policy"] == "leverage")
        & (summary["budget_multiplier"] == summary["budget_multiplier"].max())
    ].drop_duplicates("row_arm")
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].bar(rows["row_arm"], rows["median_downstream_rmse"], color="#4477AA")
    axes[0].set_ylabel("Test RMSE")
    axes[1].bar(rows["row_arm"], rows["median_explanation_drift"], color="#CC6677")
    axes[1].set_ylabel("Exact-SHAP drift vs full-row model")
    for axis in axes:
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "row_sampling_quality_explanation_tradeoff.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def _write_artifacts(
    output_dir: Path,
    config: LeverageShapExperimentConfig,
    records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    raw = _flatten_records(records)
    paired = _build_paired(raw)
    summary = _build_summary(raw)
    expected = len(config.seeds) * (2 + 3) * len(config.budget_multipliers) * 2
    gate = _build_gate(raw, paired, expected)
    raw.to_csv(output_dir / "leverage_shap_raw_runs.csv", index=False)
    paired.to_csv(output_dir / "leverage_shap_paired.csv", index=False)
    summary.to_csv(output_dir / "leverage_shap_summary.csv", index=False)
    (output_dir / "leverage_shap_gate.json").write_text(
        json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    methodology = {
        "row_sampling_axis": ["full", "uniform", "robust_leverage_mixture_with_ipw"],
        "coalition_sampling_axis": ["kernel_weight", "leverage"],
        "reference": "exact Shapley enumeration",
        "primary_source": "https://arxiv.org/abs/2410.01917",
        "official_implementation": "https://github.com/rtealwitter/leverageshap",
        "distinction": (
            "Sampling Zoo leverage selects training rows; Leverage SHAP leverage "
            "selects feature coalitions for a projected explanation regression."
        ),
    }
    (output_dir / "methodology.json").write_text(
        json.dumps(methodology, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _plot_error_curves(summary, output_dir)
    _plot_row_tradeoff(summary, output_dir)
    compact = (
        paired.groupby(["scenario", "row_arm", "budget_multiplier"], as_index=False)
        .agg(
            median_leverage_gain=("leverage_gain", "median"),
            win_rate=("leverage_gain", lambda values: float(np.mean(values >= 0.0))),
        )
    )
    report = "\n".join(
        [
            "# Leverage SHAP и RMT-семплирование строк",
            "",
            f"Статус критерия допуска: **{'пройден' if gate['passed'] else 'не пройден'}**.",
            "",
            "Эксперимент разделяет два разных применения leverage-оценок: выбор строк "
            "для обучения и выбор коалиций признаков для SHAP-регрессии.",
            "",
            "## Общий результат",
            "",
            markdown_table(pd.DataFrame([gate])),
            "",
            "## Попарное сравнение способов выбора коалиций",
            "",
            markdown_table(compact),
            "",
            "## Сводка по всем осям",
            "",
            markdown_table(summary),
            "",
            "Положительный `median_leverage_gain` означает меньшую относительную "
            "ошибку Leverage SHAP при том же числе вызовов модели.",
            "",
        ]
    )
    (output_dir / "leverage_shap_report.md").write_text(report, encoding="utf-8")
    return gate


def run_rmt_leverage_shap_experiment(
    config: LeverageShapExperimentConfig | None = None,
) -> Path:
    resolved = config or LeverageShapExperimentConfig()
    output_dir = resolved.output_dir or (
        BENCHMARK_DIR
        / "results"
        / ("run_rmt_leverage_shap_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "leverage_shap_runs.jsonl"
    records: list[dict[str, Any]] = []
    if records_path.exists():
        records = [
            json.loads(line)
            for line in records_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    completed = {record["run_key"] for record in records if record.get("status") == "completed"}
    expected = len(resolved.seeds) * 5 * len(resolved.budget_multipliers) * 2
    meta_path = output_dir / "run_meta.json"
    try:
        with tqdm(
            total=expected,
            initial=len(completed),
            desc="RMT x Leverage SHAP",
            disable=not resolved.show_progress,
            unit="run",
        ) as progress:
            for seed in resolved.seeds:
                for case in _build_cases(resolved, seed):
                    for multiplier in resolved.budget_multipliers:
                        for policy in CoalitionSamplingPolicy:
                            key = (
                                f"{case.scenario}|{case.row_arm}|seed={seed}"
                                f"|budget={multiplier:.12g}|{policy.value}"
                            )
                            if key in completed:
                                continue
                            record = _evaluate_estimator(
                                case,
                                seed=seed,
                                budget_multiplier=multiplier,
                                policy=policy,
                            )
                            with records_path.open("a", encoding="utf-8") as stream:
                                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                            records.append(record)
                            completed.add(key)
                            progress.update(1)
                            meta_path.write_text(
                                json.dumps(
                                    {
                                        "experiment": "rmt_leverage_shap",
                                        "status": "running",
                                        "record_count": len(records),
                                        "expected_records": expected,
                                        "config": asdict(resolved) | {"output_dir": str(output_dir)},
                                    },
                                    ensure_ascii=False,
                                    indent=2,
                                    default=str,
                                ),
                                encoding="utf-8",
                            )
        gate = _write_artifacts(output_dir, resolved, records)
        meta = {
            "experiment": "rmt_leverage_shap",
            "status": "completed" if len(records) == expected else "incomplete",
            "record_count": len(records),
            "expected_records": expected,
            "gate_passed": gate["passed"],
            "config": asdict(resolved) | {"output_dir": str(output_dir)},
        }
    except Exception as exc:
        meta = {
            "experiment": "rmt_leverage_shap",
            "status": "failed",
            "record_count": len(records),
            "expected_records": expected,
            "error": f"{type(exc).__name__}: {exc}",
        }
        meta_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        raise
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return output_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = LeverageShapExperimentConfig(
        budget_multipliers=(4.0,) if args.smoke else DEFAULT_SHAP_BUDGET_MULTIPLIERS,
        seeds=(11,) if args.smoke else DEFAULT_SHAP_SEEDS,
        output_dir=args.output_dir,
        show_progress=not args.no_progress,
    )
    result = run_rmt_leverage_shap_experiment(config)
    print(f"Leverage SHAP artifacts: {result}")


if __name__ == "__main__":
    main()
