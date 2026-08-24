from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "source" / "routing_geometry_replay.csv"
RAW_SOURCE = ROOT / "source" / "routing_geometry_runs.jsonl"
RUN_META = ROOT / "source" / "run_meta.json"

KEYS = ["dataset", "seed", "budget_ratio", "model"]
ARMS = [
    "B0_standard_A9",
    "B1_bulk_single_spike",
    "B2_bulk_multi_spike",
    "B3_validation_selected",
]
SPECIALIZED_ARMS = ["B1_bulk_single_spike", "B2_bulk_multi_spike"]
AMENDED_PRACTICAL_BRIER_MARGIN = 0.02


def _relative_lower_gain(candidate: pd.Series, baseline: pd.Series) -> pd.Series:
    return (baseline - candidate) / baseline.abs().clip(lower=1e-12)


def _paired_results(records: pd.DataFrame) -> pd.DataFrame:
    comparison_columns = [
        "test_primary_value",
        "test_tail_mean_absolute_error",
        "test_log_loss",
        "test_brier_score",
        "test_expected_calibration_error",
        "test_f1_macro",
        "test_worst_class_recall",
        "runtime_fit_seconds",
        "runtime_inference_seconds",
    ]
    available = [column for column in comparison_columns if column in records]
    baseline = records.loc[
        records["arm_name"] == "B0_standard_A9", KEYS + available
    ].rename(columns={column: f"b0_{column}" for column in available})
    paired = records.merge(baseline, on=KEYS, how="left", validate="many_to_one")

    lower_primary = paired["test_primary_metric"].isin({"rmse", "log_loss"})
    paired["gain_vs_b0"] = np.where(
        lower_primary,
        _relative_lower_gain(
            paired["test_primary_value"], paired["b0_test_primary_value"]
        ),
        paired["test_primary_value"] - paired["b0_test_primary_value"],
    )

    for metric in ("tail_mean_absolute_error", "log_loss", "brier_score"):
        candidate = f"test_{metric}"
        reference = f"b0_test_{metric}"
        if candidate in paired and reference in paired:
            paired[f"{metric}_gain_vs_b0"] = _relative_lower_gain(
                paired[candidate], paired[reference]
            )

    for metric, direction in {
        "expected_calibration_error": "lower",
        "f1_macro": "higher",
        "worst_class_recall": "higher",
    }.items():
        candidate = f"test_{metric}"
        reference = f"b0_test_{metric}"
        if candidate in paired and reference in paired:
            paired[f"{metric}_gain_vs_b0"] = (
                paired[reference] - paired[candidate]
                if direction == "lower"
                else paired[candidate] - paired[reference]
            )

    for metric in ("fit_seconds", "inference_seconds"):
        candidate = f"runtime_{metric}"
        reference = f"b0_runtime_{metric}"
        if candidate in paired and reference in paired:
            paired[f"{metric}_ratio_vs_b0"] = (
                paired[candidate] / paired[reference].clip(lower=1e-12)
            )
    return paired


def _bootstrap_dataset_mean(
    frame: pd.DataFrame,
    *,
    value_column: str,
    iterations: int = 10_000,
    seed: int = 20260819,
) -> dict[str, float | int | None]:
    values = (
        frame.groupby("dataset", observed=True)[value_column]
        .mean()
        .dropna()
        .to_numpy(dtype=float)
    )
    if values.size == 0:
        return {"datasets": 0, "mean": None, "ci_low": None, "ci_high": None}
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(iterations, values.size), replace=True).mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return {
        "datasets": int(values.size),
        "mean": float(values.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
    }


def _gain_summary(frame: pd.DataFrame) -> pd.Series:
    values = frame["gain_vs_b0"].dropna()
    if values.empty:
        return pd.Series(
            {
                "runs": 0,
                "mean_gain": None,
                "median_gain": None,
                "worst_gain": None,
                "best_gain": None,
                "win_rate": None,
                "harm_rate": None,
            },
            dtype=object,
        )
    return pd.Series(
        {
            "runs": int(len(values)),
            "mean_gain": float(values.mean()),
            "median_gain": float(values.median()),
            "worst_gain": float(values.min()),
            "best_gain": float(values.max()),
            "win_rate": float((values > 0).mean()),
            "harm_rate": float((values < 0).mean()),
        }
    )


def _allocation_summary(selected: pd.DataFrame) -> dict[str, Any]:
    if selected.empty:
        return {"runs": 0}
    columns = [
        "bulk_train_share",
        "spike_train_share",
        "bulk_routing_hard_share",
        "spike_routing_hard_share",
        "bulk_routing_soft_share",
        "spike_routing_soft_share",
    ]
    output: dict[str, Any] = {"runs": int(len(selected))}
    for column in columns:
        values = pd.to_numeric(selected[column], errors="coerce").dropna()
        if values.empty:
            continue
        output[column] = {
            "median": float(values.median()),
            "q25": float(values.quantile(0.25)),
            "q75": float(values.quantile(0.75)),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return output


def _selected_allocations() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for line in RAW_SOURCE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("arm_name") != "B3_validation_selected":
            continue
        if record.get("selected_arm_name") not in SPECIALIZED_ARMS:
            continue
        partition_sizes = record.get("partition_sizes") or {}
        bulk_train_rows = int(partition_sizes.get("bulk", 0))
        spike_train_rows = int(
            sum(
                int(value)
                for key, value in partition_sizes.items()
                if str(key).startswith("spike")
            )
        )
        total_train_rows = bulk_train_rows + spike_train_rows
        test_routing = (record.get("test") or {}).get("routing") or {}
        hard_counts = test_routing.get("hard_assignment_counts") or {}
        soft_mass = test_routing.get("soft_assignment_mass") or {}
        bulk_hard_rows = int(hard_counts.get("bulk", 0))
        spike_hard_rows = int(
            sum(
                int(value)
                for key, value in hard_counts.items()
                if str(key).startswith("spike")
            )
        )
        total_hard_rows = bulk_hard_rows + spike_hard_rows
        bulk_soft_mass = float(soft_mass.get("bulk", 0.0))
        spike_soft_mass = float(
            sum(
                float(value)
                for key, value in soft_mass.items()
                if str(key).startswith("spike")
            )
        )
        total_soft_mass = bulk_soft_mass + spike_soft_mass
        rows.append(
            {
                "dataset": record["dataset"],
                "problem_type": record["problem_type"],
                "seed": record["seed"],
                "budget_ratio": record["budget_ratio"],
                "selected_arm_name": record["selected_arm_name"],
                "bulk_train_rows": bulk_train_rows,
                "spike_train_rows": spike_train_rows,
                "bulk_train_share": bulk_train_rows / max(total_train_rows, 1),
                "spike_train_share": spike_train_rows / max(total_train_rows, 1),
                "bulk_routing_hard_rows": bulk_hard_rows,
                "spike_routing_hard_rows": spike_hard_rows,
                "bulk_routing_hard_share": bulk_hard_rows / max(total_hard_rows, 1),
                "spike_routing_hard_share": spike_hard_rows / max(total_hard_rows, 1),
                "bulk_routing_soft_share": bulk_soft_mass / max(total_soft_mass, 1e-12),
                "spike_routing_soft_share": spike_soft_mass / max(total_soft_mass, 1e-12),
            }
        )
    return pd.DataFrame(rows)


def _plot_candidate_gain(summary: pd.DataFrame) -> None:
    colors = {
        "B1_bulk_single_spike": "#2F6B9A",
        "B2_bulk_multi_spike": "#2C8C69",
        "B3_validation_selected": "#C76B2A",
    }
    panels = [
        ("classification", "log_loss", "Классификация: относительное снижение LogLoss"),
        ("classification", "roc_auc", "Классификация: абсолютный прирост ROC AUC"),
        ("regression", "rmse", "Регрессия: относительное снижение RMSE"),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    for axis, (problem_type, primary_metric, title) in zip(axes, panels):
        subset = summary[
            (summary["problem_type"] == problem_type)
            & (summary["test_primary_metric"] == primary_metric)
        ]
        for arm_name, group in subset.groupby("arm_name", observed=True):
            axis.plot(
                group["budget_ratio"] * 100,
                group["dataset_balanced_mean_gain"],
                marker="o",
                linewidth=2,
                color=colors[arm_name],
                label=arm_name,
            )
        axis.axhline(0, color="#333333", linewidth=1)
        axis.grid(alpha=0.25)
        axis.set_xlabel("Бюджет, % от обучающей выборки")
        axis.set_title(title)
        axis.set_ylabel("Выигрыш относительно B0")
        axis.set_xticks([1, 5, 10, 20])
    axes[0].legend(loc="lower right", fontsize=8, frameon=True)
    figure.savefig(ROOT / "candidate_gain_by_budget.png", dpi=180)
    plt.close(figure)


def _plot_selection_mix(selection_mix: pd.DataFrame) -> None:
    colors = {
        "B0_standard_A9": "#777777",
        "B1_bulk_single_spike": "#2F6B9A",
        "B2_bulk_multi_spike": "#2C8C69",
    }
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for axis, problem_type in zip(axes, ["classification", "regression"]):
        pivot = (
            selection_mix[selection_mix["problem_type"] == problem_type]
            .pivot(index="budget_ratio", columns="selected_arm_name", values="share")
            .fillna(0)
            .reindex(columns=list(colors), fill_value=0)
        )
        bottom = np.zeros(len(pivot))
        for arm_name in colors:
            values = pivot[arm_name].to_numpy()
            axis.bar(
                pivot.index.to_numpy() * 100,
                values,
                bottom=bottom,
                width=2.8,
                color=colors[arm_name],
                label=arm_name,
            )
            bottom += values
        axis.set_ylim(0, 1)
        axis.set_xlabel("Бюджет, % от обучающей выборки")
        axis.set_ylabel("Доля решений B3")
        axis.set_title("Классификация" if problem_type == "classification" else "Регрессия")
        axis.grid(axis="y", alpha=0.25)
        axis.set_xticks([1, 5, 10, 20])
    axes[0].legend(loc="lower left", fontsize=8, frameon=True)
    figure.savefig(ROOT / "b3_selection_mix.png", dpi=180)
    plt.close(figure)


def main() -> dict[str, Any]:
    records = pd.read_csv(SOURCE)
    run_meta = json.loads(RUN_META.read_text(encoding="utf-8"))

    duplicate_rows = int(records.duplicated(KEYS + ["arm_name"]).sum())
    group_arms = records.groupby(KEYS, observed=True)["arm_name"].agg(set)
    complete_groups = group_arms[group_arms == set(ARMS)].index
    complete_index = pd.MultiIndex.from_tuples(complete_groups, names=KEYS)
    complete_mask = pd.MultiIndex.from_frame(records[KEYS]).isin(complete_index)
    analysis_records = records.loc[complete_mask].copy()
    paired = _paired_results(analysis_records)

    candidate = paired[paired["arm_name"].isin(SPECIALIZED_ARMS + ["B3_validation_selected"])].copy()
    run_summary = (
        candidate.groupby(
            ["problem_type", "test_primary_metric", "budget_ratio", "arm_name"],
            observed=True,
        )
        .apply(_gain_summary)
        .reset_index()
    )
    dataset_summary = (
        candidate.groupby(
            [
                "problem_type",
                "test_primary_metric",
                "budget_ratio",
                "arm_name",
                "dataset",
            ],
            observed=True,
            as_index=False,
        )["gain_vs_b0"]
        .mean()
        .groupby(
            ["problem_type", "test_primary_metric", "budget_ratio", "arm_name"],
            observed=True,
        )
        .agg(dataset_balanced_mean_gain=("gain_vs_b0", "mean"), datasets=("dataset", "size"))
        .reset_index()
    )
    candidate_summary = run_summary.merge(
        dataset_summary,
        on=["problem_type", "test_primary_metric", "budget_ratio", "arm_name"],
        validate="one_to_one",
    )
    candidate_summary.to_csv(ROOT / "candidate_gain_summary.csv", index=False)

    b3 = paired[paired["arm_name"] == "B3_validation_selected"].copy()
    selected = b3[b3["selected_arm_name"].isin(SPECIALIZED_ARMS)].copy()
    selected["outcome"] = np.select(
        [selected["gain_vs_b0"] > 0, selected["gain_vs_b0"] < 0],
        ["win", "harm"],
        default="tie",
    )
    selected.sort_values("gain_vs_b0", ascending=False).to_csv(
        ROOT / "selected_specialized_cases.csv", index=False
    )
    selected_allocations = _selected_allocations()
    selected_allocations.to_csv(ROOT / "selected_specialized_allocation.csv", index=False)

    selection_counts = (
        b3.groupby(["problem_type", "budget_ratio", "selected_arm_name"], observed=True)
        .size()
        .rename("runs")
        .reset_index()
    )
    selection_counts["share"] = selection_counts["runs"] / selection_counts.groupby(
        ["problem_type", "budget_ratio"], observed=True
    )["runs"].transform("sum")
    selection_counts.to_csv(ROOT / "b3_selection_mix.csv", index=False)

    selected_by_dataset = (
        b3.assign(selected_specialized=b3["selected_arm_name"].isin(SPECIALIZED_ARMS))
        .groupby(["problem_type", "dataset"], observed=True)
        .agg(
            runs=("seed", "size"),
            selected_specialized=("selected_specialized", "sum"),
            mean_b3_gain=("gain_vs_b0", "mean"),
            worst_b3_gain=("gain_vs_b0", "min"),
        )
        .reset_index()
    )
    selected_by_dataset["selection_rate"] = (
        selected_by_dataset["selected_specialized"] / selected_by_dataset["runs"]
    )
    selected_by_dataset.to_csv(ROOT / "b3_selection_by_dataset.csv", index=False)

    classification = b3[b3["problem_type"] == "classification"]
    classification_log_loss = classification[
        classification["test_primary_metric"] == "log_loss"
    ]
    classification_roc_auc = classification[
        classification["test_primary_metric"] == "roc_auc"
    ]
    regression = b3[b3["problem_type"] == "regression"]
    selected_classification = selected[selected["problem_type"] == "classification"]
    selected_classification_log_loss = selected_classification[
        selected_classification["test_primary_metric"] == "log_loss"
    ]
    selected_classification_roc_auc = selected_classification[
        selected_classification["test_primary_metric"] == "roc_auc"
    ]
    selected_regression = selected[selected["problem_type"] == "regression"]

    original_brier_margin = float(run_meta["gate_profile"]["brier_harm_margin"])
    brier_failures = classification[
        classification["brier_score_gain_vs_b0"] < -original_brier_margin
    ].copy()
    brier_failures["gate_metric"] = "brier_score_gain_vs_b0"
    brier_failures["minimum_allowed_gain"] = -original_brier_margin
    brier_failures[
        KEYS
        + [
            "selected_arm_name",
            "test_primary_metric",
            "test_primary_value",
            "b0_test_primary_value",
            "test_brier_score",
            "b0_test_brier_score",
            "brier_score_gain_vs_b0",
            "gate_metric",
            "minimum_allowed_gain",
        ]
    ].to_csv(ROOT / "gate_failure_cases.csv", index=False)

    provisional_checks = {
        "all_downloaded_records_completed": bool(
            (analysis_records["status"] == "completed").all()
        ),
        "all_groups_have_four_arms": bool(len(analysis_records) == len(complete_groups) * 4),
        "no_duplicate_arm_records": duplicate_rows == 0,
        "median_gain_nonnegative": bool(b3["gain_vs_b0"].median() >= 0),
        "worst_harm_within_margin": bool(
            b3["gain_vs_b0"].min()
            >= -float(run_meta["gate_profile"]["primary_harm_margin"])
        ),
        "selected_specialization_win_rate_at_least_80pct": bool(
            not selected.empty and (selected["gain_vs_b0"] > 0).mean() >= 0.8
        ),
        "regression_tail_noninferiority": bool(
            regression["tail_mean_absolute_error_gain_vs_b0"].dropna().min()
            >= -float(run_meta["gate_profile"]["tail_harm_margin"])
        ),
        "classification_brier_noninferiority_original": bool(
            classification["brier_score_gain_vs_b0"].dropna().min()
            >= -original_brier_margin
        ),
        "classification_brier_noninferiority_amended": bool(
            classification["brier_score_gain_vs_b0"].dropna().min()
            >= -AMENDED_PRACTICAL_BRIER_MARGIN
        ),
        "classification_ece_noninferiority": bool(
            classification["expected_calibration_error_gain_vs_b0"].dropna().min()
            >= -float(run_meta["gate_profile"]["ece_harm_margin"])
        ),
        "classification_f1_noninferiority": bool(
            classification["f1_macro_gain_vs_b0"].dropna().min()
            >= -float(run_meta["gate_profile"]["f1_macro_harm_margin"])
        ),
        "classification_worst_recall_noninferiority": bool(
            classification["worst_class_recall_gain_vs_b0"].dropna().min()
            >= -float(run_meta["gate_profile"]["worst_class_recall_harm_margin"])
        ),
    }

    summary: dict[str, Any] = {
        "snapshot": {
            "source_updated_at": run_meta["updated_at"],
            "downloaded_records": int(len(records)),
            "expected_records": 1200,
            "coverage": float(len(records) / 1200),
            "complete_groups": int(len(complete_groups)),
            "datasets": int(records["dataset"].nunique()),
            "duplicate_arm_records": duplicate_rows,
            "failed_records": int((records["status"] != "completed").sum()),
            "original_brier_harm_margin": original_brier_margin,
            "amended_brier_harm_margin": AMENDED_PRACTICAL_BRIER_MARGIN,
        },
        "selection": {
            "b3_runs": int(len(b3)),
            "selected_arm_counts": {
                str(key): int(value)
                for key, value in b3["selected_arm_name"].value_counts().items()
            },
            "specialized_count": int(len(selected)),
            "specialized_rate": float(len(selected) / len(b3)),
            "specialized_win_rate": float((selected["gain_vs_b0"] > 0).mean())
            if len(selected)
            else None,
            "specialized_harm_rate": float((selected["gain_vs_b0"] < 0).mean())
            if len(selected)
            else None,
            "classification_specialized_count": int(len(selected_classification)),
            "regression_specialized_count": int(len(selected_regression)),
        },
        "b3": {
            "classification_log_loss": _gain_summary(classification_log_loss).to_dict(),
            "classification_roc_auc": _gain_summary(classification_roc_auc).to_dict(),
            "regression": _gain_summary(regression).to_dict(),
            "classification_log_loss_dataset_bootstrap": _bootstrap_dataset_mean(
                classification_log_loss, value_column="gain_vs_b0"
            ),
            "classification_roc_auc_dataset_bootstrap": _bootstrap_dataset_mean(
                classification_roc_auc, value_column="gain_vs_b0"
            ),
            "regression_dataset_bootstrap": _bootstrap_dataset_mean(
                regression, value_column="gain_vs_b0"
            ),
        },
        "selected_specialized": {
            "classification_log_loss": _gain_summary(
                selected_classification_log_loss
            ).to_dict(),
            "classification_roc_auc": _gain_summary(
                selected_classification_roc_auc
            ).to_dict(),
            "regression": _gain_summary(selected_regression).to_dict(),
            "allocation": _allocation_summary(selected_allocations),
        },
        "secondary_metrics": {
            "classification_b3_mean_log_loss_gain": float(
                classification["log_loss_gain_vs_b0"].mean()
            ),
            "classification_b3_mean_brier_gain": float(
                classification["brier_score_gain_vs_b0"].mean()
            ),
            "classification_b3_mean_f1_delta": float(
                classification["f1_macro_gain_vs_b0"].mean()
            ),
            "classification_b3_mean_worst_recall_delta": float(
                classification["worst_class_recall_gain_vs_b0"].mean()
            ),
            "regression_b3_mean_tail_gain": float(
                regression["tail_mean_absolute_error_gain_vs_b0"].mean()
            ),
            "classification_low_budget_b3_mean_log_loss_gain": float(
                classification.loc[
                    classification["budget_ratio"] <= 0.05,
                    "log_loss_gain_vs_b0",
                ].mean()
            ),
        },
        "runtime": {
            arm: {
                "fit_ratio_median": float(
                    paired.loc[paired["arm_name"] == arm, "fit_seconds_ratio_vs_b0"].median()
                ),
                "inference_ratio_median": float(
                    paired.loc[
                        paired["arm_name"] == arm, "inference_seconds_ratio_vs_b0"
                    ].median()
                ),
            }
            for arm in SPECIALIZED_ARMS
        },
        "provisional_gate_checks": provisional_checks,
    }
    (ROOT / "preliminary_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    _plot_candidate_gain(candidate_summary)
    _plot_selection_mix(selection_counts)
    return summary


if __name__ == "__main__":
    print(json.dumps(main(), ensure_ascii=False, indent=2))
