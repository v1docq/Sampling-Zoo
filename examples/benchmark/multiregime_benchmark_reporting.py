"""Artifact construction for multi-regime RMT cluster validation."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


SUMMARY_KEYS = (
    "noise_distribution",
    "regime_profile",
    "n_regimes",
    "view_strategy",
    "backend",
    "partition_policy",
    "snr",
)
PAIR_KEYS = (
    "noise_distribution",
    "regime_profile",
    "n_regimes",
    "view_strategy",
    "partition_policy",
    "snr",
    "seed",
)
POLICY_PAIR_KEYS = (
    "noise_distribution",
    "regime_profile",
    "n_regimes",
    "view_strategy",
    "backend",
    "snr",
    "seed",
)


class MultiRegimeBenchmarkArtifactBuilder:
    def build(
        self,
        records: Sequence[Mapping[str, Any]],
        output_root: str | Path,
    ) -> dict[str, Path]:
        root = Path(output_root)
        metrics_dir = root / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        raw = self._raw_table(records)
        summary = self._summary_table(raw)
        algorithms = self._algorithm_frequency(raw)
        policy_regret = self._policy_regret(raw)
        backend_agreement = self._backend_agreement(raw)
        paths = self._write_tables(
            metrics_dir,
            raw,
            summary,
            algorithms,
            policy_regret,
            backend_agreement,
        )
        paths["report"] = self._write_report(
            root,
            raw,
            summary,
            algorithms,
            policy_regret,
            backend_agreement,
        )
        return paths

    @staticmethod
    def _raw_table(records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
        rows = []
        for record in records:
            row = dict(record)
            row["diagnostics_available"] = bool(row.pop("diagnostics", {}))
            row["extra_json"] = json.dumps(
                row.pop("extra", {}),
                ensure_ascii=False,
                sort_keys=True,
                default=_json_default,
            )
            rows.append(row)
        return pd.DataFrame(rows)

    @staticmethod
    def _summary_table(raw: pd.DataFrame) -> pd.DataFrame:
        completed = _completed(raw)
        if completed.empty:
            return pd.DataFrame()
        return (
            completed.groupby(list(SUMMARY_KEYS), dropna=False, sort=True)
            .agg(
                runs=("grid_key", "count"),
                selected_n_partitions_mean=("predicted_n_clusters", "mean"),
                cluster_count_absolute_error_mean=(
                    "cluster_count_absolute_error",
                    "mean",
                ),
                cluster_count_exact_rate=("cluster_count_exact", "mean"),
                adjusted_rand_index_mean=("adjusted_rand_index", "mean"),
                adjusted_rand_index_std=("adjusted_rand_index", "std"),
                normalized_mutual_information_mean=(
                    "normalized_mutual_information",
                    "mean",
                ),
                aligned_accuracy_mean=("aligned_accuracy", "mean"),
                purity_mean=("purity", "mean"),
                min_true_regime_recall_mean=(
                    "min_true_regime_recall",
                    "mean",
                ),
                predicted_imbalance_ratio_mean=(
                    "predicted_imbalance_ratio",
                    "mean",
                ),
                candidate_set_contains_true_rate=(
                    "candidate_set_contains_true_n",
                    "mean",
                ),
                count_based_candidate_set_contains_true_rate=(
                    "count_based_candidate_set_contains_true_n",
                    "mean",
                ),
                density_candidate_rescue_rate=(
                    "density_candidate_rescued_true_n",
                    "mean",
                ),
                selected_subspace_recall_mean=(
                    "selected_subspace_recall",
                    "mean",
                ),
                fit_time_mean_sec=("fit_time_sec", "mean"),
            )
            .reset_index()
        )

    @staticmethod
    def _algorithm_frequency(raw: pd.DataFrame) -> pd.DataFrame:
        completed = _completed(raw)
        if completed.empty:
            return pd.DataFrame()
        auto = completed[completed["partition_policy"] != "fixed_oracle"]
        if auto.empty:
            return pd.DataFrame()
        keys = [
            "partition_policy",
            "noise_distribution",
            "regime_profile",
            "selected_cluster_algorithm",
        ]
        result = auto.groupby(keys, dropna=False).size().rename("runs").reset_index()
        totals = result.groupby(keys[:-1])["runs"].transform("sum")
        result["selection_frequency"] = result["runs"] / totals
        return result.sort_values(keys).reset_index(drop=True)

    @staticmethod
    def _policy_regret(raw: pd.DataFrame) -> pd.DataFrame:
        completed = _completed(raw)
        if completed.empty:
            return pd.DataFrame()
        oracle = completed[completed["partition_policy"] == "fixed_oracle"]
        auto = completed[completed["partition_policy"] != "fixed_oracle"]
        if oracle.empty or auto.empty:
            return pd.DataFrame()
        metrics = (
            "adjusted_rand_index",
            "aligned_accuracy",
            "purity",
            "min_true_regime_recall",
        )
        oracle = oracle[[*POLICY_PAIR_KEYS, *metrics]].rename(
            columns={metric: f"{metric}_oracle" for metric in metrics}
        )
        paired = auto.merge(oracle, on=list(POLICY_PAIR_KEYS), how="inner")
        for metric in metrics:
            paired[f"{metric}_regret_vs_oracle"] = (
                paired[f"{metric}_oracle"] - paired[metric]
            )
        return paired.sort_values([*POLICY_PAIR_KEYS, "partition_policy"]).reset_index(
            drop=True
        )

    @staticmethod
    def _backend_agreement(raw: pd.DataFrame) -> pd.DataFrame:
        completed = _completed(raw)
        if completed.empty or not {"numpy", "torch"}.issubset(
            set(completed["backend"])
        ):
            return pd.DataFrame()
        metrics = (
            "predicted_n_clusters",
            "adjusted_rand_index",
            "aligned_accuracy",
            "selected_subspace_recall",
        )
        numpy_runs = completed[completed["backend"] == "numpy"][[*PAIR_KEYS, *metrics]]
        torch_runs = completed[completed["backend"] == "torch"][[*PAIR_KEYS, *metrics]]
        paired = numpy_runs.merge(
            torch_runs,
            on=list(PAIR_KEYS),
            how="inner",
            suffixes=("_numpy", "_torch"),
        )
        for metric in metrics:
            paired[f"{metric}_absolute_delta"] = (
                paired[f"{metric}_numpy"] - paired[f"{metric}_torch"]
            ).abs()
        return paired.sort_values(list(PAIR_KEYS)).reset_index(drop=True)

    def _write_tables(
        self,
        metrics_dir: Path,
        raw: pd.DataFrame,
        summary: pd.DataFrame,
        algorithms: pd.DataFrame,
        policy_regret: pd.DataFrame,
        backend_agreement: pd.DataFrame,
    ) -> dict[str, Path]:
        frames = {
            "raw_csv": (metrics_dir / "rmt_multiregime_raw_runs.csv", raw),
            "summary_csv": (
                metrics_dir / "rmt_multiregime_summary_by_snr.csv",
                summary,
            ),
            "algorithm_frequency_csv": (
                metrics_dir / "rmt_multiregime_algorithm_frequency.csv",
                algorithms,
            ),
            "policy_regret_csv": (
                metrics_dir / "rmt_multiregime_policy_regret.csv",
                policy_regret,
            ),
            "backend_agreement_csv": (
                metrics_dir / "rmt_multiregime_backend_agreement.csv",
                backend_agreement,
            ),
        }
        for path, frame in frames.values():
            _atomic_write_csv(path, frame)
        return {name: path for name, (path, _) in frames.items()}

    def _write_report(
        self,
        root: Path,
        raw: pd.DataFrame,
        summary: pd.DataFrame,
        algorithms: pd.DataFrame,
        policy_regret: pd.DataFrame,
        backend_agreement: pd.DataFrame,
    ) -> Path:
        status_counts = (
            raw["status"].value_counts().to_dict() if "status" in raw else {}
        )
        high_snr = _highest_snr_rows(summary)
        regret_summary = _regret_summary(policy_regret)
        lines = [
            "# RMT multi-regime cluster validation",
            "",
            "This report separates spectral recovery, candidate-set coverage, "
            "and partition-objective quality on known low-rank regimes.",
            "",
            "## Run status",
            "",
            f"- Leaf runs: {len(raw)}",
            f"- Completed: {status_counts.get('completed', 0)}",
            f"- Failed: {status_counts.get('failed', 0)}",
            f"- Skipped: {status_counts.get('skipped', 0)}",
            "",
            "## Recovery at the highest SNR",
            "",
            _markdown_table(
                high_snr,
                (
                    "noise_distribution",
                    "regime_profile",
                    "n_regimes",
                    "backend",
                    "partition_policy",
                    "snr",
                    "selected_n_partitions_mean",
                    "cluster_count_exact_rate",
                    "adjusted_rand_index_mean",
                    "aligned_accuracy_mean",
                    "candidate_set_contains_true_rate",
                    "count_based_candidate_set_contains_true_rate",
                    "density_candidate_rescue_rate",
                ),
            ),
            "",
            "## Auto-policy regret versus fixed oracle k",
            "",
            _markdown_table(
                regret_summary,
                (
                    "partition_policy",
                    "paired_runs",
                    "ari_regret_mean",
                    "ari_regret_max",
                    "aligned_accuracy_regret_mean",
                ),
            ),
            "",
            "## Selected clustering algorithms",
            "",
            _markdown_table(
                algorithms,
                (
                    "partition_policy",
                    "noise_distribution",
                    "regime_profile",
                    "selected_cluster_algorithm",
                    "runs",
                    "selection_frequency",
                ),
            ),
            "",
            "## NumPy/Torch agreement",
            "",
            _backend_agreement_text(backend_agreement),
            "",
            "## Interpretation",
            "",
            "- `fixed_oracle` measures clustering quality when the true k is known; "
            "it is not a production method.",
            "- A low candidate-set coverage rate identifies planning constraints "
            "before the balanced objective is evaluated.",
            "- Count-based coverage excludes HDBSCAN; density rescue records when "
            "HDBSCAN restores the true k after the size guard prunes it.",
            "- Regret with candidate coverage equal to one is attributable to "
            "the objective or clustering adapter, not to candidate pruning.",
            "- Low oracle ARI at high SNR indicates an embedding or generative "
            "separability problem and should block selector-policy changes.",
            "",
            "Production partition selection remains unchanged by this benchmark.",
        ]
        path = root / "report.md"
        _atomic_write_text(path, "\n".join(lines) + "\n")
        return path


def _completed(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty or "status" not in raw:
        return pd.DataFrame()
    return raw[raw["status"] == "completed"].copy()


def _highest_snr_rows(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    keys = [column for column in SUMMARY_KEYS if column != "snr"]
    indices = summary.groupby(keys, dropna=False)["snr"].idxmax()
    return summary.loc[indices].sort_values(keys).reset_index(drop=True)


def _regret_summary(policy_regret: pd.DataFrame) -> pd.DataFrame:
    if policy_regret.empty:
        return pd.DataFrame()
    return (
        policy_regret.groupby("partition_policy", dropna=False)
        .agg(
            paired_runs=("grid_key", "count"),
            ari_regret_mean=("adjusted_rand_index_regret_vs_oracle", "mean"),
            ari_regret_max=("adjusted_rand_index_regret_vs_oracle", "max"),
            aligned_accuracy_regret_mean=(
                "aligned_accuracy_regret_vs_oracle",
                "mean",
            ),
        )
        .reset_index()
    )


def _backend_agreement_text(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No paired NumPy/Torch runs are available."
    delta_columns = [
        column for column in frame.columns if column.endswith("_absolute_delta")
    ]
    rows = [
        {
            "metric": column.removesuffix("_absolute_delta"),
            "paired_runs": len(frame),
            "mean_absolute_delta": float(frame[column].mean()),
            "max_absolute_delta": float(frame[column].max()),
        }
        for column in delta_columns
    ]
    return _markdown_table(
        pd.DataFrame(rows),
        ("metric", "paired_runs", "mean_absolute_delta", "max_absolute_delta"),
    )


def _markdown_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    selected = [column for column in columns if column in frame.columns]
    if frame.empty or not selected:
        return "No completed records are available."
    lines = [
        "| " + " | ".join(column.replace("_", " ") for column in selected) + " |",
        "|" + "|".join("---" for _ in selected) + "|",
    ]
    for _, row in frame.loc[:, selected].iterrows():
        lines.append(
            "| " + " | ".join(_format_value(row[column]) for column in selected) + " |"
        )
    return "\n".join(lines)


def _format_value(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "-"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4f}"
    return str(value).replace("|", "\\|")


def _atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
