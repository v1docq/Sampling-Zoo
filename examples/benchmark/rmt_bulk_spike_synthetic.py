"""Synthetic identifiability gate for component-aware RMT bulk/spike diagnostics."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm.auto import tqdm


ROOT_DIR = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = Path(__file__).resolve().parent
for module_path in (ROOT_DIR, BENCHMARK_DIR):
    if str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))

from rmt_experiment_utils import markdown_table
from sampling_zoo.core.sampling_strategies.spectral.bulk_spike import (
    build_bulk_spike_training_plan,
)
from sampling_zoo.core.sampling_strategies.spectral.rmt_contraction_sampler import (
    RMTContractionTensorSampler,
)


DEFAULT_BULK_SPIKE_SNRS: tuple[float, ...] = (0.5, 1.0, 2.0)
DEFAULT_BULK_SPIKE_SEEDS: tuple[int, ...] = (11, 29, 47, 71, 101)
DEFAULT_BULK_SPIKE_NOISES: tuple[str, ...] = ("gaussian", "student_t")


@dataclass(frozen=True)
class BulkSpikeSyntheticConfig:
    n_samples: int = 320
    n_features: int = 24
    localized_rank: int = 2
    spike_fraction: float = 0.15
    snr_values: Sequence[float] = DEFAULT_BULK_SPIKE_SNRS
    noise_distributions: Sequence[str] = DEFAULT_BULK_SPIKE_NOISES
    seeds: Sequence[int] = DEFAULT_BULK_SPIKE_SEEDS
    budget_ratio: float = 0.20
    null_resamples: int = 6
    n_views: int = 6
    output_dir: Path | None = None
    show_progress: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "snr_values", tuple(map(float, self.snr_values)))
        object.__setattr__(
            self,
            "noise_distributions",
            tuple(str(value).strip().lower() for value in self.noise_distributions),
        )
        object.__setattr__(self, "seeds", tuple(map(int, self.seeds)))
        if self.output_dir is not None:
            object.__setattr__(self, "output_dir", Path(self.output_dir))
        if int(self.n_samples) < 32 or int(self.n_features) < 4:
            raise ValueError("synthetic matrix must contain at least 32 rows and 4 features")
        if not 1 <= int(self.localized_rank) <= min(self.n_features, 8):
            raise ValueError("localized_rank must be in [1, min(n_features, 8)]")
        if not 0.0 < float(self.spike_fraction) < 0.5:
            raise ValueError("spike_fraction must be in (0, 0.5)")
        if any(value <= 0.0 for value in self.snr_values):
            raise ValueError("snr_values must be positive")
        unknown = set(self.noise_distributions) - {"gaussian", "student_t"}
        if unknown:
            raise ValueError(f"unsupported noise distributions: {sorted(unknown)}")
        if not self.seeds:
            raise ValueError("seeds must not be empty")
        if not 0.0 < float(self.budget_ratio) <= 1.0:
            raise ValueError("budget_ratio must be in (0, 1]")

    @property
    def expected_records(self) -> int:
        return len(self.snr_values) * len(self.noise_distributions) * len(self.seeds)


@dataclass(frozen=True)
class LocalizedSpikeDataset:
    X: np.ndarray
    true_spike_rows: np.ndarray
    empirical_snr: float


def generate_localized_spike_dataset(
    *,
    n_samples: int,
    n_features: int,
    localized_rank: int,
    spike_fraction: float,
    snr: float,
    noise_distribution: str,
    random_state: int,
) -> LocalizedSpikeDataset:
    """Generate a low-rank signal concentrated on a known subset of rows."""

    rng = np.random.default_rng(random_state)
    n_spike_rows = max(localized_rank + 2, int(round(n_samples * spike_fraction)))
    spike_indices = np.sort(rng.choice(n_samples, n_spike_rows, replace=False))
    spike_mask = np.zeros(n_samples, dtype=bool)
    spike_mask[spike_indices] = True
    right_raw = rng.normal(size=(n_features, localized_rank))
    right_basis, _ = np.linalg.qr(right_raw)
    latent = rng.normal(scale=0.04, size=(n_samples, localized_rank))
    latent[spike_mask] += rng.normal(
        loc=2.5,
        scale=0.45,
        size=(n_spike_rows, localized_rank),
    )
    signal = latent @ right_basis[:, :localized_rank].T
    if noise_distribution == "gaussian":
        noise = rng.normal(size=(n_samples, n_features))
    elif noise_distribution == "student_t":
        df = 5.0
        noise = rng.standard_t(df, size=(n_samples, n_features))
        noise /= np.sqrt(df / (df - 2.0))
    else:
        raise ValueError(f"unsupported noise_distribution: {noise_distribution}")
    signal_norm = max(float(np.linalg.norm(signal)), np.finfo(float).eps)
    noise_norm = max(float(np.linalg.norm(noise)), np.finfo(float).eps)
    signal *= np.sqrt(float(snr)) * noise_norm / signal_norm
    empirical_snr = float(np.square(np.linalg.norm(signal) / noise_norm))
    return LocalizedSpikeDataset(
        X=signal + noise,
        true_spike_rows=spike_mask,
        empirical_snr=empirical_snr,
    )


def _record_key(noise: str, snr: float, seed: int) -> str:
    return f"{noise}|snr={snr:.12g}|seed={seed}"


def _evaluate_leaf(
    config: BulkSpikeSyntheticConfig,
    *,
    noise: str,
    snr: float,
    seed: int,
) -> dict[str, Any]:
    dataset = generate_localized_spike_dataset(
        n_samples=config.n_samples,
        n_features=config.n_features,
        localized_rank=config.localized_rank,
        spike_fraction=config.spike_fraction,
        snr=snr,
        noise_distribution=noise,
        random_state=seed,
    )
    sampler = RMTContractionTensorSampler(
        n_partitions=2,
        n_views=config.n_views,
        initial_rank_fraction=0.5,
        null_diagnostic_enabled=True,
        null_model_policies=("feature_permutation", "view_resampling"),
        null_resamples=config.null_resamples,
        null_min_selection_frequency=0.80,
        component_diagnostic_enabled=True,
        embedding_mode="sv_scaled",
        backend="numpy",
        show_progress=False,
        random_state=seed,
    )
    sampler.fit(pd.DataFrame(dataset.X))
    split = sampler.spectral_component_split_
    participation = sampler.row_spectral_participation_
    if participation is None:
        raise RuntimeError("row spectral participation was not materialized")
    signalness = participation.signalness
    truth = dataset.true_spike_rows.astype(int)
    auc = float(roc_auc_score(truth, signalness))
    average_precision = float(average_precision_score(truth, signalness))
    top_count = int(np.sum(truth))
    top_indices = np.argsort(-signalness, kind="mergesort")[:top_count]
    top_precision = float(np.mean(truth[top_indices]))
    budget = max(1, int(round(config.n_samples * config.budget_ratio)))
    training_plan = build_bulk_spike_training_plan(
        participation,
        total_budget=budget,
        policy="excess_energy",
        random_state=seed,
    )
    selected_truth = truth[training_plan.selected_indices]
    selected_prevalence = float(np.mean(selected_truth))
    source_prevalence = float(np.mean(truth))
    enrichment = selected_prevalence / max(source_prevalence, 1e-12)
    return {
        "run_key": _record_key(noise, snr, seed),
        "status": "completed",
        "noise_distribution": noise,
        "snr": float(snr),
        "empirical_snr": dataset.empirical_snr,
        "seed": int(seed),
        "n_samples": int(config.n_samples),
        "n_features": int(config.n_features),
        "true_spike_fraction": source_prevalence,
        "component_status": split.status.value,
        "empirical_bulk_edge": split.empirical_bulk_edge,
        "detected_spike_count": split.n_spikes,
        "signalness_roc_auc": auc,
        "signalness_average_precision": average_precision,
        "top_k_precision": top_precision,
        "selected_spike_prevalence": selected_prevalence,
        "selected_spike_enrichment": enrichment,
        "total_budget": int(budget),
        "selected_unique_rows": int(np.unique(training_plan.selected_indices).size),
        "exact_budget": bool(training_plan.selected_indices.size == budget),
        "resolved_spike_budget_share": training_plan.resolved_spike_share,
        "split": split.to_dict(include_arrays=True),
        "participation": participation.to_dict(),
        "training_plan": training_plan.to_dict(),
    }


def _summarize(raw: pd.DataFrame) -> pd.DataFrame:
    return (
        raw.groupby(["noise_distribution", "snr"], as_index=False)
        .agg(
            completed=("status", lambda values: int(np.sum(values == "completed"))),
            stable_spike_rate=("detected_spike_count", lambda values: float(np.mean(values > 0))),
            median_detected_spikes=("detected_spike_count", "median"),
            median_signalness_auc=("signalness_roc_auc", "median"),
            worst_signalness_auc=("signalness_roc_auc", "min"),
            median_average_precision=("signalness_average_precision", "median"),
            median_top_k_precision=("top_k_precision", "median"),
            median_selected_enrichment=("selected_spike_enrichment", "median"),
            exact_budget_rate=("exact_budget", "mean"),
        )
        .sort_values(["noise_distribution", "snr"])
        .reset_index(drop=True)
    )


def _build_gate(raw: pd.DataFrame, summary: pd.DataFrame, expected: int) -> dict[str, Any]:
    confirmatory = summary[summary["snr"] >= 1.0]
    checks = {
        "all_records_completed": bool(
            len(raw) == expected and (raw["status"] == "completed").all()
        ),
        "exact_budget_everywhere": bool(raw["exact_budget"].all()),
        "stable_spike_rate_at_least_75pct": bool(
            not confirmatory.empty
            and float(confirmatory["stable_spike_rate"].min()) >= 0.75
        ),
        "median_auc_at_least_075": bool(
            not confirmatory.empty
            and float(confirmatory["median_signalness_auc"].min()) >= 0.75
        ),
        "median_enrichment_at_least_15x": bool(
            not confirmatory.empty
            and float(confirmatory["median_selected_enrichment"].min()) >= 1.5
        ),
    }
    return {"passed": bool(all(checks.values())), "checks": checks}


def _plot_summary(summary: pd.DataFrame, output_dir: Path) -> None:
    for metric, ylabel, filename in (
        ("median_signalness_auc", "Median ROC AUC", "signalness_auc_by_snr.png"),
        (
            "median_selected_enrichment",
            "Median spike-row enrichment",
            "spike_enrichment_by_snr.png",
        ),
    ):
        figure, axis = plt.subplots(figsize=(7.2, 4.6))
        for noise, group in summary.groupby("noise_distribution"):
            axis.plot(group["snr"], group[metric], marker="o", label=noise)
        axis.set_xscale("log")
        axis.set_xlabel("SNR")
        axis.set_ylabel(ylabel)
        if metric == "median_signalness_auc":
            axis.set_ylim(0.90, 1.005)
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
        figure.tight_layout()
        figure.savefig(output_dir / filename, dpi=180, bbox_inches="tight")
        plt.close(figure)


def _write_artifacts(
    output_dir: Path,
    config: BulkSpikeSyntheticConfig,
    records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    raw = pd.DataFrame(records)
    summary = _summarize(raw)
    gate = _build_gate(raw, summary, config.expected_records)
    raw.to_csv(output_dir / "bulk_spike_raw_runs.csv", index=False)
    summary.to_csv(output_dir / "bulk_spike_summary.csv", index=False)
    (output_dir / "bulk_spike_gate.json").write_text(
        json.dumps(gate, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _plot_summary(summary, output_dir)
    report = "\n".join(
        [
            "# Синтетическая проверка bulk/spike-диагностики",
            "",
            f"Статус критерия допуска: **{'пройден' if gate['passed'] else 'не пройден'}**.",
            "",
            "## Критерии",
            "",
            markdown_table(
                pd.DataFrame(
                    [
                        {"проверка": key, "результат": value}
                        for key, value in gate["checks"].items()
                    ]
                )
            ),
            "",
            "## Сводка",
            "",
            markdown_table(summary),
            "",
            "`signalness` оценивается относительно заранее известной группы строк, "
            "на которой локализован низкоранговый сигнал. Обогащение показывает, "
            "во сколько раз доля таких строк в точном бюджете выше исходной доли.",
            "",
        ]
    )
    (output_dir / "bulk_spike_report.md").write_text(report, encoding="utf-8")
    return gate


def run_rmt_bulk_spike_synthetic_experiment(
    config: BulkSpikeSyntheticConfig | None = None,
) -> Path:
    resolved = config or BulkSpikeSyntheticConfig()
    output_dir = resolved.output_dir or (
        BENCHMARK_DIR
        / "results"
        / ("run_rmt_bulk_spike_synthetic_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "bulk_spike_runs.jsonl"
    records: list[dict[str, Any]] = []
    if records_path.exists():
        records = [
            json.loads(line)
            for line in records_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    completed = {record["run_key"] for record in records if record.get("status") == "completed"}
    grid = [
        (noise, snr, seed)
        for noise in resolved.noise_distributions
        for snr in resolved.snr_values
        for seed in resolved.seeds
    ]
    meta_path = output_dir / "run_meta.json"
    try:
        for noise, snr, seed in tqdm(
            grid,
            desc="RMT bulk/spike synthetic gate",
            disable=not resolved.show_progress,
            unit="run",
        ):
            key = _record_key(noise, snr, seed)
            if key in completed:
                continue
            record = _evaluate_leaf(resolved, noise=noise, snr=snr, seed=seed)
            with records_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            records.append(record)
            meta_path.write_text(
                json.dumps(
                    {
                        "experiment": "rmt_bulk_spike_synthetic",
                        "status": "running",
                        "record_count": len(records),
                        "expected_records": resolved.expected_records,
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
            "experiment": "rmt_bulk_spike_synthetic",
            "status": "completed" if len(records) == resolved.expected_records else "incomplete",
            "record_count": len(records),
            "expected_records": resolved.expected_records,
            "gate_passed": gate["passed"],
            "config": asdict(resolved) | {"output_dir": str(output_dir)},
        }
    except Exception as exc:
        meta = {
            "experiment": "rmt_bulk_spike_synthetic",
            "status": "failed",
            "record_count": len(records),
            "expected_records": resolved.expected_records,
            "error": f"{type(exc).__name__}: {exc}",
        }
        meta_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        raise
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
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
    config = BulkSpikeSyntheticConfig(
        n_samples=96 if args.smoke else 320,
        n_features=12 if args.smoke else 24,
        snr_values=(1.0,) if args.smoke else DEFAULT_BULK_SPIKE_SNRS,
        noise_distributions=("gaussian",) if args.smoke else DEFAULT_BULK_SPIKE_NOISES,
        seeds=(11,) if args.smoke else DEFAULT_BULK_SPIKE_SEEDS,
        null_resamples=2 if args.smoke else 6,
        n_views=4 if args.smoke else 6,
        output_dir=args.output_dir,
        show_progress=not args.no_progress,
    )
    result = run_rmt_bulk_spike_synthetic_experiment(config)
    print(f"Bulk/spike synthetic artifacts: {result}")


if __name__ == "__main__":
    main()
