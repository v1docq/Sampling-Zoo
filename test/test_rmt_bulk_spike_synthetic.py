from __future__ import annotations

import json

import pandas as pd

from examples.benchmark.rmt_bulk_spike_synthetic import (
    BulkSpikeSyntheticConfig,
    run_rmt_bulk_spike_synthetic_experiment,
)


def test_tiny_bulk_spike_experiment_persists_incremental_artifacts(tmp_path) -> None:
    output = run_rmt_bulk_spike_synthetic_experiment(
        BulkSpikeSyntheticConfig(
            n_samples=96,
            n_features=12,
            localized_rank=1,
            snr_values=(1.0,),
            noise_distributions=("gaussian",),
            seeds=(7,),
            null_resamples=2,
            n_views=4,
            output_dir=tmp_path,
            show_progress=False,
        )
    )

    meta = json.loads((output / "run_meta.json").read_text(encoding="utf-8"))
    raw = pd.read_csv(output / "bulk_spike_raw_runs.csv")
    assert meta["status"] == "completed"
    assert meta["record_count"] == 1
    assert len(raw) == 1
    assert bool(raw.loc[0, "exact_budget"])
    assert (output / "bulk_spike_summary.csv").exists()
    assert (output / "bulk_spike_gate.json").exists()
    assert (output / "bulk_spike_report.md").exists()
    assert (output / "signalness_auc_by_snr.png").exists()
