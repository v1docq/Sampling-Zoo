from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from examples.benchmark.rmt_bulk_spike_topology_synthetic import (
    BulkSpikeTopologySyntheticConfig,
    generate_topology_dataset,
    run_rmt_bulk_spike_topology_synthetic,
)


def test_full_synthetic_topology_grid_counts_all_seven_regimes() -> None:
    config = BulkSpikeTopologySyntheticConfig(show_progress=False)

    assert len(config.regimes) == 7
    assert config.expected_records == 1260


def test_rare_class_and_false_outlier_regimes_encode_distinct_hypotheses() -> None:
    config = BulkSpikeTopologySyntheticConfig(n_samples=300, show_progress=False)
    rare = generate_topology_dataset(config, regime="rare_class_spike", snr=1.0, seed=11)
    false_outlier = generate_topology_dataset(
        config,
        regime="false_spectral_outlier",
        snr=1.0,
        seed=11,
    )

    assert rare.problem_type == "classification"
    assert rare.classes == (0, 1, 2)
    assert set(np.unique(rare.y)) == {0, 1, 2}
    assert np.all(rare.y[rare.true_spike_rows] == 2)
    assert false_outlier.problem_type == "regression"
    assert np.any(false_outlier.true_spike_rows)


def test_tiny_topology_smoke_persists_exact_budget_artifacts(tmp_path: Path) -> None:
    output_dir = tmp_path / "topology_smoke"
    config = BulkSpikeTopologySyntheticConfig(
        n_samples=240,
        regimes=("null", "rare_class_spike"),
        snr_values=(1.0,),
        budget_ratios=(0.10,),
        seeds=(11,),
        null_resamples=3,
        output_dir=output_dir,
        show_progress=False,
    )

    result = run_rmt_bulk_spike_topology_synthetic(config)
    metadata = json.loads((result / "run_meta.json").read_text(encoding="utf-8"))

    assert metadata["status"] == "completed"
    assert metadata["record_count"] == 8
    assert (result / "topology_raw_runs.csv").exists()
    assert (result / "topology_paired.csv").exists()
    assert (result / "topology_gate.json").exists()
    assert (result / "topology_report.md").exists()
