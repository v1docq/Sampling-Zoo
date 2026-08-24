from __future__ import annotations

import json

import numpy as np
import pandas as pd

from examples.benchmark.rmt_leverage_sketch_synthetic import (
    DEFAULT_ARMS,
    LeverageSketchSyntheticConfig,
    build_phase_s_grid,
    generate_synthetic_sketch_dataset,
    run_rmt_leverage_sketch_synthetic,
)


def test_phase_s_grid_is_deterministic_and_complete() -> None:
    config = LeverageSketchSyntheticConfig(
        scenarios=("low_rank_regression",),
        budget_ratios=(0.1, 0.2),
        seeds=(1, 2),
        show_progress=False,
    )

    first = build_phase_s_grid(config)
    second = build_phase_s_grid(config)

    assert first == second
    assert len(first) == config.expected_records == 4 * len(DEFAULT_ARMS)


def test_rare_class_generator_is_reproducible_and_binary() -> None:
    kwargs = dict(
        scenario="rare_class_classification",
        n_train=200,
        n_test=100,
        n_features=16,
        latent_rank=4,
        random_state=8,
    )
    first = generate_synthetic_sketch_dataset(**kwargs)
    second = generate_synthetic_sketch_dataset(**kwargs)

    assert np.array_equal(first.X_train, second.X_train)
    assert np.array_equal(first.y_train, second.y_train)
    assert set(np.unique(first.y_train)) == {0, 1}
    assert first.problem == "classification"


def test_phase_s_smoke_writes_incremental_and_final_artifacts(tmp_path) -> None:
    config = LeverageSketchSyntheticConfig(
        scenarios=("low_rank_regression", "rare_class_classification"),
        budget_ratios=(0.1,),
        seeds=(3,),
        n_train=160,
        n_test=100,
        n_features=12,
        latent_rank=3,
        output_root=tmp_path,
        show_progress=False,
        snapshot_every=2,
    )

    output_dir = run_rmt_leverage_sketch_synthetic(config)
    meta = json.loads((output_dir / "run_meta.json").read_text(encoding="utf-8"))
    raw = pd.read_csv(output_dir / "phase_s_raw_runs.csv")

    assert meta["status"] == "completed"
    assert meta["record_count"] == config.expected_records
    assert len(raw) == config.expected_records
    assert raw["exact_budget"].all()
    assert (raw["selected_size"] == raw["unique_selected_size"]).all()
    assert (output_dir / "phase_s_report.md").exists()
    assert (output_dir / "phase_s_gate.json").exists()
