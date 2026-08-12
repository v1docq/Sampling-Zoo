from __future__ import annotations

import json

import pandas as pd
from sklearn.datasets import make_classification

from examples.benchmark.benchmark_dataset_interfaces import (
    make_synthetic_regression_smoke_dataset,
)
from examples.benchmark.benchmark_datasets import (
    RawDatasetBundle,
    RawDatasetMetadata,
)
from examples.benchmark.rmt_bulk_spike_topology_real_experiment import (
    BulkSpikeRealExperimentConfig,
    BulkSpikeRealExperimentOrchestrator,
)


def test_real_topology_runner_selects_away_from_test_and_resumes(tmp_path) -> None:
    dataset = make_synthetic_regression_smoke_dataset(42)
    config = BulkSpikeRealExperimentConfig(
        regression_suite=None,
        classification_suite=None,
        regression_tasks=(dataset.name,),
        classification_tasks=(),
        models=("ridge",),
        budget_ratios=(0.20,),
        seeds=(42,),
        n_partitions=3,
        selection_folds=3,
        null_resamples=2,
        topology_min_partition_size=2,
        sampler_extra_params={
            "backend": "numpy",
            "n_views": 3,
            "projection_dim": 3,
            "min_sampled_rows_per_partition": 2,
        },
        output_dir=tmp_path,
        show_progress=False,
    )

    result = BulkSpikeRealExperimentOrchestrator(
        config,
        datasets=(dataset,),
    ).run()

    assert result.shape[0] == 4
    assert set(result["status"]) == {"completed"}
    assert set(result["arm_name"]) == {
        "B0_standard_A9",
        "B1_bulk_single_spike",
        "B2_bulk_multi_spike",
        "B3_validation_selected",
    }
    selected = result[result["arm_name"] == "B3_validation_selected"].iloc[0]
    assert selected["selected_arm_name"] in {
        "B0_standard_A9",
        "B1_bulk_single_spike",
        "B2_bulk_multi_spike",
    }
    assert selected["n_calibration"] > 0
    assert selected["n_selection"] > 0
    assert selected["test_primary_metric"] == "rmse"
    assert selected["full_reference_primary_metric"] == "rmse"
    assert pd.notna(selected["degradation_vs_full"])
    assert result.loc[
        result["arm_name"].isin(
            ["B1_bulk_single_spike", "B2_bulk_multi_spike"]
        ),
        "topology_exact_budget",
    ].all()

    resumed = BulkSpikeRealExperimentOrchestrator(
        config,
        datasets=(dataset,),
    ).run()
    lines = (tmp_path / "routing_geometry_runs.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    metadata = json.loads((tmp_path / "run_meta.json").read_text(encoding="utf-8"))

    assert resumed.shape[0] == 4
    assert len(lines) == 4
    assert metadata["status"] == "completed"
    assert metadata["full_reference_record_count"] == 1
    assert (tmp_path / "bulk_spike_full_references.jsonl").exists()
    assert (tmp_path / "bulk_spike_full_references.json").exists()
    assert (tmp_path / "bulk_spike_paired.csv").exists()
    assert (tmp_path / "bulk_spike_summary.csv").exists()
    assert (tmp_path / "bulk_spike_gate.json").exists()
    assert (tmp_path / "bulk_spike_report.md").exists()
    assert (tmp_path / "bulk_spike_gain_by_budget.png").exists()


def test_real_topology_runner_preserves_multiclass_probability_contract(tmp_path) -> None:
    features, target = make_classification(
        n_samples=600,
        n_features=10,
        n_informative=7,
        n_redundant=1,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=67,
    )
    frame = pd.DataFrame(features, columns=[f"x{index}" for index in range(10)])
    labels = pd.Series(target, name="target")
    dataset = RawDatasetBundle(
        name="synthetic_multiclass_topology",
        problem_type="classification",
        target_name="target",
        source_path="memory://synthetic_multiclass_topology",
        X=frame,
        y=labels,
        metadata=RawDatasetMetadata(
            n_objects=len(frame),
            n_features=frame.shape[1],
            n_train_candidates=len(frame),
            n_categorical=0,
            n_numeric=frame.shape[1],
        ),
        feature_columns=list(frame.columns),
        categorical_columns=[],
        numeric_columns=list(frame.columns),
    )
    config = BulkSpikeRealExperimentConfig(
        regression_suite=None,
        classification_suite=None,
        regression_tasks=(),
        classification_tasks=(dataset.name,),
        models=("ridge",),
        budget_ratios=(0.20,),
        seeds=(67,),
        n_partitions=3,
        selection_folds=3,
        null_resamples=2,
        topology_min_partition_size=3,
        sampler_extra_params={
            "backend": "numpy",
            "n_views": 3,
            "projection_dim": 3,
            "min_sampled_rows_per_partition": 3,
        },
        output_dir=tmp_path,
        show_progress=False,
    )

    result = BulkSpikeRealExperimentOrchestrator(
        config,
        datasets=(dataset,),
    ).run()

    assert result.shape[0] == 4
    assert set(result["status"]) == {"completed"}
    assert result["test_primary_metric"].eq("log_loss").all()
    assert result["full_reference_primary_metric"].eq("log_loss").all()
    assert result["degradation_vs_full"].notna().all()
    assert result["test_brier_score"].notna().all()
    assert result["test_expected_calibration_error"].notna().all()
    specialized = result[
        result["arm_name"].isin(
            ["B1_bulk_single_spike", "B2_bulk_multi_spike"]
        )
    ]
    assert specialized["topology_exact_budget"].all()
    assert specialized["class_coverage_guaranteed"].all()
