from __future__ import annotations

import json

import pandas as pd
import pytest
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
    PRACTICAL_EXPLORATION_GATE_PROFILE,
    STRICT_RESEARCH_GATE_PROFILE,
    get_bulk_spike_gate_profile,
)


def test_bulk_spike_gate_profiles_keep_strict_and_practical_tolerances_explicit() -> None:
    strict = get_bulk_spike_gate_profile(STRICT_RESEARCH_GATE_PROFILE)
    practical = get_bulk_spike_gate_profile(PRACTICAL_EXPLORATION_GATE_PROFILE)

    assert strict.primary_harm_margin == 0.005
    assert strict.tail_harm_margin == 0.01
    assert strict.ece_harm_margin == 0.01
    assert practical.primary_harm_margin == 0.01
    assert practical.tail_harm_margin == 0.015
    assert practical.ece_harm_margin == 0.02
    assert practical.brier_harm_margin == strict.brier_harm_margin
    with pytest.raises(ValueError, match="Unknown bulk/spike gate profile"):
        get_bulk_spike_gate_profile("not-a-profile")


def test_practical_gate_accepts_predeclared_small_harms_only() -> None:
    seeds = (1, 2, 3, 4, 5)
    rows = []
    for seed in seeds:
        is_harmful_holdout = seed == 5
        b3_log_loss = 1.007 if is_harmful_holdout else 0.99
        b3_ece = 0.115 if is_harmful_holdout else 0.10
        common = {
            "dataset": "classification_fixture",
            "problem_type": "classification",
            "seed": seed,
            "budget_ratio": 0.10,
            "model": "ridge",
            "status": "completed",
            "topology_exact_budget": True,
            "topology_unique_rows": 10,
            "topology_total_budget": 10,
            "test_primary_metric": "log_loss",
            "test_brier_score": 0.20,
            "test_f1_macro": 0.80,
            "test_worst_class_recall": 0.70,
            "runtime_fit_seconds": 1.0,
            "runtime_inference_seconds": 1.0,
        }
        rows.extend(
            [
                {
                    **common,
                    "arm_name": "B0_standard_A9",
                    "selected_arm_name": "B0_standard_A9",
                    "test_primary_value": 1.0,
                    "test_expected_calibration_error": 0.10,
                },
                {
                    **common,
                    "arm_name": "B1_bulk_single_spike",
                    "selected_arm_name": "B1_bulk_single_spike",
                    "test_primary_value": b3_log_loss,
                    "test_expected_calibration_error": b3_ece,
                },
                {
                    **common,
                    "arm_name": "B2_bulk_multi_spike",
                    "selected_arm_name": "B2_bulk_multi_spike",
                    "test_primary_value": 1.0,
                    "test_expected_calibration_error": 0.10,
                },
                {
                    **common,
                    "arm_name": "B3_validation_selected",
                    "selected_arm_name": "B1_bulk_single_spike",
                    "test_primary_value": b3_log_loss,
                    "test_expected_calibration_error": b3_ece,
                },
            ]
        )
    result = pd.DataFrame(rows)

    def evaluate(profile: str) -> dict:
        config = BulkSpikeRealExperimentConfig(
            regression_suite=None,
            classification_suite=None,
            regression_tasks=(),
            classification_tasks=("classification_fixture",),
            models=("ridge",),
            budget_ratios=(0.10,),
            seeds=seeds,
            gate_profile=profile,
            show_progress=False,
        )
        orchestrator = BulkSpikeRealExperimentOrchestrator(config)
        orchestrator.reference_records_ = {
            ("classification_fixture", seed, "ridge"): {"status": "completed"}
            for seed in seeds
        }
        return orchestrator._gate(result, orchestrator._paired_results(result))

    strict = evaluate(STRICT_RESEARCH_GATE_PROFILE)
    practical = evaluate(PRACTICAL_EXPLORATION_GATE_PROFILE)

    assert strict["status"] == "failed"
    assert not strict["checks"]["worst_harm_within_margin"]
    assert not strict["checks"]["classification_probability_and_balance_noninferiority"]
    assert practical["status"] == "passed"
    assert practical["checks"]["worst_harm_within_margin"]
    assert practical["checks"]["classification_probability_and_balance_noninferiority"]


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
    reference = json.loads(
        (tmp_path / "bulk_spike_full_references.jsonl").read_text(
            encoding="utf-8"
        )
    )

    assert resumed.shape[0] == 4
    assert len(lines) == 4
    assert metadata["status"] == "completed"
    assert metadata["full_reference_record_count"] == 1
    assert reference["n_train"] > int(selected["n_train"])
    assert reference["n_train"] + reference["n_test"] == len(dataset.X)
    assert (tmp_path / "bulk_spike_full_references.jsonl").exists()
    assert (tmp_path / "bulk_spike_full_references.json").exists()
    assert (tmp_path / "bulk_spike_paired.csv").exists()
    assert (tmp_path / "bulk_spike_summary.csv").exists()
    allocation = pd.read_csv(tmp_path / "bulk_spike_b1_allocation.csv")
    assert set(
        [
            "bulk_train_share",
            "spike_train_share",
            "bulk_routing_hard_share",
            "spike_routing_hard_share",
            "selected_by_b3",
        ]
    ).issubset(allocation.columns)
    assert allocation["bulk_train_share"].between(0.0, 1.0).all()
    assert allocation["spike_train_share"].between(0.0, 1.0).all()
    assert (tmp_path / "bulk_spike_gate.json").exists()
    gate = json.loads((tmp_path / "bulk_spike_gate.json").read_text(encoding="utf-8"))
    assert gate["gate_profile"]["name"] == STRICT_RESEARCH_GATE_PROFILE
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


def test_real_topology_public_runner_preserves_explicit_empty_task_group(
    monkeypatch,
    tmp_path,
) -> None:
    captured = {}

    def fake_run(self):
        captured["regression_tasks"] = tuple(self.config.regression_tasks)
        captured["classification_tasks"] = tuple(
            self.config.classification_tasks
        )
        return pd.DataFrame()

    monkeypatch.setattr(BulkSpikeRealExperimentOrchestrator, "run", fake_run)
    from examples.benchmark.rmt_bulk_spike_topology_real_experiment import (
        run_rmt_bulk_spike_topology_real_experiment,
    )

    run_rmt_bulk_spike_topology_real_experiment(
        regression_tasks=(),
        classification_tasks=("adult",),
        output_dir=tmp_path,
        show_progress=False,
    )

    assert captured == {
        "regression_tasks": (),
        "classification_tasks": ("adult",),
    }
