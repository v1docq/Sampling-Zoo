from __future__ import annotations

import json

import pandas as pd

from examples.benchmark.rmt_classification_geometry_guard_analysis import (
    CLASSIFICATION_REFERENCE_ARM,
    ClassificationGeometryGateAnalyzer,
    ClassificationGeometryGateSpec,
)
from examples.benchmark.rmt_cross_fitted_geometry_selector_experiment import (
    DEFAULT_CLASSIFICATION_GUARDED_SELECTOR_ARM_NAME,
)


def test_classification_geometry_gate_separates_safety_and_effectiveness(
    tmp_path,
) -> None:
    rows = []
    records = []
    for dataset, metric, reference, selected, secondary in (
        ("binary", "roc_auc", 0.80, 0.801, 0.45),
        ("multiclass", "log_loss", 0.50, 0.495, 0.495),
    ):
        for arm_name, value, selected_name in (
            (CLASSIFICATION_REFERENCE_ARM, reference, CLASSIFICATION_REFERENCE_ARM),
            (
                DEFAULT_CLASSIFICATION_GUARDED_SELECTOR_ARM_NAME,
                selected,
                "A5_gmm_posterior",
            ),
        ):
            rows.append(
                {
                    "dataset": dataset,
                    "seed": 42,
                    "budget_ratio": 0.1,
                    "model": "lightgbm",
                    "arm_name": arm_name,
                    "selected_arm_name": selected_name,
                    "status": "completed",
                    "test_primary_metric": metric,
                    "test_primary_value": value,
                    "test_roc_auc": value if metric == "roc_auc" else None,
                    "test_log_loss": secondary,
                }
            )
            records.append(
                {
                    "status": "completed",
                    "sampler_diagnostics": {
                        "class_coverage_guaranteed": True,
                    },
                }
            )
    pd.DataFrame(rows).to_csv(
        tmp_path / "routing_geometry_replay.csv",
        index=False,
    )
    with (tmp_path / "routing_geometry_runs.jsonl").open(
        "w",
        encoding="utf-8",
    ) as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    gate = ClassificationGeometryGateAnalyzer(
        ClassificationGeometryGateSpec(
            expected_records=4,
            expected_selector_records=2,
        )
    ).build(tmp_path)

    assert gate["status"] == "passed"
    assert gate["evidence_status"] == "safe_and_effective"
    assert gate["selected_nonbaseline_count"] == 2
    assert (tmp_path / "classification_geometry_guard_paired.csv").exists()
    assert (tmp_path / "classification_geometry_gate_report.md").exists()


def test_classification_geometry_gate_fails_without_class_coverage(
    tmp_path,
) -> None:
    rows = pd.DataFrame(
        [
            {
                "dataset": "binary",
                "seed": 42,
                "budget_ratio": 0.1,
                "model": "lightgbm",
                "arm_name": arm,
                "selected_arm_name": CLASSIFICATION_REFERENCE_ARM,
                "status": "completed",
                "test_primary_metric": "roc_auc",
                "test_primary_value": 0.8,
                "test_roc_auc": 0.8,
                "test_log_loss": 0.5,
            }
            for arm in (
                CLASSIFICATION_REFERENCE_ARM,
                DEFAULT_CLASSIFICATION_GUARDED_SELECTOR_ARM_NAME,
            )
        ]
    )
    rows.to_csv(tmp_path / "routing_geometry_replay.csv", index=False)
    with (tmp_path / "routing_geometry_runs.jsonl").open(
        "w",
        encoding="utf-8",
    ) as handle:
        for _ in range(2):
            handle.write(
                json.dumps(
                    {
                        "status": "completed",
                        "sampler_diagnostics": {
                            "class_coverage_guaranteed": False,
                        },
                    }
                )
                + "\n"
            )

    gate = ClassificationGeometryGateAnalyzer(
        ClassificationGeometryGateSpec(
            expected_records=2,
            expected_selector_records=1,
        )
    ).build(tmp_path)

    assert gate["status"] == "failed"
    assert gate["checks"]["class_coverage_guaranteed"] is False
