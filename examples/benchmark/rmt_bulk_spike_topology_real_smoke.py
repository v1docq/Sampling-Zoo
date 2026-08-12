"""Run a reproducible regression and multiclass bulk/spike smoke experiment."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
from sklearn.datasets import make_classification

ROOT_DIR = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = Path(__file__).resolve().parent
for module_path in (ROOT_DIR, BENCHMARK_DIR):
    if str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))

from benchmark_dataset_interfaces import make_synthetic_regression_smoke_dataset
from benchmark_datasets import RawDatasetBundle, RawDatasetMetadata
from rmt_bulk_spike_topology_real_experiment import (
    BulkSpikeRealExperimentConfig,
    BulkSpikeRealExperimentOrchestrator,
)


def make_synthetic_multiclass_topology_dataset(seed: int) -> RawDatasetBundle:
    features, target = make_classification(
        n_samples=600,
        n_features=10,
        n_informative=7,
        n_redundant=1,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=int(seed),
    )
    frame = pd.DataFrame(
        features,
        columns=[f"x{index}" for index in range(features.shape[1])],
    )
    labels = pd.Series(target, name="target")
    return RawDatasetBundle(
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


def run_smoke(output_dir: Path, *, show_progress: bool = True) -> pd.DataFrame:
    regression = make_synthetic_regression_smoke_dataset(42)
    classification = make_synthetic_multiclass_topology_dataset(42)
    config = BulkSpikeRealExperimentConfig(
        regression_suite=None,
        classification_suite=None,
        regression_tasks=(regression.name,),
        classification_tasks=(classification.name,),
        models=("ridge",),
        budget_ratios=(0.20,),
        seeds=(42,),
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
        output_dir=Path(output_dir),
        show_progress=show_progress,
    )
    return BulkSpikeRealExperimentOrchestrator(
        config,
        datasets=(regression, classification),
    ).run()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    result = run_smoke(args.output_dir, show_progress=not args.no_progress)
    print(result[["dataset", "arm_name", "status", "test_primary_value"]])


if __name__ == "__main__":
    main()
