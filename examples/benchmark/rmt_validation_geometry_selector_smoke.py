"""Executable synthetic smoke for the Phase A.1 validation geometry selector."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = Path(__file__).resolve().parent
for module_path in (ROOT_DIR, BENCHMARK_DIR):
    if str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))

from benchmark_dataset_interfaces import make_synthetic_regression_smoke_dataset
from rmt_validation_geometry_selector_experiment import (
    RoutingGeometrySelectorExperimentConfig,
    ValidationRoutingGeometryExperimentOrchestrator,
)


def run_validation_geometry_selector_smoke(
    *,
    output_dir: str | Path,
    show_progress: bool = True,
) -> pd.DataFrame:
    config = RoutingGeometrySelectorExperimentConfig(
        regression_suite=None,
        classification_suite=None,
        regression_tasks=(),
        classification_tasks=(),
        models=("ridge",),
        budget_ratios=(0.20,),
        seeds=(42,),
        n_partitions=3,
        output_dir=Path(output_dir),
        show_progress=show_progress,
    )

    class _SyntheticOrchestrator(ValidationRoutingGeometryExperimentOrchestrator):
        def _load_datasets(self):
            return [make_synthetic_regression_smoke_dataset(42)]

    result = _SyntheticOrchestrator(config).run()
    if result.shape[0] != 4 or set(result["status"]) != {"completed"}:
        raise RuntimeError("Validation geometry selector smoke did not complete four arms")
    selector = result.loc[result["arm_name"] == config.selector_arm_name]
    if selector.shape[0] != 1 or not selector["selected_arm_name"].iloc[0]:
        raise RuntimeError("Validation geometry selector smoke produced no decision")
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tmp/routing_geometry_selector_smoke"),
    )
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    frame = run_validation_geometry_selector_smoke(
        output_dir=args.output_dir,
        show_progress=not args.no_progress,
    )
    summary_columns = [
        "dataset",
        "arm_name",
        "selected_arm_name",
        "status",
        "test_primary_metric",
        "test_primary_value",
    ]
    print(frame[summary_columns].to_string(index=False))
