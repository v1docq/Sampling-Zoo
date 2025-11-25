"""Запуск LargeScaleAutoMLExperiment через текстовый конфиг."""

from __future__ import annotations
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from core.utils.amlb_config import ExperimentConfigBuilder
from core.utils.amlb_setup import ExperimentConfig, LargeScaleAutoMLExperiment
from core.repository.constant_repo import AmlbExperimentDataset


EXPERIMENT_REQUEST = """
datasets: kddcup
sampling: hierarchical_stratified(n_partitions=50)
models: fedot(preset=best_quality)
time_budget: 15
tracking_uri: file:./mlruns
"""


def run_from_request(request: str) -> None:
    builder = ExperimentConfigBuilder(default_time_budget=10)
    experiment_config: ExperimentConfig = builder.from_text(
        request,
        fedot_config=AmlbExperimentDataset.FEDOT_PRESET_CLF.value
    )

    print("Конфигурация эксперимента:")
    for key, value in experiment_config.to_dict().items():
        print(f"  {key}: {value}")

    experiment = LargeScaleAutoMLExperiment(experiment_config=experiment_config)
    experiment.run_full_benchmark()


if __name__ == "__main__":
    run_from_request(EXPERIMENT_REQUEST)
