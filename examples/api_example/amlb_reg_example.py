"""
Запуск LargeScaleAutoMLExperiment через текстовый конфиг (регрессия).

Доступные модели:
- lgbm: LightGBM (параметры: n_estimators, learning_rate, n_jobs и др.)
- random_forest: Random Forest (параметры: n_estimators, max_depth, n_jobs и др.)

Пример использования разных моделей:
- models: lgbm(n_estimators=100, learning_rate=0.1, n_jobs=-1)
- models: random_forest(n_estimators=200, max_depth=20, n_jobs=-1)
"""

from __future__ import annotations
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from core.utils.amlb_config import ExperimentConfigBuilder
from core.utils.amlb_setup import ExperimentConfig, LargeScaleAutoMLExperiment
from core.repository.constant_repo import AmlbExperimentDataset


EXPERIMENT_REQUEST = """
datasets: year_prediction_msd
cv_folds: 3
run_mode: mixed_chunk
sampling: feature_clustering(n_partitions=8, chunks_percent=70, save_filename=partitions_year)
models: lgbm(n_estimators=100, learning_rate=0.1, n_jobs=-1)
time_budget: 1500
tracking_uri: file:./mlruns
"""


def run_from_request(request: str) -> None:
    builder = ExperimentConfigBuilder(default_time_budget=1500)
    model_config = {'metric': 'rmse'}
    experiment_config: ExperimentConfig = builder.from_text(
        request,
        model_config=model_config
    )

    print("Конфигурация эксперимента:")
    for key, value in experiment_config.to_dict().items():
        print(f"  {key}: {value}")

    experiment = LargeScaleAutoMLExperiment(experiment_config=experiment_config)
    experiment.run_full_benchmark()


if __name__ == "__main__":
    run_from_request(EXPERIMENT_REQUEST)
