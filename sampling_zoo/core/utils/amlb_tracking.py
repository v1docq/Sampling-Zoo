"""Легковесный трекер экспериментов с поддержкой MLflow."""

from __future__ import annotations

import time
from typing import Dict, Optional

try:
    import mlflow
except ImportError:  # pragma: no cover - опциональная зависимость для примеров
    mlflow = None


class ExperimentTracker:
    def __init__(self, experiment_name: str, tracking_uri: Optional[str] = None):
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.current_run = None

        if mlflow is not None:
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)

    def start_run(self, run_name: str, params: Dict | None = None):
        params = params or {}
        if mlflow is None:
            self.current_run = {"run_id": str(time.time()), "params": params}
            return self.current_run

        self.current_run = mlflow.start_run(run_name=run_name)
        mlflow.log_params(params)
        return self.current_run

    def log_metrics(self, metrics: Dict[str, float]):
        if mlflow is None or self.current_run is None:
            return
        mlflow.log_metrics(metrics)

    def end_run(self):
        if mlflow is None or self.current_run is None:
            return
        mlflow.end_run()
        self.current_run = None

    @staticmethod
    def version_label(run_obj) -> str:
        if mlflow is None or run_obj is None:
            return f"local-{int(time.time())}"
        return run_obj.info.run_id
