"""UncertaintySampler с единым выводом статистик."""

import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from core.api.api_main import SamplingStrategyFactory
from core.sampling_strategies.base_sampler import HierarchicalStratifiedMixin
from core.utils.synt_data import create_noisy_dataset

DATASET_SAMPLES = 10_000


def run_uncertainty_sampler() -> None:
    data = create_noisy_dataset(DATASET_SAMPLES)
    features = data[["feature_1", "feature_2"]]

    factory = SamplingStrategyFactory()
    strategy = factory.create_and_fit(
        "uncertainty",
        data=features,
        target=data["target"],
        strategy_kwargs={"n_partitions": 3, "random_state": 42},
    )
    partitions = strategy.get_partitions(features, target=data["target"])

    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(features, data["target"])
    predictions = model.predict(features)

    indices = {name: part["feature"].index.to_numpy() for name, part in partitions.items()}
    HierarchicalStratifiedMixin.print_fold_summary("UncertaintySampler", indices, data["target"])

    uncertainty_scores = strategy.get_uncertainty_scores()
    for name, part in partitions.items():
        idx = part["feature"].index.to_numpy()
        if idx.size == 0:
            print(f"{name}: 0 samples (skipped)")
            continue

        y_true = np.asarray(part["target"])
        y_pred = predictions[idx]
        errors = (y_true != y_pred).sum()
        cm = confusion_matrix(y_true, y_pred)
        avg_unc = float(np.mean(uncertainty_scores[idx]))

        print(f"{name}: {errors} errors out of {len(idx)} ({errors / len(idx):.2%}), avg uncertainty={avg_unc:.4f}")
        print("Confusion matrix for", name)
        print(cm)


if __name__ == "__main__":
    run_uncertainty_sampler()
