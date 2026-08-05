"""Executable synthetic smoke for Phase A routing geometry replay."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from sampling_zoo.core.utils.sampling_ensemble import SamplingEnsemble

from routing_geometry_replay import (
    FittedEnsembleRoutingReplay,
    RoutingGeometryArm,
    default_routing_geometry_arms,
)


def run_routing_geometry_synthetic_smoke(
    *,
    problem_types: Sequence[str] = ("regression", "classification"),
    arms: Sequence[RoutingGeometryArm] | None = None,
    n_samples: int = 480,
    seed: int = 42,
    n_estimators: int = 24,
    output_dir: str | Path | None = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Fit chunk experts once per problem and replay the agreed routing arms."""

    records: list[dict[str, Any]] = []
    for problem_type in problem_types:
        X, y = _make_synthetic_problem(problem_type, n_samples=n_samples, seed=seed)
        stratify = y if problem_type == "classification" else None
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=0.30,
            random_state=seed,
            stratify=stratify,
        )
        X_train = X_train.reset_index(drop=True)
        X_val = X_val.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        y_val = y_val.reset_index(drop=True)

        ensemble = SamplingEnsemble(
            problem=problem_type,
            partitioner_config=_rmt_smoke_config(seed, show_progress),
            model_factory=_model_factory(problem_type, seed, n_estimators),
            ensemble_method="routed_weighted",
            show_progress=show_progress,
        )
        ensemble.train_partition_models(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            class_samples=(
                _class_representatives(X_train, y_train)
                if problem_type == "classification"
                else None
            ),
            cv_fold=1,
            validation_metric=(
                "log_loss" if problem_type == "classification" else "rmse"
            ),
            train_all_chunks=True,
            save_models_to_disk=False,
        )
        replay_results = FittedEnsembleRoutingReplay(
            show_progress=show_progress,
        ).run_validation(
            ensemble=ensemble,
            X_val=X_val,
            y_val=y_val,
            arms=arms or default_routing_geometry_arms(),
            metadata={"problem_type": problem_type, "seed": seed},
        )
        records.extend(
            FittedEnsembleRoutingReplay.records(
                replay_results,
                metadata={
                    "problem_type": problem_type,
                    "seed": seed,
                    "n_train": len(X_train),
                    "n_validation": len(X_val),
                    "n_experts": len(ensemble.models),
                },
            )
        )

    frame = _flatten_records(records)
    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        frame.to_csv(destination / "routing_geometry_replay.csv", index=False)
    return frame


def _rmt_smoke_config(seed: int, show_progress: bool) -> dict[str, Any]:
    return {
        "strategy": "rmt_contraction",
        "n_partitions": 3,
        "partition_selection_method": "fixed",
        "n_views": 4,
        "projection_dim": 4,
        "embedding_mode": "sv_scaled",
        "selection_method": "all",
        "routing_representation": "source_centroid",
        "routing_metric": "squared_euclidean",
        "routing_kernel": "softmax",
        "routing_temperature": 1.0,
        "routing_shrinkage": 0.05,
        "routing_covariance_shrinkage": 0.10,
        "backend": "numpy",
        "random_state": seed,
        "show_progress": show_progress,
        "router": "spectral",
    }


def _model_factory(problem_type: str, seed: int, n_estimators: int):
    if problem_type == "classification":
        return lambda: RandomForestClassifier(
            n_estimators=n_estimators,
            min_samples_leaf=3,
            random_state=seed,
            n_jobs=1,
        )
    return lambda: RandomForestRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=3,
        random_state=seed,
        n_jobs=1,
    )


def _make_synthetic_problem(
    problem_type: str,
    *,
    n_samples: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    regime = rng.integers(0, 3, size=n_samples)
    X = rng.normal(size=(n_samples, 8))
    X[:, 0] = regime * 3.0 + rng.normal(scale=0.35, size=n_samples)
    X[:, 1] *= np.where(regime == 0, 3.0, 0.4)
    X[:, 2] *= np.where(regime == 2, 2.5, 0.5)
    frame = pd.DataFrame(X, columns=[f"x_{idx}" for idx in range(X.shape[1])])
    if problem_type == "regression":
        target = (
            np.where(regime == 0, 2.0 * X[:, 1], 0.0)
            + np.where(regime == 1, -3.0 * X[:, 3], 0.0)
            + np.where(regime == 2, 1.5 * X[:, 2] + X[:, 4], 0.0)
            + rng.normal(scale=0.15, size=n_samples)
        )
        return frame, pd.Series(target, name="target")
    if problem_type == "classification":
        logits = np.column_stack(
            [
                1.2 * X[:, 1] - 0.3 * X[:, 4],
                -1.0 * X[:, 3] + 0.4 * X[:, 5],
                0.8 * X[:, 2] + 0.5 * X[:, 6],
            ]
        )
        logits[np.arange(n_samples), regime] += 1.5
        target = np.argmax(logits + rng.normal(scale=0.35, size=logits.shape), axis=1)
        return frame, pd.Series(target, name="target")
    raise ValueError("problem_type must be regression or classification")


def _class_representatives(
    features: pd.DataFrame,
    target: pd.Series,
) -> dict[Any, tuple[pd.Series, Any]]:
    representatives = {}
    values = target.to_numpy()
    for label in np.unique(values):
        index = int(np.flatnonzero(values == label)[0])
        representatives[label] = (features.iloc[index], target.iloc[index])
    return representatives


def _flatten_records(records: Iterable[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for record in records:
        rows.append(
            {
                "problem_type": record["problem_type"],
                "seed": record["seed"],
                "n_train": record["n_train"],
                "n_validation": record["n_validation"],
                "n_experts": record["n_experts"],
                "arm_name": record["arm_name"],
                "temperature": record["temperature"],
                "primary_metric": record["primary_metric"],
                "primary_value": record["primary_value"],
                **{
                    f"metric_{name}": value
                    for name, value in record["metrics"].items()
                },
                **{
                    f"routing_{name}": value
                    for name, value in record["routing"].items()
                },
                **{
                    f"diagnostic_{name}": value
                    for name, value in record["diagnostics"].items()
                },
            }
        )
    return pd.DataFrame(rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("tmp/routing_geometry_smoke"))
    parser.add_argument("--n-samples", type=int, default=480)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    result = run_routing_geometry_synthetic_smoke(
        n_samples=args.n_samples,
        seed=args.seed,
        output_dir=args.output_dir,
        show_progress=not args.no_progress,
    )
    print(result.to_string(index=False))
