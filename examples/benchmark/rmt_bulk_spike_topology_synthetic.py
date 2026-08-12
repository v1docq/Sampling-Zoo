"""Synthetic gate for exact-budget bulk/spike expert topologies B0-B4."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Any, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from tqdm.auto import tqdm


ROOT_DIR = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = Path(__file__).resolve().parent
for module_path in (ROOT_DIR, BENCHMARK_DIR):
    if str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))

from rmt_experiment_utils import json_ready, markdown_table  # noqa: E402
from sampling_zoo.core.experiment.routing_geometry_selection import (  # noqa: E402
    ClassificationRoutingGuardSpec,
    CrossFittedRoutingGeometrySelectionSpec,
    CrossFittedRoutingGeometrySelector,
    RoutingGeometryFoldScore,
    RoutingGeometryGuardMetricScore,
)
from sampling_zoo.core.metrics.eval_metrics import calculate_metrics  # noqa: E402
from sampling_zoo.core.sampling_strategies.spectral.backend.matrix_backend import (  # noqa: E402
    MatrixRMTBackend,
)
from sampling_zoo.core.sampling_strategies.spectral.bulk_spike import (  # noqa: E402
    ComponentSplitStatus,
    RowSpectralParticipationContract,
    SpectralComponentSplitContract,
    compute_row_spectral_participation,
)
from sampling_zoo.core.sampling_strategies.spectral.bulk_spike_topology import (  # noqa: E402
    BulkSpikeExpertTopologySpec,
    BulkSpikeHierarchicalRouter,
    BulkSpikePartitionContract,
    BulkSpikeTopologyBuilder,
    BulkSpikeTopologyFoldScore,
    BulkSpikeTopologyMode,
    CrossFittedBulkSpikeTopologySelector,
)
from sampling_zoo.core.sampling_strategies.spectral.routing_contracts import (  # noqa: E402
    PartitionGeometrySpec,
)
from sampling_zoo.core.sampling_strategies.spectral.routing_geometry import (  # noqa: E402
    PartitionGeometryBuilder,
    route_partition_geometry,
)


DEFAULT_TOPOLOGY_REGIMES: tuple[str, ...] = (
    "null",
    "one_spike",
    "multiple_spikes",
    "student_t",
    "heteroskedastic",
    "rare_class_spike",
    "false_spectral_outlier",
)
DEFAULT_TOPOLOGY_SNRS: tuple[float, ...] = (0.5, 1.0, 2.0)
DEFAULT_TOPOLOGY_BUDGETS: tuple[float, ...] = (0.05, 0.10, 0.20)
DEFAULT_TOPOLOGY_SEEDS: tuple[int, ...] = (11, 29, 47, 71, 101)
OFFICIAL_TOPOLOGY_ARMS: tuple[str, ...] = (
    "B0_standard_A9",
    "B1_bulk_single_spike",
    "B2_bulk_multi_spike",
    "B3_validation_selected",
)


@dataclass(frozen=True)
class BulkSpikeTopologySyntheticConfig:
    n_samples: int = 600
    n_features: int = 20
    regimes: Sequence[str] = DEFAULT_TOPOLOGY_REGIMES
    snr_values: Sequence[float] = DEFAULT_TOPOLOGY_SNRS
    budget_ratios: Sequence[float] = DEFAULT_TOPOLOGY_BUDGETS
    seeds: Sequence[int] = DEFAULT_TOPOLOGY_SEEDS
    spectral_rank: int = 8
    null_resamples: int = 8
    validation_fraction: float = 0.20
    test_fraction: float = 0.20
    max_experts: int = 5
    min_partition_size: int = 3
    output_dir: Optional[Path] = None
    show_progress: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "regimes", tuple(str(value) for value in self.regimes))
        object.__setattr__(self, "snr_values", tuple(float(value) for value in self.snr_values))
        object.__setattr__(
            self,
            "budget_ratios",
            tuple(float(value) for value in self.budget_ratios),
        )
        object.__setattr__(self, "seeds", tuple(int(value) for value in self.seeds))
        if self.output_dir is not None:
            object.__setattr__(self, "output_dir", Path(self.output_dir))
        unknown = set(self.regimes) - set(DEFAULT_TOPOLOGY_REGIMES)
        if unknown:
            raise ValueError(f"unknown bulk/spike synthetic regimes: {sorted(unknown)}")
        if int(self.n_samples) < 200 or int(self.n_features) < 8:
            raise ValueError("synthetic topology data are too small")
        if any(value <= 0.0 for value in self.snr_values):
            raise ValueError("snr_values must be positive")
        if any(not 0.0 < value <= 1.0 for value in self.budget_ratios):
            raise ValueError("budget_ratios must be in (0, 1]")
        if not self.seeds:
            raise ValueError("seeds must be non-empty")

    @property
    def expected_records(self) -> int:
        return (
            len(self.regimes)
            * len(self.snr_values)
            * len(self.budget_ratios)
            * len(self.seeds)
            * len(OFFICIAL_TOPOLOGY_ARMS)
        )


@dataclass(frozen=True)
class SyntheticTopologyDataset:
    X: np.ndarray
    y: np.ndarray
    problem_type: str
    classes: tuple[Any, ...]
    true_spike_rows: np.ndarray


@dataclass(frozen=True)
class SyntheticSpectralState:
    center: np.ndarray
    right_basis: np.ndarray
    singular_values: np.ndarray
    split: SpectralComponentSplitContract
    train_embedding: np.ndarray
    train_participation: RowSpectralParticipationContract

    def transform(
        self,
        X: np.ndarray,
    ) -> tuple[np.ndarray, RowSpectralParticipationContract]:
        centered = np.asarray(X, dtype=float) - self.center
        embedding = centered @ self.right_basis.T
        left_coordinates = embedding / np.maximum(
            self.singular_values.reshape(1, -1),
            np.finfo(float).eps,
        )
        return embedding, compute_row_spectral_participation(left_coordinates, self.split)


@dataclass(frozen=True)
class FittedSyntheticTopology:
    topology: BulkSpikePartitionContract
    models: tuple[Any, ...]
    source_labels: np.ndarray
    selected_geometry_arm: str
    fit_seconds: float
    random_state: int


def generate_topology_dataset(
    config: BulkSpikeTopologySyntheticConfig,
    *,
    regime: str,
    snr: float,
    seed: int,
) -> SyntheticTopologyDataset:
    rng = np.random.default_rng(seed)
    n, p = config.n_samples, config.n_features
    if regime == "student_t":
        noise = rng.standard_t(5.0, size=(n, p)) / np.sqrt(5.0 / 3.0)
    else:
        noise = rng.normal(size=(n, p))
    base = noise.copy()
    X = noise.copy()
    spike_mask = np.zeros(n, dtype=bool)
    latent_target = base[:, :4] @ np.asarray([1.2, -0.8, 0.5, 0.3])

    n_spike = max(20, int(round(0.12 * n)))
    spike_rows = np.sort(rng.choice(n, n_spike, replace=False))
    one_direction = rng.normal(size=p)
    one_direction /= np.linalg.norm(one_direction)
    amplitude = 3.0 * np.sqrt(float(snr))

    if regime in {
        "one_spike",
        "student_t",
        "heteroskedastic",
        "rare_class_spike",
        "false_spectral_outlier",
    }:
        spike_mask[spike_rows] = True
        X[spike_rows] += amplitude * one_direction
        if regime != "false_spectral_outlier":
            latent_target[spike_rows] += 2.0 * float(snr)
    elif regime == "multiple_spikes":
        second_rows = np.sort(
            rng.choice(np.setdiff1d(np.arange(n), spike_rows), n_spike, replace=False)
        )
        second_direction = rng.normal(size=p)
        second_direction -= second_direction.dot(one_direction) * one_direction
        second_direction /= np.linalg.norm(second_direction)
        spike_mask[np.concatenate([spike_rows, second_rows])] = True
        X[spike_rows] += amplitude * one_direction
        X[second_rows] += amplitude * second_direction
        latent_target[spike_rows] += 2.0 * float(snr)
        latent_target[second_rows] -= 2.0 * float(snr)

    if regime == "heteroskedastic":
        scales = 0.4 + 1.8 * np.linspace(0.0, 1.0, n)
        X += rng.normal(size=X.shape) * scales[:, None]

    if regime == "rare_class_spike":
        quantile = np.quantile(latent_target[~spike_mask], 0.5)
        y = (latent_target > quantile).astype(int)
        y[spike_mask] = 2
        return SyntheticTopologyDataset(
            X=X,
            y=y,
            problem_type="classification",
            classes=(0, 1, 2),
            true_spike_rows=spike_mask,
        )

    target_noise = rng.normal(scale=1.5 / np.sqrt(float(snr)), size=n)
    y = latent_target + target_noise
    return SyntheticTopologyDataset(
        X=X,
        y=y,
        problem_type="regression",
        classes=(),
        true_spike_rows=spike_mask,
    )


def _fit_spectral_state(
    X_train: np.ndarray,
    *,
    rank: int,
    null_resamples: int,
    seed: int,
) -> SyntheticSpectralState:
    center = np.mean(X_train, axis=0)
    centered = X_train - center
    U, singular, Vt = np.linalg.svd(centered, full_matrices=False)
    width = min(int(rank), U.shape[1])
    U = U[:, :width]
    singular = singular[:width]
    Vt = Vt[:width]
    rng = np.random.default_rng(seed + 70_000)
    scales = np.std(centered, axis=0, ddof=1)
    reference = []
    for _ in range(int(null_resamples)):
        surrogate = rng.normal(size=centered.shape) * scales
        reference.append(np.linalg.svd(surrogate, compute_uv=False)[:width])
    reference_array = np.vstack(reference)
    edge_samples = reference_array[:, 0]
    edge = float(np.quantile(edge_samples, 0.95))
    frequencies = np.mean(
        singular.reshape(1, -1) > edge_samples.reshape(-1, 1),
        axis=0,
    )
    spike_mask = (singular > edge) & (frequencies >= 0.80)
    split = SpectralComponentSplitContract(
        status=(
            ComponentSplitStatus.OK
            if np.any(spike_mask)
            else ComponentSplitStatus.NO_STABLE_SPIKES
        ),
        singular_values=singular,
        empirical_bulk_edge=edge,
        selection_frequencies=frequencies,
        spike_mask=spike_mask,
        bulk_mask=~spike_mask,
        min_selection_frequency=0.80,
        frequency_source="moment_matched_gaussian",
        reason=(
            "stable synthetic components exceed the empirical edge"
            if np.any(spike_mask)
            else "no stable synthetic component exceeded the empirical edge"
        ),
    )
    participation = compute_row_spectral_participation(U, split)
    return SyntheticSpectralState(
        center=center,
        right_basis=Vt,
        singular_values=singular,
        split=split,
        train_embedding=centered @ Vt.T,
        train_participation=participation,
    )


class BulkSpikeTopologySyntheticOrchestrator:
    """Resumable pure-core/thin-shell synthetic topology experiment."""

    def __init__(self, config: BulkSpikeTopologySyntheticConfig) -> None:
        self.config = config
        self.output_dir = self._resolve_output_dir()
        self.topology_builder = BulkSpikeTopologyBuilder()
        self.hierarchical_router = BulkSpikeHierarchicalRouter()
        self.geometry_builder = PartitionGeometryBuilder()
        self.backend = MatrixRMTBackend()

    def run(self) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        records_path = self.output_dir / "topology_runs.jsonl"
        records = self._load_records(records_path)
        completed = {
            self._record_key(record)
            for record in records
            if record.get("status") == "completed"
        }
        self._write_meta("running", len(records))
        try:
            grid = [
                (regime, snr, seed)
                for regime in self.config.regimes
                for snr in self.config.snr_values
                for seed in self.config.seeds
            ]
            for regime, snr, seed in tqdm(
                grid,
                desc="Bulk/spike topology synthetic datasets",
                unit="dataset",
                disable=not self.config.show_progress,
            ):
                expected_leaf_keys = {
                    (
                        str(regime),
                        round(float(snr), 12),
                        int(seed),
                        round(float(budget_ratio), 12),
                        str(arm),
                    )
                    for budget_ratio in self.config.budget_ratios
                    for arm in OFFICIAL_TOPOLOGY_ARMS
                }
                if expected_leaf_keys.issubset(completed):
                    continue
                leaf_records = self._run_dataset(regime=regime, snr=snr, seed=seed)
                for record in leaf_records:
                    key = self._record_key(record)
                    if key in completed:
                        continue
                    with records_path.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(json_ready(record), ensure_ascii=False) + "\n")
                    records.append(json_ready(record))
                    completed.add(key)
                    self._write_meta("running", len(records))
            gate = self._build_artifacts(pd.DataFrame(records))
            status = (
                "completed"
                if len(records) == self.config.expected_records
                and bool((pd.DataFrame(records)["status"] == "completed").all())
                else "completed_with_failures"
            )
            self._write_meta(status, len(records), gate=gate)
        except Exception as exc:
            self._write_meta(
                "failed",
                len(records),
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        return self.output_dir

    def _run_dataset(self, *, regime: str, snr: float, seed: int) -> list[dict[str, Any]]:
        dataset = generate_topology_dataset(
            self.config,
            regime=regime,
            snr=snr,
            seed=seed,
        )
        indices = np.arange(self.config.n_samples)
        stratify = dataset.y if dataset.problem_type == "classification" else None
        train_idx, test_idx = train_test_split(
            indices,
            test_size=self.config.test_fraction,
            random_state=seed,
            stratify=stratify,
        )
        validation_share = self.config.validation_fraction / (1.0 - self.config.test_fraction)
        train_stratify = dataset.y[train_idx] if stratify is not None else None
        train_idx, val_idx = train_test_split(
            train_idx,
            test_size=validation_share,
            random_state=seed + 1,
            stratify=train_stratify,
        )
        X_train, y_train = dataset.X[train_idx], dataset.y[train_idx]
        X_val, y_val = dataset.X[val_idx], dataset.y[val_idx]
        X_test, y_test = dataset.X[test_idx], dataset.y[test_idx]
        spectral = _fit_spectral_state(
            X_train,
            rank=self.config.spectral_rank,
            null_resamples=self.config.null_resamples,
            seed=seed,
        )
        val_embedding, val_participation = spectral.transform(X_val)
        test_embedding, test_participation = spectral.transform(X_test)
        rows = []
        for budget_ratio in self.config.budget_ratios:
            total_budget = max(1, int(round(len(X_train) * budget_ratio)))
            topologies = self._build_topologies(
                spectral,
                total_budget=total_budget,
                seed=seed,
            )
            fitted = {
                arm: self._fit_topology(
                    arm,
                    topology,
                    X_train,
                    y_train,
                    spectral.train_embedding,
                    spectral.train_participation,
                    dataset.problem_type,
                    dataset.classes,
                    seed,
                )
                for arm, topology in topologies.items()
            }
            evaluations = {
                arm: self._evaluate_topology(
                    fitted_topology,
                    X_val=X_val,
                    y_val=y_val,
                    X_test=X_test,
                    y_test=y_test,
                    train_embedding=spectral.train_embedding,
                    val_embedding=val_embedding,
                    test_embedding=test_embedding,
                    train_participation=spectral.train_participation,
                    val_participation=val_participation,
                    test_participation=test_participation,
                    problem_type=dataset.problem_type,
                    classes=dataset.classes,
                )
                for arm, fitted_topology in fitted.items()
            }
            selected_arm, selection = self._select_topology(
                evaluations,
                split=spectral.split,
                topologies=topologies,
                y_val=y_val,
                problem_type=dataset.problem_type,
                classes=dataset.classes,
            )
            evaluations["B3_validation_selected"] = {
                **evaluations[selected_arm],
                "selected_topology_arm": selected_arm,
                "topology_selection": selection,
            }
            for arm in OFFICIAL_TOPOLOGY_ARMS:
                evaluation = evaluations[arm]
                topology = topologies.get(arm, topologies[selected_arm])
                rows.append(
                    self._record(
                        regime=regime,
                        snr=snr,
                        seed=seed,
                        budget_ratio=budget_ratio,
                        arm=arm,
                        topology=topology,
                        evaluation=evaluation,
                        split=spectral.split,
                        y_test=y_test,
                        problem_type=dataset.problem_type,
                        classes=dataset.classes,
                    )
                )
        return rows

    def _build_topologies(
        self,
        spectral: SyntheticSpectralState,
        *,
        total_budget: int,
        seed: int,
    ) -> dict[str, BulkSpikePartitionContract]:
        baseline = self._build_standard_partitions(
            spectral.train_embedding,
            total_budget=total_budget,
            seed=seed,
        )
        common = {
            "max_experts": self.config.max_experts,
            "min_partition_size": self.config.min_partition_size,
            "signal_threshold": 0.50,
        }
        return {
            "B0_standard_A9": self.topology_builder.build(
                BulkSpikeExpertTopologySpec(
                    name="B0_standard_A9",
                    mode="standard",
                    **common,
                ),
                spectral.train_participation,
                total_budget=total_budget,
                baseline_partitions=baseline,
                random_state=seed,
            ),
            "B1_bulk_single_spike": self.topology_builder.build(
                BulkSpikeExpertTopologySpec(
                    name="B1_bulk_single_spike",
                    mode="single_spike",
                    **common,
                ),
                spectral.train_participation,
                total_budget=total_budget,
                random_state=seed + 10,
            ),
            "B2_bulk_multi_spike": self.topology_builder.build(
                BulkSpikeExpertTopologySpec(
                    name="B2_bulk_multi_spike",
                    mode="multi_spike",
                    **common,
                ),
                spectral.train_participation,
                total_budget=total_budget,
                random_state=seed + 20,
            ),
        }

    def _build_standard_partitions(
        self,
        embedding: np.ndarray,
        *,
        total_budget: int,
        seed: int,
    ) -> dict[str, np.ndarray]:
        n_clusters = min(3, total_budget, len(embedding))
        labels = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit_predict(
            embedding
        )
        candidates = [np.flatnonzero(labels == cluster) for cluster in range(n_clusters)]
        capacities = np.asarray([len(values) for values in candidates], dtype=int)
        budgets = self._allocate_counts(total_budget, capacities)
        rng = np.random.default_rng(seed)
        return {
            f"chunk_{cluster}": np.sort(
                rng.choice(values, size=int(budget), replace=False)
            )
            for cluster, (values, budget) in enumerate(zip(candidates, budgets))
            if budget > 0
        }

    @staticmethod
    def _allocate_counts(total: int, capacities: np.ndarray) -> np.ndarray:
        masses = capacities.astype(float)
        ideal = masses / np.sum(masses) * int(total)
        counts = np.minimum(np.floor(ideal).astype(int), capacities)
        remaining = int(total - np.sum(counts))
        remainders = ideal - np.floor(ideal)
        while remaining:
            eligible = np.flatnonzero(counts < capacities)
            chosen = sorted(eligible.tolist(), key=lambda idx: (-remainders[idx], idx))[0]
            counts[chosen] += 1
            remainders[chosen] -= 1.0
            remaining -= 1
        return counts

    def _fit_topology(
        self,
        arm: str,
        topology: BulkSpikePartitionContract,
        X_train: np.ndarray,
        y_train: np.ndarray,
        train_embedding: np.ndarray,
        participation: RowSpectralParticipationContract,
        problem_type: str,
        classes: tuple[Any, ...],
        seed: int,
    ) -> FittedSyntheticTopology:
        started = perf_counter()
        models = []
        for partition_index, rows in enumerate(topology.row_indices):
            if problem_type == "regression":
                model = Ridge(alpha=1.0)
            elif np.unique(y_train[rows]).size < 2:
                model = DummyClassifier(strategy="prior")
            else:
                model = LogisticRegression(max_iter=300, random_state=seed + partition_index)
            model.fit(X_train[rows], y_train[rows])
            models.append(model)
        source_labels = self._source_labels(
            arm,
            topology,
            train_embedding,
            participation,
        )
        return FittedSyntheticTopology(
            topology=topology,
            models=tuple(models),
            source_labels=source_labels,
            selected_geometry_arm="A9_validation_selected" if arm == "B0_standard_A9" else "hierarchical",
            fit_seconds=float(perf_counter() - started),
            random_state=int(seed),
        )

    @staticmethod
    def _source_labels(
        arm: str,
        topology: BulkSpikePartitionContract,
        embedding: np.ndarray,
        participation: RowSpectralParticipationContract,
    ) -> np.ndarray:
        if arm == "B0_standard_A9":
            centers = np.vstack(
                [np.mean(embedding[rows], axis=0) for rows in topology.row_indices]
            )
            return np.argmin(
                np.sum((embedding[:, None, :] - centers[None, :, :]) ** 2, axis=2),
                axis=1,
            )
        labels = np.zeros(len(embedding), dtype=int)
        spike_columns = [
            index for index, regime in enumerate(topology.regimes) if regime.startswith("spike")
        ]
        if not spike_columns:
            return labels
        spike_components = [
            int(topology.regimes[index].split(":", 1)[1])
            if ":" in topology.regimes[index]
            else 0
            for index in spike_columns
        ]
        if len(spike_columns) == 1:
            labels[participation.signalness >= 0.5] = spike_columns[0]
        else:
            assignments = np.argmax(
                participation.spike_signatures[:, spike_components],
                axis=1,
            )
            active = participation.signalness >= 0.5
            labels[active] = np.asarray(spike_columns)[assignments[active]]
        return labels

    def _evaluate_topology(
        self,
        fitted: FittedSyntheticTopology,
        *,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        train_embedding: np.ndarray,
        val_embedding: np.ndarray,
        test_embedding: np.ndarray,
        train_participation: RowSpectralParticipationContract,
        val_participation: RowSpectralParticipationContract,
        test_participation: RowSpectralParticipationContract,
        problem_type: str,
        classes: tuple[Any, ...],
    ) -> dict[str, Any]:
        val_outputs = self._expert_outputs(fitted.models, X_val, problem_type, classes)
        test_outputs = self._expert_outputs(fitted.models, X_test, problem_type, classes)
        if fitted.topology.mode is BulkSpikeTopologyMode.STANDARD:
            val_weights, test_weights, geometry_arm = self._route_standard_a9(
                fitted,
                train_embedding,
                val_embedding,
                test_embedding,
                val_outputs,
                y_val,
                problem_type,
                classes,
            )
        else:
            val_weights, test_weights, geometry_arm = self._route_hierarchical_geometry(
                fitted,
                train_participation,
                val_participation,
                test_participation,
                val_outputs,
                y_val,
                problem_type,
                classes,
            )
        val_prediction = self._blend(val_weights, val_outputs, problem_type)
        test_prediction = self._blend(test_weights, test_outputs, problem_type)
        val_metrics = self._metrics(y_val, val_prediction, problem_type, classes)
        test_metrics = self._metrics(y_test, test_prediction, problem_type, classes)
        oracle_metrics = self._oracle_metrics(y_test, test_outputs, problem_type, classes)
        probability_valid = (
            True
            if problem_type == "regression"
            else bool(
                np.all(test_prediction >= 0.0)
                and np.allclose(np.sum(test_prediction, axis=1), 1.0)
            )
        )
        return {
            "validation_metrics": val_metrics,
            "test_metrics": test_metrics,
            "oracle_metrics": oracle_metrics,
            "val_prediction": val_prediction,
            "test_prediction": test_prediction,
            "test_expert_outputs": test_outputs,
            "test_weights": test_weights,
            "selected_geometry_arm": geometry_arm,
            "probability_valid": probability_valid,
            "fit_seconds": fitted.fit_seconds,
            "mean_max_routing_probability": float(np.mean(np.max(test_weights, axis=1))),
            "mean_routing_entropy": float(
                np.mean(-np.sum(test_weights * np.log(np.maximum(test_weights, 1e-12)), axis=1))
            ),
        }

    def _route_standard_a9(
        self,
        fitted: FittedSyntheticTopology,
        train_embedding: np.ndarray,
        val_embedding: np.ndarray,
        test_embedding: np.ndarray,
        val_outputs: np.ndarray,
        y_val: np.ndarray,
        problem_type: str,
        classes: tuple[Any, ...],
    ) -> tuple[np.ndarray, np.ndarray, str]:
        partitions = {
            name: rows
            for name, rows in zip(
                fitted.topology.partition_names,
                fitted.topology.row_indices,
            )
        }
        mapping = {name: index for index, name in enumerate(partitions)}
        specs = {
            "A2_median_scaled_euclidean": PartitionGeometrySpec(
                metric="median_scaled_euclidean",
                uniform_shrinkage=0.05,
            ),
            "A5_gmm_posterior": PartitionGeometrySpec(
                metric="full_shrinkage_mahalanobis",
                kernel="gmm_posterior",
                uniform_shrinkage=0.05,
            ),
            "A6_cosine_negative_control": PartitionGeometrySpec(
                metric="cosine",
                uniform_shrinkage=0.05,
            ),
        }
        geometries = {
            arm: self.geometry_builder.build(
                embedding=train_embedding,
                cluster_labels=fitted.source_labels,
                partitions=partitions,
                partition_to_cluster=mapping,
                spec=spec,
            )
            for arm, spec in specs.items()
        }
        scores = self._cross_fitted_geometry_scores(
            geometries,
            val_embedding=val_embedding,
            val_outputs=val_outputs,
            y_val=y_val,
            problem_type=problem_type,
            classes=classes,
            random_state=fitted.random_state,
        )
        selector = CrossFittedRoutingGeometrySelector(
            CrossFittedRoutingGeometrySelectionSpec(
                classification_guard=(
                    ClassificationRoutingGuardSpec()
                    if problem_type == "classification"
                    else None
                ),
                bootstrap_iterations=500,
                random_state=fitted.random_state,
            )
        )
        selected_arm = selector.select(scores).selected_arm
        geometry = geometries[selected_arm]
        temperature, val_weights = self._best_geometry_temperature(
            geometry,
            embedding=val_embedding,
            outputs=val_outputs,
            target=y_val,
            problem_type=problem_type,
            classes=classes,
        )
        _, test_routing = route_partition_geometry(
            backend=self.backend,
            embedding=test_embedding,
            geometry=geometry,
            temperature=temperature,
        )
        return val_weights, test_routing.weights, selected_arm

    def _cross_fitted_geometry_scores(
        self,
        geometries: Mapping[str, Any],
        *,
        val_embedding: np.ndarray,
        val_outputs: np.ndarray,
        y_val: np.ndarray,
        problem_type: str,
        classes: tuple[Any, ...],
        random_state: int,
    ) -> list[RoutingGeometryFoldScore]:
        if problem_type == "classification":
            splitter = StratifiedKFold(
                n_splits=3,
                shuffle=True,
                random_state=random_state,
            )
            folds = splitter.split(val_embedding, y_val)
        else:
            splitter = KFold(n_splits=3, shuffle=True, random_state=random_state)
            folds = splitter.split(val_embedding)
        scores: list[RoutingGeometryFoldScore] = []
        for fold_id, (calibration_rows, held_out_rows) in enumerate(folds):
            for arm, geometry in geometries.items():
                temperature, _ = self._best_geometry_temperature(
                    geometry,
                    embedding=val_embedding[calibration_rows],
                    outputs=val_outputs[calibration_rows],
                    target=y_val[calibration_rows],
                    problem_type=problem_type,
                    classes=classes,
                )
                _, held_out_routing = route_partition_geometry(
                    backend=self.backend,
                    embedding=val_embedding[held_out_rows],
                    geometry=geometry,
                    temperature=temperature,
                )
                prediction = self._blend(
                    held_out_routing.weights,
                    val_outputs[held_out_rows],
                    problem_type,
                )
                metrics = self._metrics(
                    y_val[held_out_rows],
                    prediction,
                    problem_type,
                    classes,
                )
                primary_metric, primary_value = self._primary(
                    metrics,
                    problem_type,
                    classes,
                )
                guard_metrics = ()
                if problem_type == "classification":
                    guard_metrics = tuple(
                        RoutingGeometryGuardMetricScore(
                            metric=name,
                            value=metrics[name],
                            direction=(
                                "higher"
                                if name in {"f1_macro", "worst_class_recall"}
                                else "lower"
                            ),
                        )
                        for name in (
                            "brier_score",
                            "expected_calibration_error",
                            "f1_macro",
                            "worst_class_recall",
                        )
                    )
                scores.append(
                    RoutingGeometryFoldScore(
                        arm_name=arm,
                        fold_id=str(fold_id),
                        primary_metric=primary_metric,
                        primary_value=primary_value,
                        primary_direction=(
                            "higher" if primary_metric == "roc_auc" else "lower"
                        ),
                        robust_metric="mae" if problem_type == "regression" else None,
                        robust_value=(
                            metrics["mae"] if problem_type == "regression" else None
                        ),
                        robust_direction=(
                            "lower" if problem_type == "regression" else None
                        ),
                        evaluation_rows=len(held_out_rows),
                        selected_temperature=temperature,
                        guard_metrics=guard_metrics,
                    )
                )
        return scores

    def _best_geometry_temperature(
        self,
        geometry: Any,
        *,
        embedding: np.ndarray,
        outputs: np.ndarray,
        target: np.ndarray,
        problem_type: str,
        classes: tuple[Any, ...],
    ) -> tuple[float, np.ndarray]:
        candidates = []
        for temperature in (0.1, 0.25, 0.5, 1.0, 2.0):
            _, routing = route_partition_geometry(
                backend=self.backend,
                embedding=embedding,
                geometry=geometry,
                temperature=temperature,
            )
            prediction = self._blend(routing.weights, outputs, problem_type)
            metrics = self._metrics(target, prediction, problem_type, classes)
            primary_name, primary_value = self._primary(metrics, problem_type, classes)
            score = primary_value if primary_name in {"rmse", "log_loss"} else -primary_value
            candidates.append((score, temperature, routing.weights))
        _, temperature, weights = sorted(candidates, key=lambda item: (item[0], item[1]))[0]
        return float(temperature), weights

    def _route_hierarchical_geometry(
        self,
        fitted: FittedSyntheticTopology,
        train_participation: RowSpectralParticipationContract,
        val_participation: RowSpectralParticipationContract,
        test_participation: RowSpectralParticipationContract,
        val_outputs: np.ndarray,
        y_val: np.ndarray,
        problem_type: str,
        classes: tuple[Any, ...],
    ) -> tuple[np.ndarray, np.ndarray, str]:
        topology = fitted.topology
        spike_columns = [
            index for index, regime in enumerate(topology.regimes) if regime.startswith("spike")
        ]
        if len(spike_columns) <= 1:
            return (
                self._route_hierarchical(topology, val_participation).weights,
                self._route_hierarchical(topology, test_participation).weights,
                "H0_signalness_gate",
            )
        components = [
            int(topology.regimes[index].split(":", 1)[1])
            for index in spike_columns
        ]
        train_signatures = train_participation.spike_signatures[:, components]
        val_signatures = val_participation.spike_signatures[:, components]
        test_signatures = test_participation.spike_signatures[:, components]
        partitions = {
            topology.partition_names[index]: topology.row_indices[index]
            for index in spike_columns
        }
        mapping = {
            topology.partition_names[index]: index
            for index in spike_columns
        }
        specs = {
            "H1_diag_mahalanobis": PartitionGeometrySpec(
                metric="diag_shrinkage_mahalanobis",
                uniform_shrinkage=0.02,
            ),
            "H2_full_mahalanobis": PartitionGeometrySpec(
                metric="full_shrinkage_mahalanobis",
                uniform_shrinkage=0.02,
            ),
            "H3_gmm_posterior": PartitionGeometrySpec(
                metric="full_shrinkage_mahalanobis",
                kernel="gmm_posterior",
                uniform_shrinkage=0.02,
            ),
        }
        candidates = []
        for arm, spec in specs.items():
            geometry = self.geometry_builder.build(
                embedding=train_signatures,
                cluster_labels=fitted.source_labels,
                partitions=partitions,
                partition_to_cluster=mapping,
                spec=spec,
            )
            for temperature in (0.1, 0.25, 0.5, 1.0, 2.0):
                _, conditional = route_partition_geometry(
                    backend=self.backend,
                    embedding=val_signatures,
                    geometry=geometry,
                    temperature=temperature,
                )
                val_routing = self._route_hierarchical(
                    topology,
                    val_participation,
                    conditional.weights,
                )
                prediction = self._blend(val_routing.weights, val_outputs, problem_type)
                metrics = self._metrics(y_val, prediction, problem_type, classes)
                primary_name, primary_value = self._primary(metrics, problem_type, classes)
                score = primary_value if primary_name in {"rmse", "log_loss"} else -primary_value
                candidates.append(
                    (score, arm, temperature, geometry, val_routing.weights)
                )
        _, arm, temperature, geometry, val_weights = sorted(
            candidates,
            key=lambda item: (item[0], item[1], item[2]),
        )[0]
        _, conditional = route_partition_geometry(
            backend=self.backend,
            embedding=test_signatures,
            geometry=geometry,
            temperature=temperature,
        )
        test_weights = self._route_hierarchical(
            topology,
            test_participation,
            conditional.weights,
        ).weights
        return val_weights, test_weights, arm

    def _route_hierarchical(
        self,
        topology: BulkSpikePartitionContract,
        target_participation: RowSpectralParticipationContract,
        conditional: Optional[np.ndarray] = None,
    ):
        spike_columns = [
            index for index, regime in enumerate(topology.regimes) if regime.startswith("spike")
        ]
        if conditional is None and len(spike_columns) > 1:
            components = [
                int(topology.regimes[index].split(":", 1)[1])
                for index in spike_columns
            ]
            conditional = target_participation.spike_signatures[:, components]
        return self.hierarchical_router.route(
            topology,
            spike_probability=target_participation.signalness,
            conditional_spike_weights=conditional,
        )

    @staticmethod
    def _expert_outputs(
        models: Sequence[Any],
        X: np.ndarray,
        problem_type: str,
        classes: tuple[Any, ...],
    ) -> np.ndarray:
        if problem_type == "regression":
            return np.column_stack([model.predict(X) for model in models])
        outputs = np.zeros((len(X), len(models), len(classes)), dtype=float)
        class_to_index = {label: index for index, label in enumerate(classes)}
        for model_index, model in enumerate(models):
            probabilities = model.predict_proba(X)
            for local_index, label in enumerate(model.classes_):
                outputs[:, model_index, class_to_index[label]] = probabilities[:, local_index]
        sums = np.sum(outputs, axis=2, keepdims=True)
        return np.divide(
            outputs,
            sums,
            out=np.full_like(outputs, 1.0 / len(classes)),
            where=sums > 0.0,
        )

    @staticmethod
    def _blend(weights: np.ndarray, outputs: np.ndarray, problem_type: str) -> np.ndarray:
        if problem_type == "regression":
            return np.sum(weights * outputs, axis=1)
        probabilities = np.sum(weights[:, :, None] * outputs, axis=1)
        probabilities = np.clip(probabilities, 1e-15, 1.0)
        return probabilities / np.sum(probabilities, axis=1, keepdims=True)

    @staticmethod
    def _metrics(
        target: np.ndarray,
        prediction: np.ndarray,
        problem_type: str,
        classes: tuple[Any, ...],
    ) -> dict[str, float]:
        labels = (
            prediction
            if problem_type == "regression"
            else np.asarray(classes)[np.argmax(prediction, axis=1)]
        )
        metrics = calculate_metrics(
            target,
            labels,
            None if problem_type == "regression" else prediction,
            problem_type,
            classes=classes or None,
        )
        if problem_type == "regression":
            absolute_error = np.abs(target - prediction)
            cutoff = np.quantile(absolute_error, 0.90)
            metrics["mae"] = float(np.mean(absolute_error))
            metrics["tail_mae"] = float(np.mean(absolute_error[absolute_error >= cutoff]))
        return {name: float(value) for name, value in metrics.items()}

    def _oracle_metrics(
        self,
        target: np.ndarray,
        outputs: np.ndarray,
        problem_type: str,
        classes: tuple[Any, ...],
    ) -> dict[str, float]:
        if problem_type == "regression":
            best = np.argmin(np.abs(outputs - target[:, None]), axis=1)
            prediction = outputs[np.arange(len(target)), best]
        else:
            class_to_index = {label: index for index, label in enumerate(classes)}
            encoded = np.asarray([class_to_index[value] for value in target])
            true_probability = outputs[np.arange(len(target))[:, None], np.arange(outputs.shape[1])[None, :], encoded[:, None]]
            best = np.argmax(true_probability, axis=1)
            prediction = outputs[np.arange(len(target)), best]
        return self._metrics(target, prediction, problem_type, classes)

    def _select_topology(
        self,
        evaluations: Mapping[str, Mapping[str, Any]],
        *,
        split: SpectralComponentSplitContract,
        topologies: Mapping[str, BulkSpikePartitionContract],
        y_val: np.ndarray,
        problem_type: str,
        classes: tuple[Any, ...],
    ) -> tuple[str, dict[str, Any]]:
        candidate_arms = []
        b1 = topologies["B1_bulk_single_spike"]
        b2 = topologies["B2_bulk_multi_spike"]
        if split.n_spikes >= 1 and b1.n_experts >= 2:
            candidate_arms.append("B1_bulk_single_spike")
        if (
            split.n_spikes >= 2
            and b2.mode is BulkSpikeTopologyMode.MULTI_SPIKE
            and b2.n_experts >= 3
        ):
            candidate_arms.append("B2_bulk_multi_spike")
        if not candidate_arms:
            return "B0_standard_A9", {
                "selected_arm": "B0_standard_A9",
                "fallback_arm": "B0_standard_A9",
                "status": "fallback_to_b0",
                "mean_gain": 0.0,
                "median_gain": 0.0,
                "confidence_interval": (0.0, 0.0),
                "positive_fold_fraction": 0.0,
                "reason": "no_spectrally_eligible_bulk_spike_topology",
                "eligible_candidate_arms": [],
            }
        fold_indices = np.array_split(np.arange(len(y_val)), 5)
        scores = []
        for fold_id, indices in enumerate(fold_indices):
            for arm in ("B0_standard_A9", *candidate_arms):
                prediction = evaluations[arm]["val_prediction"][indices]
                metrics = self._metrics(y_val[indices], prediction, problem_type, classes)
                metric, value = self._primary(metrics, problem_type, classes)
                scores.append(
                    BulkSpikeTopologyFoldScore(
                        arm_name=arm,
                        fold_id=str(fold_id),
                        value=value,
                        direction="higher" if metric == "roc_auc" else "lower",
                    )
                )
        selection = CrossFittedBulkSpikeTopologySelector(
            candidate_arms=tuple(candidate_arms),
            noninferiority_margin=0.0,
            min_positive_fold_fraction=0.8,
            min_mean_gain=0.01,
            min_median_gain=0.005,
            bootstrap_iterations=500,
            random_state=17,
        ).select(scores)
        return selection.selected_arm, {
            "selected_arm": selection.selected_arm,
            "fallback_arm": selection.fallback_arm,
            "status": selection.status,
            "mean_gain": selection.mean_gain,
            "median_gain": selection.median_gain,
            "confidence_interval": selection.confidence_interval,
            "positive_fold_fraction": selection.positive_fold_fraction,
            "reason": selection.reason,
            "eligible_candidate_arms": candidate_arms,
        }

    @staticmethod
    def _primary(
        metrics: Mapping[str, float],
        problem_type: str,
        classes: tuple[Any, ...],
    ) -> tuple[str, float]:
        if problem_type == "regression":
            return "rmse", float(metrics["rmse"])
        metric = "roc_auc" if len(classes) == 2 else "log_loss"
        return metric, float(metrics[metric])

    def _record(
        self,
        *,
        regime: str,
        snr: float,
        seed: int,
        budget_ratio: float,
        arm: str,
        topology: BulkSpikePartitionContract,
        evaluation: Mapping[str, Any],
        split: SpectralComponentSplitContract,
        y_test: np.ndarray,
        problem_type: str,
        classes: tuple[Any, ...],
    ) -> dict[str, Any]:
        test_metrics = evaluation["test_metrics"]
        primary_metric = "rmse" if "rmse" in test_metrics else (
            "roc_auc" if np.isfinite(test_metrics.get("roc_auc", np.nan)) else "log_loss"
        )
        primary_value = float(test_metrics[primary_metric])
        oracle_primary_value = float(evaluation["oracle_metrics"][primary_metric])
        if primary_metric in {"rmse", "log_loss"}:
            oracle_gap = (primary_value - oracle_primary_value) / max(
                abs(primary_value),
                1e-12,
            )
        else:
            oracle_gap = oracle_primary_value - primary_value
        baseline_selected = evaluation.get("selected_topology_arm")
        reversed_prediction = self._blend(
            evaluation["test_weights"][:, ::-1],
            evaluation["test_expert_outputs"][:, ::-1],
            problem_type,
        )
        return {
            "status": "completed",
            "regime": regime,
            "snr": float(snr),
            "seed": int(seed),
            "budget_ratio": float(budget_ratio),
            "arm": arm,
            "realized_topology_mode": topology.mode.value,
            "selected_topology_arm": baseline_selected or arm,
            "primary_metric": primary_metric,
            "primary_value": primary_value,
            "oracle_primary_value": oracle_primary_value,
            "oracle_gap": float(oracle_gap),
            "metrics": test_metrics,
            "oracle_metrics": evaluation["oracle_metrics"],
            "topology": topology.to_dict(),
            "topology_selection": evaluation.get("topology_selection"),
            "selected_geometry_arm": evaluation["selected_geometry_arm"],
            "detected_spike_count": split.n_spikes,
            "n_experts": topology.n_experts,
            "probability_valid": evaluation["probability_valid"],
            "fit_seconds": evaluation["fit_seconds"],
            "mean_max_routing_probability": evaluation[
                "mean_max_routing_probability"
            ],
            "mean_routing_entropy": evaluation["mean_routing_entropy"],
            "exact_budget": topology.selected_indices.size == topology.total_budget,
            "unique_budget": np.unique(topology.selected_indices).size == topology.total_budget,
            "row_permutation_metric_invariant": self._metric_permutation_invariant(
                y_test,
                evaluation["test_prediction"],
                test_metrics,
                primary_metric,
                problem_type,
                classes,
            ),
            "expert_order_permutation_invariant": bool(
                np.allclose(
                    evaluation["test_prediction"],
                    reversed_prediction,
                    rtol=1e-12,
                    atol=1e-12,
                )
            ),
        }

    def _metric_permutation_invariant(
        self,
        target: np.ndarray,
        prediction: np.ndarray,
        metrics: Mapping[str, float],
        primary_metric: str,
        problem_type: str,
        classes: tuple[Any, ...],
    ) -> bool:
        permutation = np.arange(len(target) - 1, -1, -1)
        permuted = self._metrics(
            np.asarray(target)[permutation],
            np.asarray(prediction)[permutation],
            problem_type,
            classes,
        )
        return bool(
            np.isclose(
                float(metrics[primary_metric]),
                float(permuted[primary_metric]),
                rtol=1e-12,
                atol=1e-12,
            )
        )

    def _build_artifacts(self, raw: pd.DataFrame) -> dict[str, Any]:
        raw.to_csv(self.output_dir / "topology_raw_runs.csv", index=False)
        paired = self._pair_with_b0(raw)
        paired.to_csv(self.output_dir / "topology_paired.csv", index=False)
        summary = self._summarize(paired)
        summary.to_csv(self.output_dir / "topology_summary.csv", index=False)
        gate = self._evaluate_gate(raw, paired)
        (self.output_dir / "topology_gate.json").write_text(
            json.dumps(json_ready(gate), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        self._write_report(summary, gate)
        self._write_figures(summary)
        return gate

    @staticmethod
    def _pair_with_b0(raw: pd.DataFrame) -> pd.DataFrame:
        keys = ["regime", "snr", "seed", "budget_ratio"]
        baseline = raw[raw["arm"] == "B0_standard_A9"][
            keys + ["primary_value"]
        ].rename(columns={"primary_value": "b0_primary_value"})
        paired = raw.merge(baseline, on=keys, how="left", validate="many_to_one")
        lower = paired["primary_metric"].isin({"rmse", "log_loss"})
        paired["gain_vs_b0"] = np.where(
            lower,
            (paired["b0_primary_value"] - paired["primary_value"])
            / paired["b0_primary_value"].clip(lower=1e-12),
            paired["primary_value"] - paired["b0_primary_value"],
        )
        return paired

    @staticmethod
    def _summarize(paired: pd.DataFrame) -> pd.DataFrame:
        return (
            paired.groupby(["regime", "snr", "budget_ratio", "arm"], as_index=False)
            .agg(
                runs=("seed", "size"),
                mean_gain=("gain_vs_b0", "mean"),
                median_gain=("gain_vs_b0", "median"),
                worst_gain=("gain_vs_b0", "min"),
                win_rate=("gain_vs_b0", lambda values: float((values > 0.0).mean())),
                mean_fit_seconds=("fit_seconds", "mean"),
                mean_detected_spikes=("detected_spike_count", "mean"),
                mean_expert_count=("n_experts", "mean"),
                mean_routing_entropy=("mean_routing_entropy", "mean"),
                mean_oracle_gap=("oracle_gap", "mean"),
            )
        )

    def _evaluate_gate(self, raw: pd.DataFrame, paired: pd.DataFrame) -> dict[str, Any]:
        b3 = paired[paired["arm"] == "B3_validation_selected"]
        confirmatory = b3[b3["snr"] >= 1.0]
        null_like = b3[b3["regime"].isin({"null", "false_spectral_outlier"})]
        null_fallback = float(
            np.mean(null_like["selected_topology_arm"] == "B0_standard_A9")
        )
        null_b2 = paired[
            (paired["arm"] == "B2_bulk_multi_spike")
            & paired["regime"].isin({"null", "false_spectral_outlier"})
        ]
        spurious_multi_expert_rate = float(np.mean(null_b2["n_experts"] > 2))
        selected_specializations = b3[
            b3["selected_topology_arm"] != "B0_standard_A9"
        ]
        specialization_win_rate = (
            float(np.mean(selected_specializations["gain_vs_b0"] > 0.0))
            if not selected_specializations.empty
            else 0.0
        )
        specialization_median_gain = (
            float(selected_specializations["gain_vs_b0"].median())
            if not selected_specializations.empty
            else 0.0
        )
        multi_spike_materialized = bool(
            (
                (paired["arm"] == "B2_bulk_multi_spike")
                & (paired["realized_topology_mode"] == "multi_spike")
                & (paired["n_experts"] >= 3)
            ).any()
        )
        checks = {
            "all_records_completed": bool(
                len(raw) == self.config.expected_records
                and (raw["status"] == "completed").all()
            ),
            "exact_unique_budget": bool(raw["exact_budget"].all() and raw["unique_budget"].all()),
            "classification_probabilities_valid": bool(raw["probability_valid"].all()),
            "null_and_false_outlier_fallback_rate_at_least_80pct": null_fallback >= 0.80,
            "spurious_multi_expert_rate_at_most_20pct": spurious_multi_expert_rate <= 0.20,
            "multi_spike_topology_materialized": multi_spike_materialized,
            "selected_specialization_win_rate_at_least_80pct": (
                not selected_specializations.empty and specialization_win_rate >= 0.80
            ),
            "selected_specialization_median_gain_positive": (
                not selected_specializations.empty and specialization_median_gain > 0.0
            ),
            "confirmatory_median_gain_nonnegative": bool(
                not confirmatory.empty and float(confirmatory["gain_vs_b0"].median()) >= 0.0
            ),
            "worst_primary_harm_at_most_05pct": bool(
                not confirmatory.empty and float(confirmatory["gain_vs_b0"].min()) >= -0.005
            ),
            "row_permutation_metric_invariant": bool(raw["row_permutation_metric_invariant"].all()),
            "expert_order_permutation_invariant": bool(
                raw["expert_order_permutation_invariant"].all()
            ),
        }
        return {
            "status": "passed" if all(checks.values()) else "failed",
            "checks": checks,
            "null_fallback_rate": null_fallback,
            "spurious_multi_expert_rate": spurious_multi_expert_rate,
            "selected_specialization_count": int(len(selected_specializations)),
            "selected_specialization_win_rate": specialization_win_rate,
            "selected_specialization_median_gain": specialization_median_gain,
            "confirmatory_median_gain": float(confirmatory["gain_vs_b0"].median()),
            "confirmatory_worst_gain": float(confirmatory["gain_vs_b0"].min()),
            "expected_records": self.config.expected_records,
            "observed_records": len(raw),
        }

    def _write_report(self, summary: pd.DataFrame, gate: Mapping[str, Any]) -> None:
        checks = pd.DataFrame(
            [{"проверка": name, "результат": value} for name, value in gate["checks"].items()]
        )
        lines = [
            "# Синтетическая проверка топологий тела и сигнала (bulk/spike)",
            "",
            f"Статус критерия допуска: **{gate['status']}**.",
            "",
            (
                f"Параметры сетки: число режимов — {len(self.config.regimes)}, "
                "число значений отношения сигнал/шум (SNR) — "
                f"{len(self.config.snr_values)}, число бюджетов — "
                f"{len(self.config.budget_ratios)}, число начальных значений "
                f"генератора — {len(self.config.seeds)}, число официальных "
                f"вариантов — {len(OFFICIAL_TOPOLOGY_ARMS)}. "
                f"Ожидаемый объём равен {self.config.expected_records} записям. "
                "B4 вычисляется как диагностическая верхняя граница внутри каждой "
                "записи и не увеличивает число запусков."
            ),
            "",
            "## Критерии",
            "",
            markdown_table(checks),
            "",
            "## Сводка",
            "",
            markdown_table(summary),
        ]
        (self.output_dir / "topology_report.md").write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )

    def _write_figures(self, summary: pd.DataFrame) -> None:
        b3 = summary[summary["arm"] == "B3_validation_selected"]
        figure, axis = plt.subplots(figsize=(10, 6))
        for regime, values in b3.groupby("regime"):
            curve = values.groupby("budget_ratio", as_index=False)["mean_gain"].mean()
            axis.plot(curve["budget_ratio"], curve["mean_gain"], marker="o", label=regime)
        axis.axhline(0.0, color="#555555", linewidth=0.8)
        axis.set_xlabel("Доля бюджета")
        axis.set_ylabel("Средний выигрыш B3 относительно B0")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7, ncol=2)
        figure.tight_layout()
        figure.savefig(self.output_dir / "b3_gain_by_budget.png", dpi=180)
        plt.close(figure)

    def _resolve_output_dir(self) -> Path:
        if self.config.output_dir is not None:
            return Path(self.config.output_dir)
        return BENCHMARK_DIR / "results" / (
            "run_rmt_bulk_spike_topology_synthetic_"
            + datetime.now().strftime("%Y%m%d_%H%M%S")
        )

    def _write_meta(
        self,
        status: str,
        record_count: int,
        *,
        gate: Optional[Mapping[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        config = asdict(self.config)
        config["output_dir"] = str(self.output_dir)
        payload = {
            "experiment": "rmt_bulk_spike_topology_synthetic",
            "status": status,
            "updated_at": datetime.now().isoformat(),
            "record_count": int(record_count),
            "expected_records": self.config.expected_records,
            "config": json_ready(config),
            "gate": json_ready(gate),
            "error": error,
        }
        (self.output_dir / "run_meta.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @staticmethod
    def _load_records(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    @staticmethod
    def _record_key(record: Mapping[str, Any]) -> tuple[str, float, int, float, str]:
        return (
            str(record.get("regime")),
            round(float(record.get("snr")), 12),
            int(record.get("seed")),
            round(float(record.get("budget_ratio")), 12),
            str(record.get("arm")),
        )


def run_rmt_bulk_spike_topology_synthetic(
    config: Optional[BulkSpikeTopologySyntheticConfig] = None,
) -> Path:
    return BulkSpikeTopologySyntheticOrchestrator(
        config or BulkSpikeTopologySyntheticConfig()
    ).run()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = BulkSpikeTopologySyntheticConfig(
        regimes=("null", "one_spike", "rare_class_spike") if args.smoke else DEFAULT_TOPOLOGY_REGIMES,
        snr_values=(1.0,) if args.smoke else DEFAULT_TOPOLOGY_SNRS,
        budget_ratios=(0.10,) if args.smoke else DEFAULT_TOPOLOGY_BUDGETS,
        seeds=(11,) if args.smoke else DEFAULT_TOPOLOGY_SEEDS,
        n_samples=300 if args.smoke else 600,
        null_resamples=3 if args.smoke else 8,
        output_dir=args.output_dir,
        show_progress=not args.no_progress,
    )
    print(run_rmt_bulk_spike_topology_synthetic(config))


if __name__ == "__main__":
    main()
