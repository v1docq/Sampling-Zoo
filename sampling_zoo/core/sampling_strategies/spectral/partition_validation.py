"""Deterministic validation proxy for spectral partition candidates."""

from __future__ import annotations

import math
from typing import Any, Callable, Optional, Sequence, Tuple

import numpy as np

from .cluster_selection_contracts import (
    PartitionDownstreamComponents,
    PartitionValidationComponents,
    PartitionValidationPlan,
)
from .partition_sampling import select_partition_indices
from ...experiment.budgeting import build_partition_budget_plan


class PartitionValidationProxyEvaluator:
    """Score routed local constant experts on one shared internal holdout."""

    def __init__(
        self,
        embedding: np.ndarray,
        target: Any,
        *,
        target_type: str,
        validation_fraction: float = 0.2,
        min_partition_train_rows: int = 8,
        classification_smoothing: float = 1.0,
        random_state: Optional[int] = 42,
    ) -> None:
        self.embedding = _as_embedding(embedding)
        self.target = _as_target(target, self.embedding.shape[0])
        self.target_type = _validate_target_type(target_type)
        self.classification_smoothing = _validate_positive_float(
            "classification_smoothing",
            classification_smoothing,
        )
        self.plan = build_partition_validation_plan(
            self.target,
            target_type=self.target_type,
            validation_fraction=validation_fraction,
            min_partition_train_rows=min_partition_train_rows,
            random_state=random_state,
        )
        self._encoded_target, self._class_labels = self._encode_target()

    def evaluate(self, labels: Sequence[int]) -> PartitionValidationComponents:
        normalized_labels = _normalize_partition_labels(
            labels,
            self.embedding.shape[0],
        )
        train_indices = np.asarray(self.plan.train_indices, dtype=int)
        validation_indices = np.asarray(self.plan.validation_indices, dtype=int)
        centers, available = self._build_train_centers(
            normalized_labels,
            train_indices,
        )
        routed = _route_to_centers(
            self.embedding[validation_indices],
            centers,
        )
        routed_counts = np.bincount(
            routed,
            minlength=centers.shape[0],
        )
        if self.target_type == "regression":
            baseline_loss, candidate_loss = self._evaluate_regression(
                normalized_labels,
                train_indices,
                validation_indices,
                routed,
                available,
            )
            loss_name = "rmse"
        else:
            baseline_loss, candidate_loss = self._evaluate_classification(
                normalized_labels,
                train_indices,
                validation_indices,
                routed,
                available,
            )
            loss_name = "log_loss"

        fallback_rows = int(np.sum(routed_counts[~available]))
        return PartitionValidationComponents(
            target_type=self.target_type,
            loss_name=loss_name,
            baseline_loss=float(baseline_loss),
            candidate_loss=float(candidate_loss),
            relative_gain=_relative_loss_gain(baseline_loss, candidate_loss),
            routed_validation_counts=tuple(
                int(count) for count in routed_counts.tolist()
            ),
            fallback_partition_count=int(np.sum(~available)),
            fallback_validation_fraction=float(
                fallback_rows / max(validation_indices.size, 1)
            ),
        )

    def _encode_target(self) -> Tuple[np.ndarray, Tuple[str, ...]]:
        if self.target_type == "regression":
            numeric = np.asarray(self.target, dtype=float)
            if not np.all(np.isfinite(numeric)):
                raise ValueError("regression target must contain finite values")
            return numeric, ()
        classes, encoded = np.unique(self.target, return_inverse=True)
        if classes.size < 2:
            raise ValueError("classification target must contain at least two classes")
        return encoded.astype(int), tuple(str(value) for value in classes.tolist())

    def _build_train_centers(
        self,
        labels: np.ndarray,
        train_indices: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        n_partitions = int(labels.max()) + 1
        centers = np.empty((n_partitions, self.embedding.shape[1]), dtype=float)
        available = np.zeros(n_partitions, dtype=bool)
        train_labels = labels[train_indices]
        global_train_center = np.mean(self.embedding[train_indices], axis=0)
        for partition_id in range(n_partitions):
            local_train = train_indices[train_labels == partition_id]
            available[partition_id] = (
                local_train.size >= self.plan.min_partition_train_rows
            )
            centers[partition_id] = (
                np.mean(self.embedding[local_train], axis=0)
                if local_train.size > 0
                else global_train_center
            )
        return centers, available

    def _evaluate_regression(
        self,
        labels: np.ndarray,
        train_indices: np.ndarray,
        validation_indices: np.ndarray,
        routed: np.ndarray,
        available: np.ndarray,
    ) -> Tuple[float, float]:
        target = np.asarray(self._encoded_target, dtype=float)
        global_mean = float(np.mean(target[train_indices]))
        predictions = np.full(validation_indices.size, global_mean, dtype=float)
        for partition_id in range(available.size):
            if not available[partition_id]:
                continue
            local_train = train_indices[labels[train_indices] == partition_id]
            predictions[routed == partition_id] = float(
                np.mean(target[local_train])
            )
        validation_target = target[validation_indices]
        baseline = np.full(validation_indices.size, global_mean, dtype=float)
        return (
            _root_mean_squared_error(validation_target, baseline),
            _root_mean_squared_error(validation_target, predictions),
        )

    def _evaluate_classification(
        self,
        labels: np.ndarray,
        train_indices: np.ndarray,
        validation_indices: np.ndarray,
        routed: np.ndarray,
        available: np.ndarray,
    ) -> Tuple[float, float]:
        target = np.asarray(self._encoded_target, dtype=int)
        n_classes = len(self._class_labels)
        global_probabilities = _smoothed_class_probabilities(
            target[train_indices],
            n_classes=n_classes,
            smoothing=self.classification_smoothing,
        )
        probabilities = np.tile(
            global_probabilities,
            (validation_indices.size, 1),
        )
        for partition_id in range(available.size):
            if not available[partition_id]:
                continue
            local_train = train_indices[labels[train_indices] == partition_id]
            probabilities[routed == partition_id] = _smoothed_class_probabilities(
                target[local_train],
                n_classes=n_classes,
                smoothing=self.classification_smoothing,
            )
        validation_target = target[validation_indices]
        baseline = np.tile(
            global_probabilities,
            (validation_indices.size, 1),
        )
        return (
            _multiclass_log_loss(validation_target, baseline),
            _multiclass_log_loss(validation_target, probabilities),
        )


class PartitionDownstreamProxyEvaluator:
    """Rerank candidates with budgeted experts and routed holdout predictions."""

    def __init__(
        self,
        embedding: np.ndarray,
        features: np.ndarray,
        target: Any,
        sample_scores: np.ndarray,
        *,
        target_type: str,
        model_factory: Callable[[], Any],
        budget_ratio: float,
        min_partition_rows: int,
        max_imbalance_ratio: float | None,
        min_partition_fraction: float,
        selection_method: str,
        routing_temperature: float,
        routing_shrinkage: float,
        validation_fraction: float = 0.2,
        random_state: Optional[int] = 42,
    ) -> None:
        self.embedding = _as_embedding(embedding)
        self.features = _as_features(features, self.embedding.shape[0])
        self.target = _as_target(target, self.embedding.shape[0])
        self.sample_scores = _as_scores(sample_scores, self.embedding.shape[0])
        self.target_type = _validate_target_type(target_type)
        if self.target_type != "regression":
            raise ValueError(
                "downstream_proxy currently supports regression targets only"
            )
        if not callable(model_factory):
            raise ValueError("downstream_proxy requires a callable model_factory")
        self.model_factory = model_factory
        self.budget_ratio = float(budget_ratio)
        self.min_partition_rows = int(min_partition_rows)
        self.max_imbalance_ratio = max_imbalance_ratio
        self.min_partition_fraction = float(min_partition_fraction)
        self.selection_method = str(selection_method)
        self.routing_temperature = max(float(routing_temperature), 1e-12)
        self.routing_shrinkage = min(max(float(routing_shrinkage), 0.0), 1.0)
        self.plan = build_partition_validation_plan(
            self.target,
            target_type=self.target_type,
            validation_fraction=validation_fraction,
            min_partition_train_rows=self.min_partition_rows,
            random_state=random_state,
        )

    def evaluate(self, labels: Sequence[int]) -> PartitionDownstreamComponents:
        normalized_labels = _normalize_partition_labels(
            labels,
            self.embedding.shape[0],
        )
        train_indices = np.asarray(self.plan.train_indices, dtype=int)
        validation_indices = np.asarray(self.plan.validation_indices, dtype=int)
        partition_ids = tuple(range(int(normalized_labels.max()) + 1))
        source_indices = tuple(
            train_indices[normalized_labels[train_indices] == partition_id]
            for partition_id in partition_ids
        )
        budget_plan = build_partition_budget_plan(
            {
                f"chunk_{partition_id}": int(indices.size)
                for partition_id, indices in zip(partition_ids, source_indices)
            },
            # Preserve the sampler's absolute budget. The holdout reduces the
            # available training pool, not the number of rows the candidate
            # is expected to use at final fit time.
            total_rows=int(self.embedding.shape[0]),
            budget_ratio=self.budget_ratio,
            min_rows_per_partition=self.min_partition_rows,
            max_imbalance_ratio=self.max_imbalance_ratio,
            min_partition_fraction=self.min_partition_fraction,
        )
        allocation = budget_plan.allocation_map
        global_loss = self._global_baseline_loss(train_indices, validation_indices)
        if not budget_plan.feasible:
            return PartitionDownstreamComponents(
                status="infeasible_budget",
                loss_name="rmse",
                global_baseline_loss=global_loss,
                concatenated_loss=math.inf,
                candidate_loss=math.inf,
                relative_gain_vs_global=-math.inf,
                relative_gain_vs_concatenated=-math.inf,
                routed_validation_counts=tuple(0 for _ in partition_ids),
                sampled_partition_sizes=tuple(
                    allocation[f"chunk_{partition_id}"]
                    for partition_id in partition_ids
                ),
                unique_sampled_rows=0,
                budget_reference_rows=int(self.embedding.shape[0]),
                proxy_train_rows=int(train_indices.size),
                requested_budget_size=int(budget_plan.requested_budget_size),
                selected_budget_size=int(budget_plan.selected_size),
                budget_violations=tuple(
                    violation.value for violation in budget_plan.violations
                ),
            )

        sampled_indices = tuple(
            select_partition_indices(
                indices,
                target_size=allocation[f"chunk_{partition_id}"],
                selection_method=self.selection_method,
                scores=self.sample_scores,
                embedding=self.embedding,
            )
            for partition_id, indices in zip(partition_ids, source_indices)
        )
        expert_predictions = []
        centers = []
        for indices, train_partition_indices in zip(
            sampled_indices,
            source_indices,
        ):
            model = self.model_factory()
            model.fit(self.features[indices], self.target[indices])
            expert_predictions.append(
                np.asarray(
                    model.predict(self.features[validation_indices]),
                    dtype=float,
                ).reshape(-1)
            )
            centers.append(np.mean(self.embedding[train_partition_indices], axis=0))

        expert_matrix = np.column_stack(expert_predictions)
        routing = _soft_routing_probabilities(
            self.embedding[validation_indices],
            np.asarray(centers, dtype=float),
            temperature=self.routing_temperature,
            shrinkage=self.routing_shrinkage,
        )
        candidate_prediction = np.sum(expert_matrix * routing, axis=1)
        validation_target = np.asarray(self.target[validation_indices], dtype=float)
        candidate_loss = _root_mean_squared_error(
            validation_target,
            candidate_prediction,
        )

        concatenated_indices = np.unique(np.concatenate(sampled_indices))
        concatenated_model = self.model_factory()
        concatenated_model.fit(
            self.features[concatenated_indices],
            self.target[concatenated_indices],
        )
        concatenated_prediction = np.asarray(
            concatenated_model.predict(self.features[validation_indices]),
            dtype=float,
        ).reshape(-1)
        concatenated_loss = _root_mean_squared_error(
            validation_target,
            concatenated_prediction,
        )
        routed_counts = np.bincount(
            np.argmax(routing, axis=1),
            minlength=len(partition_ids),
        )
        return PartitionDownstreamComponents(
            status="ok",
            loss_name="rmse",
            global_baseline_loss=global_loss,
            concatenated_loss=concatenated_loss,
            candidate_loss=candidate_loss,
            relative_gain_vs_global=_relative_loss_gain(
                global_loss,
                candidate_loss,
            ),
            relative_gain_vs_concatenated=_relative_loss_gain(
                concatenated_loss,
                candidate_loss,
            ),
            routed_validation_counts=tuple(int(value) for value in routed_counts),
            sampled_partition_sizes=tuple(int(indices.size) for indices in sampled_indices),
            unique_sampled_rows=int(concatenated_indices.size),
            budget_reference_rows=int(self.embedding.shape[0]),
            proxy_train_rows=int(train_indices.size),
            requested_budget_size=int(budget_plan.requested_budget_size),
            selected_budget_size=int(budget_plan.selected_size),
            budget_violations=tuple(
                violation.value for violation in budget_plan.violations
            ),
        )

    def _global_baseline_loss(
        self,
        train_indices: np.ndarray,
        validation_indices: np.ndarray,
    ) -> float:
        target = np.asarray(self.target, dtype=float)
        baseline = np.full(
            validation_indices.size,
            float(np.mean(target[train_indices])),
            dtype=float,
        )
        return _root_mean_squared_error(target[validation_indices], baseline)


def build_partition_validation_plan(
    target: Any,
    *,
    target_type: str,
    validation_fraction: float,
    min_partition_train_rows: int,
    random_state: Optional[int],
) -> PartitionValidationPlan:
    """Create one deterministic holdout plan reused by every candidate."""

    target_array = np.asarray(target)
    if target_array.ndim != 1 or target_array.size < 3:
        raise ValueError("validation proxy requires at least three target rows")
    target_type = _validate_target_type(target_type)
    validation_fraction = float(validation_fraction)
    if not np.isfinite(validation_fraction) or not 0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be in (0, 0.5)")
    min_partition_train_rows = int(min_partition_train_rows)
    if min_partition_train_rows < 1:
        raise ValueError("min_partition_train_rows must be positive")

    rng = np.random.default_rng(random_state)
    if target_type == "classification":
        classes, encoded = np.unique(target_array, return_inverse=True)
        if classes.size < 2:
            raise ValueError(
                "classification target must contain at least two classes"
            )
        validation_indices = []
        for class_index in range(classes.size):
            class_indices = np.flatnonzero(encoded == class_index)
            shuffled = rng.permutation(class_indices)
            validation_count = 0
            if class_indices.size >= 2:
                validation_count = min(
                    max(1, int(math.ceil(validation_fraction * class_indices.size))),
                    class_indices.size - 1,
                )
            validation_indices.extend(shuffled[:validation_count].tolist())
        class_labels = tuple(str(value) for value in classes.tolist())
    else:
        shuffled = rng.permutation(target_array.size)
        validation_count = min(
            max(1, int(math.ceil(validation_fraction * target_array.size))),
            target_array.size - 1,
        )
        validation_indices = shuffled[:validation_count].tolist()
        class_labels = ()

    validation_set = set(int(index) for index in validation_indices)
    train_indices = tuple(
        index for index in range(target_array.size) if index not in validation_set
    )
    validation_tuple = tuple(sorted(validation_set))
    if not train_indices or not validation_tuple:
        raise ValueError("validation proxy split must contain train and validation rows")
    return PartitionValidationPlan(
        target_type=target_type,
        train_indices=train_indices,
        validation_indices=validation_tuple,
        requested_validation_fraction=validation_fraction,
        effective_validation_fraction=(len(validation_tuple) / target_array.size),
        random_state=random_state,
        min_partition_train_rows=min_partition_train_rows,
        class_labels=class_labels,
    )


def _as_embedding(embedding: np.ndarray) -> np.ndarray:
    normalized = np.asarray(embedding, dtype=float)
    if normalized.ndim != 2 or normalized.shape[0] < 3:
        raise ValueError("embedding must be a 2D matrix with at least three rows")
    if not np.all(np.isfinite(normalized)):
        raise ValueError("embedding must contain finite values")
    return normalized


def _as_target(target: Any, n_samples: int) -> np.ndarray:
    normalized = np.asarray(target)
    if normalized.ndim != 1 or normalized.size != n_samples:
        raise ValueError("target must be one-dimensional and align with embedding")
    return normalized


def _as_features(features: Any, n_samples: int) -> np.ndarray:
    normalized = np.asarray(features, dtype=float)
    if normalized.ndim != 2 or normalized.shape[0] != n_samples:
        raise ValueError("features must be a 2D matrix aligned with embedding")
    if not np.all(np.isfinite(normalized)):
        raise ValueError("features must contain finite values")
    return normalized


def _as_scores(scores: Any, n_samples: int) -> np.ndarray:
    normalized = np.asarray(scores, dtype=float)
    if normalized.ndim != 1 or normalized.size != n_samples:
        raise ValueError("sample_scores must align with embedding")
    if not np.all(np.isfinite(normalized)):
        raise ValueError("sample_scores must contain finite values")
    return normalized


def _validate_target_type(target_type: str) -> str:
    normalized = str(target_type).strip().lower()
    if normalized not in {"regression", "classification"}:
        raise ValueError("target_type must be regression or classification")
    return normalized


def _validate_positive_float(name: str, value: float) -> float:
    normalized = float(value)
    if not np.isfinite(normalized) or normalized <= 0:
        raise ValueError(f"{name} must be a positive finite value")
    return normalized


def _normalize_partition_labels(labels: Sequence[int], n_samples: int) -> np.ndarray:
    normalized = np.asarray(labels)
    if normalized.ndim != 1 or normalized.size != n_samples:
        raise ValueError("partition labels must align with embedding rows")
    unique = np.unique(normalized)
    if unique.size < 1:
        raise ValueError("partition labels must contain at least one partition")
    remap = {value: index for index, value in enumerate(unique.tolist())}
    return np.asarray([remap[value] for value in normalized], dtype=int)


def _route_to_centers(embedding: np.ndarray, centers: np.ndarray) -> np.ndarray:
    distances = np.sum(
        (embedding[:, None, :] - centers[None, :, :]) ** 2,
        axis=2,
    )
    return np.argmin(distances, axis=1).astype(int)


def _soft_routing_probabilities(
    embedding: np.ndarray,
    centers: np.ndarray,
    *,
    temperature: float,
    shrinkage: float,
) -> np.ndarray:
    distances = np.sum(
        (embedding[:, None, :] - centers[None, :, :]) ** 2,
        axis=2,
    )
    logits = -distances / max(float(temperature), 1e-12)
    logits -= np.max(logits, axis=1, keepdims=True)
    weights = np.exp(logits)
    probabilities = weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    if shrinkage > 0:
        probabilities = (
            (1.0 - shrinkage) * probabilities
            + shrinkage / probabilities.shape[1]
        )
    return probabilities


def _root_mean_squared_error(target: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean((target - prediction) ** 2)))


def _smoothed_class_probabilities(
    encoded_target: np.ndarray,
    *,
    n_classes: int,
    smoothing: float,
) -> np.ndarray:
    counts = np.bincount(encoded_target, minlength=n_classes).astype(float)
    probabilities = counts + float(smoothing)
    return probabilities / probabilities.sum()


def _multiclass_log_loss(target: np.ndarray, probabilities: np.ndarray) -> float:
    clipped = np.clip(probabilities, 1e-12, 1.0)
    return float(-np.mean(np.log(clipped[np.arange(target.size), target])))


def _relative_loss_gain(baseline_loss: float, candidate_loss: float) -> float:
    denominator = max(abs(float(baseline_loss)), 1e-12)
    return float((float(baseline_loss) - float(candidate_loss)) / denominator)
