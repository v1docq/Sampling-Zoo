"""EM-style routed retraining for routed chunk ensembles."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

import numpy as np
import pandas as pd

from sampling_zoo.core.metrics.eval_metrics import calculate_metrics, get_metric_comparator
from sampling_zoo.core.utils.progress import progress_iter


@dataclass(frozen=True)
class RoutedEMConfig:
    mode: str = "none"
    max_iterations: int = 2
    min_improvement: float = 1e-4
    assignment_policy: str = "hard_top1"
    min_partition_size: int = 32
    refit_router: bool = True
    keep_best: bool = True

    @staticmethod
    def from_mapping(config: Mapping[str, Any]) -> "RoutedEMConfig":
        return RoutedEMConfig(
            mode=str(config.get("routing_refinement", "none")).strip().lower(),
            max_iterations=max(0, int(config.get("em_max_iterations", 2))),
            min_improvement=max(0.0, float(config.get("em_min_improvement", 1e-4))),
            assignment_policy=str(config.get("em_assignment_policy", "hard_top1")).strip().lower(),
            min_partition_size=max(1, int(config.get("em_min_partition_size", 32))),
            refit_router=bool(config.get("em_refit_router", True)),
            keep_best=bool(config.get("em_keep_best", True)),
        )


class RoutedEMModelRefiner:
    """Refines routed chunk models by alternating router assignments and model refits."""

    def __init__(
        self,
        *,
        ensemble: Any,
        config: Mapping[str, Any],
        show_progress: bool = True,
    ) -> None:
        self.ensemble = ensemble
        self.config = RoutedEMConfig.from_mapping(config)
        self.show_progress = bool(show_progress)

    def refine(
        self,
        *,
        partitions: Mapping[str, Any],
        X_val: pd.DataFrame,
        y_val: pd.Series,
        validation_metric: str,
    ) -> Dict[str, Any]:
        if self.config.mode in {"", "none"}:
            return self._disabled()
        if self.config.mode != "em_retraining":
            return self._skipped(f"unsupported_mode:{self.config.mode}")
        if self.config.assignment_policy != "hard_top1":
            return self._skipped(f"unsupported_assignment_policy:{self.config.assignment_policy}")
        if getattr(self.ensemble, "problem", None) != "regression":
            return self._skipped("unsupported_problem")
        if len(X_val) == 0 or len(y_val) == 0:
            return self._skipped("no_validation_data")
        if len(getattr(self.ensemble, "models", [])) < 2:
            return self._skipped("too_few_models")

        train_pool = self._build_training_pool(partitions)
        if train_pool is None:
            return self._skipped("empty_training_pool")
        X_pool, y_pool = train_pool

        metric_is_better = get_metric_comparator(validation_metric)
        initial_metrics = self.ensemble._evaluate_current_ensemble(X_val, y_val)
        best_metric = float(initial_metrics[validation_metric])
        initial_metric = best_metric
        best_snapshot = self._snapshot()
        best_iteration = 0
        previous_assignments: Optional[np.ndarray] = None
        iterations: list[dict[str, Any]] = []
        stop_reason = "max_iterations"

        iterator = progress_iter(
            range(1, self.config.max_iterations + 1),
            enabled=self.show_progress,
            total=self.config.max_iterations,
            desc="EM routed retraining",
        )
        for iteration in iterator:
            active_models = list(self.ensemble.models)
            routing_before = self.ensemble._routing_weights(X_pool, active_models)
            entropy_before = self._mean_normalized_entropy(routing_before)
            assignments = np.argmax(routing_before, axis=1)
            assignment_change_rate = (
                None if previous_assignments is None else float(np.mean(assignments != previous_assignments))
            )
            counts = self._assignment_counts(assignments, active_models)
            imbalance_ratio = self._imbalance_ratio(counts)
            if min(counts.values()) < self.config.min_partition_size:
                stop_reason = "min_partition_size"
                iterations.append(
                    self._iteration_payload(
                        iteration=iteration,
                        metric=None,
                        improvement=0.0,
                        counts=counts,
                        imbalance_ratio=imbalance_ratio,
                        assignment_change_rate=assignment_change_rate,
                        entropy_before=entropy_before,
                        entropy_after=None,
                        accepted=False,
                        stop_reason=stop_reason,
                    )
                )
                break

            self._retrain_models_from_assignments(
                X_pool=X_pool,
                y_pool=y_pool,
                X_val=X_val,
                y_val=y_val,
                assignments=assignments,
                active_models=active_models,
            )
            self._refresh_router_and_local_metrics(X_val, y_val)
            current_metrics = self.ensemble._evaluate_current_ensemble(X_val, y_val)
            current_metric = float(current_metrics[validation_metric])
            improvement = self._metric_improvement(validation_metric, best_metric, current_metric)
            accepted = metric_is_better(current_metric, best_metric) and improvement >= self.config.min_improvement
            routing_after = self.ensemble._routing_weights(X_pool, list(self.ensemble.models))
            entropy_after = self._mean_normalized_entropy(routing_after)
            iterations.append(
                self._iteration_payload(
                    iteration=iteration,
                    metric=current_metric,
                    improvement=improvement,
                    counts=counts,
                    imbalance_ratio=imbalance_ratio,
                    assignment_change_rate=assignment_change_rate,
                    entropy_before=entropy_before,
                    entropy_after=entropy_after,
                    accepted=accepted,
                    stop_reason=None,
                )
            )

            if not accepted:
                stop_reason = "no_improvement"
                if self.config.keep_best:
                    self._restore(best_snapshot)
                break

            best_metric = current_metric
            best_iteration = iteration
            best_snapshot = self._snapshot()
            previous_assignments = assignments

        if self.config.keep_best:
            self._restore(best_snapshot)

        final_weights = self.ensemble._routing_weights(X_pool, list(self.ensemble.models))
        final_counts = self._assignment_counts(np.argmax(final_weights, axis=1), list(self.ensemble.models))
        return {
            "mode": self.config.mode,
            "status": "completed",
            "initial_metric": initial_metric,
            "best_metric": best_metric,
            "metric_improvement": self._metric_improvement(validation_metric, initial_metric, best_metric),
            "best_iteration": int(best_iteration),
            "stop_reason": stop_reason,
            "iterations": iterations,
            "final_partition_sizes": final_counts,
            "final_imbalance_ratio": self._imbalance_ratio(final_counts),
            "config": self.config.__dict__.copy(),
        }

    @staticmethod
    def _disabled() -> Dict[str, Any]:
        return {"mode": "none", "status": "disabled"}

    def _skipped(self, reason: str) -> Dict[str, Any]:
        return {"mode": self.config.mode, "status": "skipped", "stop_reason": reason, "config": self.config.__dict__.copy()}

    @staticmethod
    def _build_training_pool(partitions: Mapping[str, Any]) -> Optional[tuple[pd.DataFrame, pd.Series]]:
        features = []
        targets = []
        for partition in partitions.values():
            if not isinstance(partition, Mapping) or "feature" not in partition or "target" not in partition:
                continue
            feature = partition["feature"]
            target = partition["target"]
            if len(feature) == 0:
                continue
            features.append(feature if isinstance(feature, pd.DataFrame) else pd.DataFrame(feature))
            targets.append(target if isinstance(target, pd.Series) else pd.Series(target))
        if not features:
            return None
        return (
            pd.concat(features, ignore_index=True),
            pd.concat(targets, ignore_index=True),
        )

    def _retrain_models_from_assignments(
        self,
        *,
        X_pool: pd.DataFrame,
        y_pool: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        assignments: np.ndarray,
        active_models: list[Mapping[str, Any]],
    ) -> None:
        new_models = []
        new_metrics: Dict[str, Any] = {}
        for model_idx, previous_model_info in enumerate(active_models):
            name = str(previous_model_info.get("name", f"chunk_{model_idx}"))
            mask = assignments == model_idx
            X_part = X_pool.iloc[mask].reset_index(drop=True)
            y_part = y_pool.iloc[mask].reset_index(drop=True)
            model = self.ensemble._create_model_instance()
            model.fit(X_part, y_part)
            val_pred, val_proba = self.ensemble._run_inference(model, X_val, calculation_mode="non-batch")
            metrics = calculate_metrics(
                y_true=y_val,
                problem_type=self.ensemble.problem,
                y_labels=val_pred,
                y_proba=val_proba if self.ensemble.problem == "classification" else None,
            )
            model_info = self.ensemble._build_partition_model_info(
                name,
                model,
                {"feature": X_part, "target": y_part},
                metrics,
                val_pred,
                val_proba,
            )
            new_models.append(model_info)
            new_metrics[name] = metrics
        self.ensemble.models = new_models
        self.ensemble.partition_metrics = new_metrics

    def _refresh_router_and_local_metrics(self, X_val: pd.DataFrame, y_val: pd.Series) -> None:
        if self.config.refit_router:
            self.ensemble.router.fit(
                X_val=X_val,
                y_val=y_val,
                active_models=list(self.ensemble.models),
                partitioner=self.ensemble.partitioner,
            )
        if self.ensemble.router.weights_are_final() or self.ensemble.router.router_mode == "learned_head":
            local_metrics = self.ensemble._build_local_validation_metrics(X_val, y_val, list(self.ensemble.models))
            self.ensemble._attach_local_validation_metrics(self.ensemble.models, local_metrics)

    def _snapshot(self) -> dict[str, Any]:
        return {
            "models": [dict(model_info) for model_info in self.ensemble.models],
            "partition_metrics": dict(self.ensemble.partition_metrics),
            "router": copy.deepcopy(self.ensemble.router),
        }

    def _restore(self, snapshot: Mapping[str, Any]) -> None:
        self.ensemble.models = [dict(model_info) for model_info in snapshot["models"]]
        self.ensemble.partition_metrics = dict(snapshot["partition_metrics"])
        self.ensemble.router = snapshot["router"]
        self.ensemble.router_mode = self.ensemble.router.router_mode

    @staticmethod
    def _metric_improvement(metric: str, previous: float, current: float) -> float:
        lower_is_better = metric.lower() in {"rmse", "mse", "mae", "log_loss", "loss"}
        return float(previous - current) if lower_is_better else float(current - previous)

    @staticmethod
    def _assignment_counts(assignments: np.ndarray, active_models: list[Mapping[str, Any]]) -> Dict[str, int]:
        return {
            str(model_info.get("name", f"chunk_{model_idx}")): int(np.sum(assignments == model_idx))
            for model_idx, model_info in enumerate(active_models)
        }

    @staticmethod
    def _imbalance_ratio(counts: Mapping[str, int]) -> Optional[float]:
        positive = [int(count) for count in counts.values() if int(count) > 0]
        if not positive:
            return None
        return float(max(positive) / max(min(positive), 1))

    @staticmethod
    def _mean_normalized_entropy(weights: np.ndarray) -> float:
        if weights.size == 0 or weights.shape[1] <= 1:
            return 0.0
        entropy = -np.sum(weights * np.log(weights + 1e-12), axis=1)
        return float(np.mean(entropy / np.log(weights.shape[1])))

    @staticmethod
    def _iteration_payload(
        *,
        iteration: int,
        metric: Optional[float],
        improvement: float,
        counts: Mapping[str, int],
        imbalance_ratio: Optional[float],
        assignment_change_rate: Optional[float],
        entropy_before: float,
        entropy_after: Optional[float],
        accepted: bool,
        stop_reason: Optional[str],
    ) -> dict[str, Any]:
        return {
            "iteration": int(iteration),
            "validation_metric": metric,
            "improvement": float(improvement),
            "partition_sizes": dict(counts),
            "imbalance_ratio": imbalance_ratio,
            "assignment_change_rate": assignment_change_rate,
            "routing_entropy_before": entropy_before,
            "routing_entropy_after": entropy_after,
            "accepted": bool(accepted),
            "stop_reason": stop_reason,
        }
