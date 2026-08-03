"""Routing utilities for routed chunk ensembles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from sampling_zoo.core.metrics.eval_metrics import calculate_metrics
from sampling_zoo.core.utils.progress import progress_bar

torch = None
nn = None
_TORCH_IMPORT_ATTEMPTED = False


@dataclass(frozen=True)
class GatingTrainingData:
    features: np.ndarray
    prior: np.ndarray
    predictions: np.ndarray
    target: np.ndarray


def _load_torch_backend() -> tuple[Any, Any]:
    global torch, nn, _TORCH_IMPORT_ATTEMPTED
    if torch is not None and nn is not None:
        return torch, nn
    if not _TORCH_IMPORT_ATTEMPTED:
        _TORCH_IMPORT_ATTEMPTED = True
        try:
            import torch as _torch
            from torch import nn as _nn
        except Exception:  # pragma: no cover - torch is optional
            torch = None
            nn = None
        else:
            torch = _torch
            nn = _nn
    return torch, nn


def _make_constrained_gating_network(n_features: int, n_models: int, hidden_dim: int) -> Any:
    _torch, nn_module = _load_torch_backend()
    if nn_module is None:
        raise RuntimeError("torch is not available")

    class _ConstrainedGatingNetwork(nn_module.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn_module.Sequential(
                nn_module.Linear(n_features, hidden_dim),
                nn_module.LayerNorm(hidden_dim),
                nn_module.GELU(),
                nn_module.Linear(hidden_dim, hidden_dim),
                nn_module.GELU(),
                nn_module.Linear(hidden_dim, n_models),
            )

        def forward(self, x):  # type: ignore[no-untyped-def]
            return self.net(x)

    return _ConstrainedGatingNetwork()


class RoutedWeightedRouter:
    """Owns routing weights, learned gating, and routing diagnostics for chunk ensembles."""

    def __init__(
        self,
        *,
        problem: str,
        config: Optional[Dict[str, Any]] = None,
        show_progress: bool = True,
    ) -> None:
        self.problem = problem
        self.config = config or {}
        self.show_progress = bool(show_progress)
        self.router_mode = self._normalize_router_mode(self.config.get("router", "spectral"))
        self.router_head_ = None
        self.router_head_model_names_: List[str] = []
        self.gating_model_ = None
        self.gating_model_names_: List[str] = []
        self.gating_device_ = None
        self.gating_input_mean_: Optional[np.ndarray] = None
        self.gating_input_scale_: Optional[np.ndarray] = None
        self.diagnostics_: Dict[str, Any] = {
            "router_mode": self.router_mode,
            "status": "not_fitted" if self.router_mode != "spectral" else "disabled",
        }

    @staticmethod
    def _normalize_router_mode(router_mode: Any) -> str:
        normalized = str(router_mode or "spectral").strip().lower()
        if normalized not in {"spectral", "learned_head", "constrained_gating"}:
            raise ValueError("router must be one of: spectral, learned_head, constrained_gating")
        return normalized

    @staticmethod
    def can_route(partitioner: Any) -> bool:
        return partitioner is not None and hasattr(partitioner, "predict_partition_proba")

    def fit(
        self,
        *,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        active_models: List[Dict[str, Any]],
        partitioner: Any,
    ) -> None:
        self.router_head_ = None
        self.router_head_model_names_ = []
        self.gating_model_ = None
        self.gating_model_names_ = []
        if self.router_mode == "spectral":
            self.diagnostics_ = {"router_mode": self.router_mode, "status": "disabled"}
            return
        if self.router_mode == "learned_head":
            self._fit_legacy_learned_head(X_val, y_val, active_models, partitioner)
            return
        self._fit_constrained_gating(X_val, y_val, active_models, partitioner)

    def weights(
        self,
        *,
        features: pd.DataFrame,
        active_models: List[Dict[str, Any]],
        partitioner: Any,
    ) -> np.ndarray:
        base_weights = self.base_weights(features=features, active_models=active_models, partitioner=partitioner)
        if self.router_mode == "learned_head" and self.router_head_ is not None:
            return self._router_head_weights(base_weights, active_models)
        if self.router_mode == "constrained_gating" and self.gating_model_ is not None:
            return self._constrained_gating_weights(features, base_weights, active_models, partitioner)
        return base_weights

    def weights_are_final(self) -> bool:
        return self.router_mode == "constrained_gating" and self.gating_model_ is not None

    def validation_prior_weights(
        self,
        active_models: List[Dict[str, Any]],
        *,
        ensemble_method: str,
    ) -> np.ndarray:
        if self.weights_are_final():
            return np.full(len(active_models), 1.0 / max(len(active_models), 1), dtype=float)
        raw_weights = []
        eps = 1e-8
        for model_info in active_models:
            metrics = model_info.get("metrics", {}) or {}
            if ensemble_method == "routed_weighted":
                local_metrics = model_info.get("local_metrics", {}) or {}
                if local_metrics and int(model_info.get("local_assigned_count", 0) or 0) > 0:
                    metrics = local_metrics
            if self.problem == "regression":
                if "rmse" in metrics and np.isfinite(metrics["rmse"]):
                    raw_weights.append(1.0 / (float(metrics["rmse"]) + eps))
                elif "mae" in metrics and np.isfinite(metrics["mae"]):
                    raw_weights.append(1.0 / (float(metrics["mae"]) + eps))
                else:
                    raw_weights.append(1.0)
            else:
                value = metrics.get("f1_weighted", metrics.get("accuracy", 1.0))
                raw_weights.append(max(float(value), eps) if np.isfinite(value) else 1.0)

        weights = np.asarray(raw_weights, dtype=float)
        if not np.all(np.isfinite(weights)) or weights.sum() <= 0:
            weights = np.ones(len(active_models), dtype=float)
        return weights / weights.sum()

    def base_weights(
        self,
        *,
        features: pd.DataFrame,
        active_models: List[Dict[str, Any]],
        partitioner: Any,
    ) -> np.ndarray:
        n_samples = len(features)
        n_models = len(active_models)
        if partitioner is None:
            return np.full((n_samples, n_models), 1.0 / max(n_models, 1))

        model_names = [model_info.get("name") for model_info in active_models]
        try:
            if hasattr(partitioner, "predict_partition_proba"):
                proba = np.asarray(partitioner.predict_partition_proba(features), dtype=float)
                partition_names = list(getattr(partitioner, "partition_names_", []))
                if partition_names and proba.shape[1] == len(partition_names):
                    name_to_col = {name: idx for idx, name in enumerate(partition_names)}
                    aligned = np.zeros((proba.shape[0], n_models), dtype=float)
                    for model_idx, name in enumerate(model_names):
                        if name in name_to_col:
                            aligned[:, model_idx] = proba[:, name_to_col[name]]
                    if aligned.sum() > 0:
                        row_sums = aligned.sum(axis=1, keepdims=True)
                        return np.where(row_sums > 0, aligned / row_sums, 1.0 / n_models)

            if hasattr(partitioner, "predict_partitions"):
                labels = np.asarray(partitioner.predict_partitions(features))
                aligned = np.full((labels.shape[0], n_models), 0.0, dtype=float)
                for model_idx, name in enumerate(model_names):
                    try:
                        label_id = int(str(name).split("_")[-1])
                    except Exception:
                        label_id = model_idx
                    aligned[:, model_idx] = (labels == label_id).astype(float)
                row_sums = aligned.sum(axis=1, keepdims=True)
                return np.where(row_sums > 0, aligned / row_sums, 1.0 / n_models)
        except Exception:
            pass

        return np.full((n_samples, n_models), 1.0 / max(n_models, 1))

    def build_local_validation_metrics(
        self,
        *,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        active_models: List[Dict[str, Any]],
        partitioner: Any,
        use_base_router: bool = False,
    ) -> Dict[str, Any]:
        if not active_models or len(X_val) == 0:
            return {}
        routing_weights = (
            self.base_weights(features=X_val, active_models=active_models, partitioner=partitioner)
            if use_base_router
            else self.weights(features=X_val, active_models=active_models, partitioner=partitioner)
        )
        assignments = np.argmax(routing_weights, axis=1)
        y_values = pd.Series(y_val).reset_index(drop=True)
        local_metrics: Dict[str, Any] = {}
        for model_idx, model_info in enumerate(active_models):
            name = str(model_info.get("name", f"chunk_{model_idx}"))
            mask = assignments == model_idx
            assigned_count = int(np.sum(mask))
            if assigned_count == 0:
                local_metrics[name] = {"assigned_count": 0, "metrics": {}}
                continue
            val_predictions = np.asarray(model_info.get("val_predictions"))
            metrics = calculate_metrics(
                y_true=y_values.iloc[mask].to_numpy(),
                y_labels=val_predictions[mask],
                y_proba=None,
                problem_type=self.problem,
            )
            local_metrics[name] = {
                "assigned_count": assigned_count,
                "mean_routing_probability": float(np.mean(routing_weights[mask, model_idx])),
                "metrics": metrics,
            }
        return local_metrics

    def diagnostics(
        self,
        *,
        features: pd.DataFrame,
        active_models: List[Dict[str, Any]],
        partitioner: Any,
    ) -> Dict[str, Any]:
        n_samples = len(features)
        n_models = len(active_models)
        if n_samples == 0 or n_models == 0:
            return {
                "n_rows": int(n_samples),
                "n_models": int(n_models),
                "hard_assignment_counts": {},
                "soft_assignment_mass": {},
                "router": dict(self.diagnostics_),
            }

        routing_weights = self.weights(features=features, active_models=active_models, partitioner=partitioner)
        model_names = [str(model_info.get("name", f"chunk_{idx}")) for idx, model_info in enumerate(active_models)]
        max_probability = np.max(routing_weights, axis=1)
        entropy = np.maximum(-np.sum(routing_weights * np.log(routing_weights + 1e-12), axis=1), 0.0)
        normalized_entropy = np.zeros_like(entropy) if n_models <= 1 else entropy / np.log(n_models)
        assignments = np.argmax(routing_weights, axis=1)

        return {
            "n_rows": int(n_samples),
            "n_models": int(n_models),
            "router_mode": self.router_mode,
            "router_status": self.diagnostics_.get("status"),
            "router_head_status": self.diagnostics_.get("status") if self.router_mode == "learned_head" else None,
            "mean_max_probability": float(np.mean(max_probability)),
            "median_max_probability": float(np.median(max_probability)),
            "mean_entropy": float(np.mean(entropy)),
            "mean_normalized_entropy": float(np.mean(normalized_entropy)),
            "hard_assignment_counts": {
                name: int(np.sum(assignments == idx))
                for idx, name in enumerate(model_names)
            },
            "soft_assignment_mass": {
                name: float(np.sum(routing_weights[:, idx]))
                for idx, name in enumerate(model_names)
            },
            "router": dict(self.diagnostics_),
        }

    def _fit_constrained_gating(
        self,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        active_models: List[Dict[str, Any]],
        partitioner: Any,
    ) -> None:
        if self.problem != "regression":
            self.diagnostics_ = {"router_mode": self.router_mode, "status": "skipped_unsupported_problem"}
            return
        torch_module, _nn_module = _load_torch_backend()
        if torch_module is None or _nn_module is None:
            self.diagnostics_ = {"router_mode": self.router_mode, "status": "skipped_missing_torch"}
            return
        if len(active_models) < 2 or len(X_val) == 0:
            self.diagnostics_ = {"router_mode": self.router_mode, "status": "skipped_too_few_models"}
            return

        training_data = self._build_gating_training_data(X_val, y_val, active_models, partitioner)
        if training_data is None:
            self.diagnostics_ = {"router_mode": self.router_mode, "status": "skipped_no_features"}
            return

        device = self._resolve_gating_device()
        hidden_dim = int(self.config.get("gating_hidden_dim", 64))
        epochs = int(self.config.get("gating_epochs", 200))
        lr = float(self.config.get("gating_lr", 1e-3))
        kl_weight = float(self.config.get("gating_kl_weight", 0.10))
        balance_weight = float(self.config.get("gating_balance_weight", 0.01))
        weight_decay = float(self.config.get("gating_weight_decay", 1e-4))
        batch_size = int(self.config.get("gating_batch_size", 2048))

        self.gating_input_mean_ = training_data.features.mean(axis=0)
        self.gating_input_scale_ = training_data.features.std(axis=0) + 1e-6
        X = self._standardize_gating_features(training_data.features)

        model = _make_constrained_gating_network(
            n_features=X.shape[1],
            n_models=training_data.predictions.shape[1],
            hidden_dim=max(4, hidden_dim),
        ).to(device)
        optimizer = torch_module.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        tensors = {
            "X": torch_module.as_tensor(X, dtype=torch_module.float32, device=device),
            "prior": torch_module.as_tensor(training_data.prior, dtype=torch_module.float32, device=device),
            "pred": torch_module.as_tensor(training_data.predictions, dtype=torch_module.float32, device=device),
            "target": torch_module.as_tensor(training_data.target, dtype=torch_module.float32, device=device),
        }
        n_samples = X.shape[0]
        batch_size = max(1, min(batch_size, n_samples))
        last = {"loss": np.nan, "rmse": np.nan, "kl": np.nan, "balance": np.nan}
        with progress_bar(
            enabled=self.show_progress,
            desc="Train constrained router",
            total=max(epochs, 1),
        ) as bar:
            for epoch in range(max(epochs, 1)):
                order = torch_module.randperm(n_samples, device=device)
                for start in range(0, n_samples, batch_size):
                    idx = order[start:start + batch_size]
                    loss, parts = self._gating_loss(
                        model,
                        tensors["X"][idx],
                        tensors["prior"][idx],
                        tensors["pred"][idx],
                        tensors["target"][idx],
                        kl_weight=kl_weight,
                        balance_weight=balance_weight,
                    )
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    last = parts
                if epoch % 10 == 0 or epoch == epochs - 1:
                    try:
                        bar.set_postfix({"rmse": f"{last['rmse']:.4g}", "kl": f"{last['kl']:.4g}"})
                    except Exception:
                        pass
                bar.update(1)

        self.gating_model_ = model.eval()
        self.gating_model_names_ = [str(model_info.get("name")) for model_info in active_models]
        self.gating_device_ = device
        train_weights = self._constrained_gating_weights(X_val, training_data.prior, active_models, partitioner)
        train_pred = np.sum(training_data.predictions * train_weights, axis=1)
        prior_pred = np.sum(training_data.predictions * training_data.prior, axis=1)
        train_rmse = float(np.sqrt(np.mean((train_pred - training_data.target) ** 2)))
        prior_rmse = float(np.sqrt(np.mean((prior_pred - training_data.target) ** 2)))
        self.diagnostics_ = {
            "router_mode": self.router_mode,
            "status": "fitted",
            "training_rmse": train_rmse,
            "prior_rmse": prior_rmse,
            "rmse_delta_vs_prior": train_rmse - prior_rmse,
            "final_loss": float(last["loss"]),
            "final_kl": float(last["kl"]),
            "final_balance": float(last["balance"]),
            "n_epochs": int(max(epochs, 1)),
            "n_router_features": int(X.shape[1]),
            "device": str(device),
            "hidden_dim": int(max(4, hidden_dim)),
            "learning_rate": lr,
            "batch_size": batch_size,
            "weight_decay": weight_decay,
            "kl_weight": kl_weight,
            "balance_weight": balance_weight,
        }

    def _build_gating_training_data(
        self,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        active_models: List[Dict[str, Any]],
        partitioner: Any,
    ) -> Optional[GatingTrainingData]:
        features = self._gating_features(X_val, partitioner)
        if features is None or features.size == 0:
            return None
        prior = self.base_weights(features=X_val, active_models=active_models, partitioner=partitioner)
        predictions = np.column_stack([
            np.asarray(model_info.get("val_predictions"), dtype=float)
            for model_info in active_models
        ])
        target = pd.Series(y_val).reset_index(drop=True).to_numpy(dtype=float)
        return GatingTrainingData(
            features=np.asarray(features, dtype=np.float32),
            prior=np.asarray(prior, dtype=np.float32),
            predictions=np.asarray(predictions, dtype=np.float32),
            target=np.asarray(target, dtype=np.float32),
        )

    def _gating_features(self, features: pd.DataFrame, partitioner: Any) -> Optional[np.ndarray]:
        try:
            if hasattr(partitioner, "transform_embedding"):
                return np.asarray(partitioner.transform_embedding(features), dtype=np.float32)
        except Exception:
            return None
        numeric = features.select_dtypes(include=[np.number, "bool"]) if isinstance(features, pd.DataFrame) else None
        if numeric is not None and numeric.shape[1] > 0:
            return numeric.to_numpy(dtype=np.float32)
        return None

    @staticmethod
    def _gating_loss(
        model: Any,
        X: Any,
        prior: Any,
        predictions: Any,
        target: Any,
        *,
        kl_weight: float,
        balance_weight: float,
    ) -> tuple[Any, Dict[str, float]]:
        weights = torch.softmax(model(X), dim=1)
        y_hat = torch.sum(weights * predictions, dim=1)
        mse = torch.mean((y_hat - target) ** 2)
        kl = torch.mean(torch.sum(weights * (torch.log(weights + 1e-8) - torch.log(prior + 1e-8)), dim=1))
        uniform = torch.full_like(weights.mean(dim=0), 1.0 / weights.shape[1])
        balance = torch.mean((weights.mean(dim=0) - uniform) ** 2)
        loss = mse + kl_weight * kl + balance_weight * balance
        return loss, {
            "loss": float(loss.detach().cpu().item()),
            "rmse": float(torch.sqrt(mse.detach()).cpu().item()),
            "kl": float(kl.detach().cpu().item()),
            "balance": float(balance.detach().cpu().item()),
        }

    def _constrained_gating_weights(
        self,
        features: pd.DataFrame,
        base_weights: np.ndarray,
        active_models: List[Dict[str, Any]],
        partitioner: Any,
    ) -> np.ndarray:
        model_names = [str(model_info.get("name")) for model_info in active_models]
        if model_names != self.gating_model_names_ or self.gating_model_ is None:
            return base_weights
        X = self._gating_features(features, partitioner)
        if X is None:
            return base_weights
        X = self._standardize_gating_features(np.asarray(X, dtype=np.float32))
        try:
            with torch.no_grad():
                logits = self.gating_model_(
                    torch.as_tensor(X, dtype=torch.float32, device=self.gating_device_)
                )
                weights = torch.softmax(logits, dim=1).detach().cpu().numpy()
        except Exception:
            return base_weights
        if weights.shape != base_weights.shape:
            return base_weights
        return weights

    def _standardize_gating_features(self, features: np.ndarray) -> np.ndarray:
        if self.gating_input_mean_ is None or self.gating_input_scale_ is None:
            return features
        return (features - self.gating_input_mean_) / self.gating_input_scale_

    def _resolve_gating_device(self) -> Any:
        requested = str(self.config.get("gating_device", "auto")).strip().lower()
        if requested == "auto":
            requested = "cuda" if torch.cuda.is_available() else "cpu"
        if requested == "cuda" and not torch.cuda.is_available():
            requested = "cpu"
        return torch.device(requested)

    def _fit_legacy_learned_head(
        self,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        active_models: List[Dict[str, Any]],
        partitioner: Any,
    ) -> None:
        if len(active_models) < 2 or len(X_val) == 0:
            self.diagnostics_ = {"router_mode": self.router_mode, "status": "skipped_too_few_models"}
            return

        base_weights = self.base_weights(features=X_val, active_models=active_models, partitioner=partitioner)
        labels = self._best_validation_expert_labels(y_val, active_models)
        unique_labels = np.unique(labels)
        if unique_labels.size < 2:
            self.diagnostics_ = {
                "router_mode": self.router_mode,
                "status": "skipped_single_best_expert",
                "class_counts": self._label_counts(labels, active_models),
            }
            return

        router = RandomForestClassifier(
            n_estimators=max(1, int(self.config.get("router_n_estimators", 64))),
            max_depth=None if self.config.get("router_max_depth", 4) is None else int(self.config.get("router_max_depth", 4)),
            random_state=int(self.config.get("random_state", 42)),
            n_jobs=1,
        )
        router.fit(base_weights, labels)
        predicted = router.predict(base_weights)

        self.router_head_ = router
        self.router_head_model_names_ = [str(model_info.get("name")) for model_info in active_models]
        self.diagnostics_ = {
            "router_mode": self.router_mode,
            "status": "fitted",
            "training_accuracy": float(np.mean(predicted == labels)),
            "class_counts": self._label_counts(labels, active_models),
            "n_router_features": int(base_weights.shape[1]),
        }

    def _best_validation_expert_labels(
        self,
        y_val: pd.Series,
        active_models: List[Dict[str, Any]],
    ) -> np.ndarray:
        y_values = pd.Series(y_val).reset_index(drop=True).to_numpy()
        predictions = np.column_stack([
            np.asarray(model_info.get("val_predictions"))
            for model_info in active_models
        ])
        if self.problem == "regression":
            losses = np.abs(predictions - y_values.reshape(-1, 1))
        else:
            losses = (predictions != y_values.reshape(-1, 1)).astype(float)
        return np.argmin(losses, axis=1).astype(int)

    @staticmethod
    def _label_counts(labels: np.ndarray, active_models: List[Dict[str, Any]]) -> Dict[str, int]:
        return {
            str(model_info.get("name", f"chunk_{model_idx}")): int(np.sum(labels == model_idx))
            for model_idx, model_info in enumerate(active_models)
        }

    def _router_head_weights(
        self,
        base_weights: np.ndarray,
        active_models: List[Dict[str, Any]],
    ) -> np.ndarray:
        model_names = [str(model_info.get("name")) for model_info in active_models]
        if model_names != self.router_head_model_names_:
            return base_weights
        try:
            proba = np.asarray(self.router_head_.predict_proba(base_weights), dtype=float)
            classes = np.asarray(getattr(self.router_head_, "classes_", []), dtype=int)
        except Exception:
            return base_weights

        aligned = np.zeros_like(base_weights, dtype=float)
        for class_col, model_idx in enumerate(classes):
            if 0 <= int(model_idx) < aligned.shape[1]:
                aligned[:, int(model_idx)] = proba[:, class_col]
        row_sums = aligned.sum(axis=1, keepdims=True)
        return np.where(row_sums > 0, aligned / row_sums, base_weights)
