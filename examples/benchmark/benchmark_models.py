from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
from scipy import sparse
from sklearn.base import ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neural_network import MLPClassifier

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from examples.benchmark.benchmark_model_profiles import (  # noqa: E402
    TabPFNFinetuneConfig,
    TabPFNModelProfile,
    normalize_tabpfn_finetune_config,
)

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except Exception:  # pragma: no cover - optional
    LGBMClassifier = None
    LGBMRegressor = None

torch = None
nn = None
optim = None
_TORCH_IMPORT_ATTEMPTED = False

TabPFNClassifier = None
TabPFNRegressor = None
ModelVersion = None
_TABPFN_IMPORT_ATTEMPTED = False

FinetunedTabPFNClassifier = None
FinetunedTabPFNRegressor = None
_TABPFN_FINETUNING_IMPORT_ATTEMPTED = False

TabICLClassifier = None
TabICLRegressor = None
_TABICL_IMPORT_ATTEMPTED = False

@dataclass
class SearchResult:
    best_params: Dict[str, Any]
    best_score: float


def _to_dense(matrix: np.ndarray | sparse.spmatrix) -> np.ndarray:
    return matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)


def _sample_from_partitions(partitions: Mapping[str, np.ndarray], n_train: int, sample_ratio: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    picks: list[int] = []
    for _, indices in partitions.items():
        idx = np.asarray(indices, dtype=int)
        if idx.size == 0:
            continue
        take = min(idx.size, max(1, int(round(idx.size * sample_ratio))))
        picks.extend(rng.choice(idx, size=take, replace=False).tolist())

    if not picks:
        fallback_take = max(10, int(n_train * sample_ratio))
        picks = rng.choice(np.arange(n_train), size=min(fallback_take, n_train), replace=False).tolist()

    return np.unique(np.asarray(picks, dtype=int))


def _build_inner_split(y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    train_idx, valid_idx = next(splitter.split(np.zeros_like(y), y))
    return train_idx, valid_idx


def _score_sample_indices(sample_indices: Sequence[int], y_train: np.ndarray, val_idx: np.ndarray) -> float:
    sample_idx = np.asarray(sample_indices, dtype=int)
    if sample_idx.size == 0:
        return -1.0
    sample_set = set(sample_idx.tolist())
    overlap = np.array([idx for idx in val_idx if idx in sample_set], dtype=int)
    coverage = len(sample_set) / max(len(y_train), 1)
    class_balance = len(np.unique(y_train[sample_idx])) / max(len(np.unique(y_train)), 1)
    overlap_penalty = 0.25 if overlap.size > 0 else 0.0
    return 0.6 * class_balance + 0.4 * coverage - overlap_penalty
def _search_params(
    candidates: Iterable[Dict[str, Any]],
    sampler_factory: Callable[[Dict[str, Any]], tuple[Sequence[int], Dict[str, Any]]],
    y_train: np.ndarray,
    seed: int,
) -> SearchResult:
    _, valid_idx = _build_inner_split(y_train, seed)
    best_score = -np.inf
    best_params: Dict[str, Any] = {}
    for params in candidates:
        indices, extra = sampler_factory(params)
        score = _score_sample_indices(indices, y_train, valid_idx)
        if score > best_score:
            best_score = score
            best_params = {**params, **extra}
    return SearchResult(best_params=best_params, best_score=float(best_score))


def _load_torch_modules() -> tuple[Any, Any, Any]:
    global torch, nn, optim, _TORCH_IMPORT_ATTEMPTED
    if torch is not None:
        return torch, nn, optim
    if not _TORCH_IMPORT_ATTEMPTED:
        _TORCH_IMPORT_ATTEMPTED = True
        try:
            import torch as _torch
            import torch.nn as _nn
            import torch.optim as _optim
        except Exception:  # pragma: no cover - optional
            torch = None
            nn = None
            optim = None
        else:
            torch = _torch
            nn = _nn
            optim = _optim
    return torch, nn, optim


def _make_pytorch_classifier(seed: int) -> ClassifierMixin:
    torch_module, nn_module, optim_module = _load_torch_modules()
    if torch_module is None or nn_module is None or optim_module is None:
        return MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=100, random_state=seed)

    class TorchMLPClassifier(ClassifierMixin):
        def __init__(self, input_dim: Optional[int] = None, hidden_dim: int = 128, epochs: int = 12, lr: float = 1e-3):
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.epochs = epochs
            self.lr = lr

        def _build_model(self, in_dim: int):
            return nn_module.Sequential(
                nn_module.Linear(in_dim, self.hidden_dim),
                nn_module.ReLU(),
                nn_module.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn_module.ReLU(),
                nn_module.Linear(self.hidden_dim // 2, 2),
            )

        def fit(self, X, y):
            X = np.asarray(X, dtype=np.float32)
            y = np.asarray(y, dtype=np.int64)
            torch_module.manual_seed(seed)
            self.model_ = self._build_model(X.shape[1])
            optimizer = optim_module.Adam(self.model_.parameters(), lr=self.lr)
            criterion = nn_module.CrossEntropyLoss()
            self.model_.train()
            x_t = torch_module.from_numpy(X)
            y_t = torch_module.from_numpy(y)
            for _ in range(self.epochs):
                optimizer.zero_grad()
                logits = self.model_(x_t)
                loss = criterion(logits, y_t)
                loss.backward()
                optimizer.step()
            return self

        def predict_proba(self, X):
            X = np.asarray(X, dtype=np.float32)
            self.model_.eval()
            with torch_module.no_grad():
                logits = self.model_(torch_module.from_numpy(X))
                probs = torch_module.softmax(logits, dim=1).cpu().numpy()
            return probs

        def predict(self, X):
            return np.argmax(self.predict_proba(X), axis=1)

    return TorchMLPClassifier()


def _env_flag_enabled(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _torch_cuda_is_available() -> bool:
    torch_module, _, _ = _load_torch_modules()
    if torch_module is None:
        return False
    try:
        return bool(torch_module.cuda.is_available())
    except Exception:
        return False


def _resolve_tabpfn_device() -> str:
    """Prefer CUDA for TabPFN when the active torch backend can see it."""
    explicit_device = os.getenv("TABPFN_DEVICE")
    if explicit_device:
        return explicit_device.strip()
    return "cuda" if _torch_cuda_is_available() else "cpu"


def _prepare_tabpfn_runtime() -> None:
    """Keep benchmark runs independent from optional telemetry endpoints."""
    os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")


def _load_tabpfn_classes() -> tuple[Any, Any, Any]:
    global ModelVersion, TabPFNClassifier, TabPFNRegressor, _TABPFN_IMPORT_ATTEMPTED
    _prepare_tabpfn_runtime()
    if not _TABPFN_IMPORT_ATTEMPTED:
        _TABPFN_IMPORT_ATTEMPTED = True
        try:
            from tabpfn import TabPFNClassifier as _TabPFNClassifier
            from tabpfn import TabPFNRegressor as _TabPFNRegressor
            from tabpfn.constants import ModelVersion as _ModelVersion
        except Exception:  # pragma: no cover - optional
            TabPFNClassifier = None
            TabPFNRegressor = None
            ModelVersion = None
        else:
            TabPFNClassifier = _TabPFNClassifier
            TabPFNRegressor = _TabPFNRegressor
            ModelVersion = _ModelVersion
    return TabPFNClassifier, TabPFNRegressor, ModelVersion


def _load_tabpfn_finetuning_classes() -> tuple[Any, Any]:
    global FinetunedTabPFNClassifier
    global FinetunedTabPFNRegressor
    global _TABPFN_FINETUNING_IMPORT_ATTEMPTED

    _prepare_tabpfn_runtime()
    if not _TABPFN_FINETUNING_IMPORT_ATTEMPTED:
        _TABPFN_FINETUNING_IMPORT_ATTEMPTED = True
        try:
            from tabpfn.finetuning import (
                FinetunedTabPFNClassifier as _FinetunedTabPFNClassifier,
            )
            from tabpfn.finetuning import (
                FinetunedTabPFNRegressor as _FinetunedTabPFNRegressor,
            )
        except Exception:  # pragma: no cover - optional
            FinetunedTabPFNClassifier = None
            FinetunedTabPFNRegressor = None
        else:
            FinetunedTabPFNClassifier = _FinetunedTabPFNClassifier
            FinetunedTabPFNRegressor = _FinetunedTabPFNRegressor
    return FinetunedTabPFNClassifier, FinetunedTabPFNRegressor


def _load_tabicl_classes() -> tuple[Any, Any]:
    global TabICLClassifier, TabICLRegressor, _TABICL_IMPORT_ATTEMPTED
    if not _TABICL_IMPORT_ATTEMPTED:
        _TABICL_IMPORT_ATTEMPTED = True
        try:
            from tabicl import TabICLClassifier as _TabICLClassifier
            from tabicl import TabICLRegressor as _TabICLRegressor
        except Exception:  # pragma: no cover - optional
            TabICLClassifier = None
            TabICLRegressor = None
        else:
            TabICLClassifier = _TabICLClassifier
            TabICLRegressor = _TabICLRegressor
    return TabICLClassifier, TabICLRegressor


def _make_tabpfn_kwargs(seed: int, n_estimators: int = 12) -> Dict[str, Any]:
    device = _resolve_tabpfn_device()
    kwargs: Dict[str, Any] = {
        "n_estimators": n_estimators,
        "device": device,
        "random_state": seed,
    }
    if device == "cpu" and (
        _env_flag_enabled("TABPFN_ALLOW_CPU_LARGE_DATASET")
        or _env_flag_enabled("TABPFN_IGNORE_PRETRAINING_LIMITS")
    ):
        kwargs["ignore_pretraining_limits"] = True
    return kwargs


def _create_tabpfn_model(model_cls: Any, seed: int) -> Any:
    _, _, model_version = _load_tabpfn_classes()
    if model_version is None:
        raise ValueError("tabpfn is not available. Install tabpfn to use this model.")
    return model_cls.create_default_for_version(
        model_version.V2_5,
        **_make_tabpfn_kwargs(seed),
    )


def _create_finetuned_tabpfn_model(
    model_cls: Any,
    seed: int,
    config: TabPFNFinetuneConfig,
) -> Any:
    device = _resolve_tabpfn_device()
    if config.require_cuda and not device.lower().startswith("cuda"):
        raise ValueError(
            "tabpfn_finetuned requires a CUDA device by default. "
            "Install a CUDA-enabled torch build or explicitly set "
            "require_cuda=False for a small CPU smoke run."
        )
    return model_cls(**config.model_kwargs(device=device, seed=seed))




def make_model_pool(
    seed: int = 42,
    model_names: Optional[Sequence[str]] = None,
    problem_type: Optional[str] = None,
    n_jobs: int | None = None,
    tabpfn_finetune_config: (
        TabPFNFinetuneConfig | Mapping[str, Any] | None
    ) = None,
) -> Dict[str, Any]:
    """Build model factories, optionally limiting parallel estimator workers.

    ``n_jobs=None`` preserves the historical per-model defaults. Passing an
    explicit value applies the same worker limit to parallel sklearn and
    LightGBM estimators.
    """
    if n_jobs is not None:
        if (
            isinstance(n_jobs, bool)
            or not isinstance(n_jobs, Integral)
            or n_jobs == 0
            or n_jobs < -1
        ):
            raise ValueError("n_jobs must be None, -1, or a positive integer")
        n_jobs = int(n_jobs)

    available_names = {
        "random_forest",
        "lightgbm",
        "hist_gradient_boosting",
        "ridge",
        "tabpfn",
        TabPFNModelProfile.IN_CONTEXT.value,
        TabPFNModelProfile.FINETUNED.value,
        "tabicl",
    }
    if model_names is None:
        requested = {"random_forest", "lightgbm"}
    else:
        requested = {name.strip().lower() for name in model_names}

    unknown = requested - available_names
    if unknown:
        raise ValueError(f"Unsupported model(s): {sorted(unknown)}")

    model_pool: Dict[str, Any] = {}
    normalized_problem = problem_type.strip().lower() if problem_type else "classification"

    if "random_forest" in requested:
        random_forest_n_jobs = -1 if n_jobs is None else n_jobs
        if normalized_problem == "regression":
            model_pool["random_forest"] = lambda: RandomForestRegressor(
                n_estimators=80,
                max_depth=10,
                min_samples_leaf=2,
                n_jobs=random_forest_n_jobs,
                random_state=seed,
            )
        else:
            model_pool["random_forest"] = lambda: RandomForestClassifier(
                n_estimators=80,
                max_depth=10,
                min_samples_leaf=2,
                n_jobs=random_forest_n_jobs,
                random_state=seed,
            )

    if "lightgbm" in requested:
        if LGBMClassifier is None or LGBMRegressor is None:
            raise ValueError("lightgbm is not available. Install lightgbm or choose another model.")
        lightgbm_parallel_kwargs = {} if n_jobs is None else {"n_jobs": n_jobs}
        if normalized_problem == "regression":
            model_pool["lightgbm"] = lambda: LGBMRegressor(
                random_state=seed,
                verbosity=-1,
                **lightgbm_parallel_kwargs,
            )
        else:
            model_pool["lightgbm"] = lambda: LGBMClassifier(
                random_state=seed,
                verbosity=-1,
                **lightgbm_parallel_kwargs,
            )

    if "hist_gradient_boosting" in requested:
        if normalized_problem == "regression":
            model_pool["hist_gradient_boosting"] = lambda: HistGradientBoostingRegressor(
                max_depth=8,
                learning_rate=0.06,
                max_iter=250,
                random_state=seed,
            )
        else:
            model_pool["hist_gradient_boosting"] = lambda: HistGradientBoostingClassifier(
                max_depth=8,
                learning_rate=0.06,
                max_iter=250,
                random_state=seed,
            )

    if "ridge" in requested:
        if normalized_problem == "regression":
            model_pool["ridge"] = lambda: Ridge()
        else:
            model_pool["ridge"] = lambda: LogisticRegression(max_iter=500, random_state=seed)

    tabpfn_in_context_names = requested.intersection(
        {"tabpfn", TabPFNModelProfile.IN_CONTEXT.value}
    )
    if tabpfn_in_context_names:
        tabpfn_classifier, tabpfn_regressor, _model_version = _load_tabpfn_classes()
        if tabpfn_classifier is None or tabpfn_regressor is None:
            raise ValueError("tabpfn is not available. Install tabpfn to use this model.")
        model_cls = (
            tabpfn_classifier
            if normalized_problem == "classification"
            else tabpfn_regressor
        )
        for model_name in sorted(tabpfn_in_context_names):
            model_pool[model_name] = (
                lambda model_cls=model_cls: _create_tabpfn_model(
                    model_cls,
                    seed,
                )
            )

    if TabPFNModelProfile.FINETUNED.value in requested:
        finetuned_classifier, finetuned_regressor = (
            _load_tabpfn_finetuning_classes()
        )
        if finetuned_classifier is None or finetuned_regressor is None:
            raise ValueError(
                "TabPFN fine-tuning is not available. Install a TabPFN "
                "version that exposes tabpfn.finetuning."
            )
        finetune_config = normalize_tabpfn_finetune_config(
            tabpfn_finetune_config
        )
        finetuned_model_cls = (
            finetuned_classifier
            if normalized_problem == "classification"
            else finetuned_regressor
        )
        model_pool[TabPFNModelProfile.FINETUNED.value] = (
            lambda: _create_finetuned_tabpfn_model(
                finetuned_model_cls,
                seed,
                finetune_config,
            )
        )

    if "tabicl" in requested:
        tabicl_classifier, tabicl_regressor = _load_tabicl_classes()
        if tabicl_classifier is None or tabicl_regressor is None:
            raise ValueError("tabicl is not available. Install tabicl to use this model.")
        if normalized_problem == "classification":
            model_pool["tabicl"] = lambda: tabicl_classifier(n_estimators=12)
        else:
            model_pool["tabicl"] = lambda: tabicl_regressor(n_estimators=12)

    return model_pool
