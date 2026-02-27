from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence

import os

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import sys

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neural_network import MLPClassifier

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover - optional
    LGBMClassifier = None

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception:  # pragma: no cover - optional
    torch = None


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


def _make_pytorch_classifier(seed: int) -> ClassifierMixin:
    if torch is None:
        return MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=100, random_state=seed)

    class TorchMLPClassifier(ClassifierMixin):
        def __init__(self, input_dim: Optional[int] = None, hidden_dim: int = 128, epochs: int = 12, lr: float = 1e-3):
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.epochs = epochs
            self.lr = lr

        def _build_model(self, in_dim: int):
            return nn.Sequential(
                nn.Linear(in_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(self.hidden_dim // 2, 2),
            )

        def fit(self, X, y):
            X = np.asarray(X, dtype=np.float32)
            y = np.asarray(y, dtype=np.int64)
            torch.manual_seed(seed)
            self.model_ = self._build_model(X.shape[1])
            optimizer = optim.Adam(self.model_.parameters(), lr=self.lr)
            criterion = nn.CrossEntropyLoss()
            self.model_.train()
            x_t = torch.from_numpy(X)
            y_t = torch.from_numpy(y)
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
            with torch.no_grad():
                logits = self.model_(torch.from_numpy(X))
                probs = torch.softmax(logits, dim=1).cpu().numpy()
            return probs

        def predict(self, X):
            return np.argmax(self.predict_proba(X), axis=1)

    return TorchMLPClassifier()


def make_model_pool(seed: int = 42) -> Dict[str, ClassifierMixin]:
    models: Dict[str, ClassifierMixin] = {
        "random_forest": RandomForestClassifier(
            n_estimators=80,
            max_depth=10,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=seed,
        ),
        #"pytorch_mlp": _make_pytorch_classifier(seed),
    }

    if LGBMClassifier is not None:
        models["lightgbm"] = LGBMClassifier(
            n_estimators=220,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            verbosity=-1,
        )
    else:
        models["hist_gradient_boosting"] = HistGradientBoostingClassifier(
            max_depth=8,
            learning_rate=0.06,
            max_iter=250,
            random_state=seed,
        )

    return models