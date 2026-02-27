from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples" / "benchmark"))

MODULE_PATH = Path(__file__).resolve().parents[1] / "examples" / "benchmark" / "run_260226.py"
spec = importlib.util.spec_from_file_location("run_260226", MODULE_PATH)
run_module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(run_module)


_apply_budget_policy = run_module._apply_budget_policy
resolve_datasets = run_module.resolve_datasets


def test_apply_budget_policy_top_up_when_informative_is_small() -> None:
    result = _apply_budget_policy(
        informative_indices=[0, 3],
        informative_scores=[0.9, 0.1],
        train_size=20,
        budget_ratio=0.2,
        seed=7,
    )

    assert result["budget_size"] == 4
    assert result["informative_size"] == 2
    assert result["sample_indices"].shape[0] == 4
    assert {0, 3}.issubset(set(result["sample_indices"].tolist()))


def test_apply_budget_policy_top_k_when_informative_is_large() -> None:
    result = _apply_budget_policy(
        informative_indices=[1, 2, 5, 9],
        informative_scores=[0.2, 0.8, 0.1, 0.6],
        train_size=10,
        budget_ratio=0.2,
        seed=1,
    )

    assert result["budget_size"] == 2
    assert set(result["sample_indices"].tolist()) == {2, 9}
    assert result["policy_action"] == "truncate_to_top_k"


def test_resolve_datasets_includes_amlb_categories_without_duplicates() -> None:
    datasets = resolve_datasets(
        full_benchmark=False,
        include_amlb=True,
        amlb_categories=["small_samples_many_classes", "large_samples_binary"],
    )

    assert "mixed_hard" in datasets
    assert "amlb_adult" in datasets
    assert "amlb_optdigits" in datasets
    assert len(datasets) == len(np.unique(datasets))
