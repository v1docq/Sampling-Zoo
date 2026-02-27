import importlib.util
from pathlib import Path

import numpy as np
import sys


BASE_DIR = Path(__file__).resolve().parents[1] / "examples" / "benchmark"
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


def _load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, BASE_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


run_mod = _load_module("run_260226_mod", "run_260226.py")
datasets_mod = _load_module("benchmark_datasets_mod", "benchmark_datasets.py")


def test_apply_data_budget_trim_and_pad():
    trimmed, trim_meta = run_mod._apply_data_budget(
        informative_indices=np.array([0, 1, 2, 3, 4]),
        train_size=10,
        target_percent=20,
        seed=42,
        sample_scores=[0.1, 0.9, 0.8, 0.2, 0.7],
    )
    assert trimmed.tolist() == [1, 2]
    assert trim_meta["budget_action"] == "trim_top_k"

    padded, pad_meta = run_mod._apply_data_budget(
        informative_indices=np.array([0]),
        train_size=10,
        target_percent=30,
        seed=42,
    )
    assert len(padded) == 3
    assert 0 in padded.tolist()
    assert pad_meta["budget_action"] == "pad_with_random"


def test_resolve_dataset_names_with_amlb_categories():
    resolved = datasets_mod.resolve_dataset_names(
        dataset_names=["mixed_hard"],
        include_amlb=True,
        amlb_categories=["small_samples_many_classes"],
    )
    assert "mixed_hard" in resolved
    assert "amlb_anneal" in resolved
    assert "amlb_car" in resolved
