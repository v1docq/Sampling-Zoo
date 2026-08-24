from pathlib import Path
from types import SimpleNamespace

import pytest

from examples.benchmark import benchmark_datasets


def test_openml_task_cache_dir_uses_resolved_modern_cache_directory(
    monkeypatch,
    tmp_path: Path,
) -> None:
    resolved_cache = tmp_path / "org" / "openml" / "www"
    monkeypatch.setattr(
        benchmark_datasets.openml.config,
        "get_cache_directory",
        lambda: str(resolved_cache),
    )

    assert benchmark_datasets._openml_task_cache_dir(233211) == (
        resolved_cache / "tasks" / "233211"
    )


def test_registered_medium_tasks_skip_suite_wide_listing(monkeypatch) -> None:
    calls: list[tuple[int, int, str]] = []

    def fake_load_suite_dataset(
        task_id: int,
        suite_id: int,
        task_name: str,
    ) -> SimpleNamespace:
        calls.append((task_id, suite_id, task_name))
        return SimpleNamespace(
            problem_type="regression",
            metadata=SimpleNamespace(n_objects=100, n_features=10),
        )

    monkeypatch.setattr(
        benchmark_datasets,
        "load_suite_dataset",
        fake_load_suite_dataset,
    )
    monkeypatch.setattr(
        benchmark_datasets.openml.study,
        "get_suite",
        lambda _suite_id: pytest.fail("registered tasks must skip get_suite"),
    )

    bundles = benchmark_datasets._load_suite_group(
        suite_id=269,
        requested_task_names=("diamonds", "pol"),
        expected_problem_type="regression",
        show_progress=False,
    )

    assert len(bundles) == 2
    assert calls == [
        (233211, 269, "diamonds"),
        (359946, 269, "pol"),
    ]
