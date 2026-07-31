from __future__ import annotations

import pytest

from examples.benchmark import benchmark_models


class _FakeCuda:
    def __init__(self, available: bool) -> None:
        self.available = available

    def is_available(self) -> bool:
        return self.available


class _FakeTorch:
    def __init__(self, cuda_available: bool) -> None:
        self.cuda = _FakeCuda(cuda_available)


class _FakeTabPFN:
    calls: list[dict] = []

    @classmethod
    def create_default_for_version(cls, version, **kwargs):
        cls.calls.append({"version": version, "kwargs": kwargs})
        return kwargs


class _FakeModelVersion:
    V2_5 = "v2.5"


def test_tabpfn_device_prefers_cuda_when_torch_can_use_it(monkeypatch) -> None:
    monkeypatch.delenv("TABPFN_DEVICE", raising=False)
    monkeypatch.setattr(benchmark_models, "torch", _FakeTorch(cuda_available=True))

    assert benchmark_models._resolve_tabpfn_device() == "cuda"


def test_tabpfn_device_respects_explicit_env_override(monkeypatch) -> None:
    monkeypatch.setenv("TABPFN_DEVICE", "cpu")
    monkeypatch.setattr(benchmark_models, "torch", _FakeTorch(cuda_available=True))

    assert benchmark_models._resolve_tabpfn_device() == "cpu"


def test_tabpfn_factory_passes_resolved_device(monkeypatch) -> None:
    _FakeTabPFN.calls = []
    monkeypatch.delenv("TABPFN_DEVICE", raising=False)
    monkeypatch.setattr(benchmark_models, "torch", _FakeTorch(cuda_available=True))
    monkeypatch.setattr(benchmark_models, "_TABPFN_IMPORT_ATTEMPTED", True)
    monkeypatch.setattr(benchmark_models, "TabPFNClassifier", _FakeTabPFN)
    monkeypatch.setattr(benchmark_models, "TabPFNRegressor", _FakeTabPFN)
    monkeypatch.setattr(benchmark_models, "ModelVersion", _FakeModelVersion)

    model_pool = benchmark_models.make_model_pool(
        seed=17,
        model_names=("tabpfn",),
        problem_type="regression",
    )
    model_kwargs = model_pool["tabpfn"]()

    assert model_kwargs["device"] == "cuda"
    assert model_kwargs["random_state"] == 17
    assert _FakeTabPFN.calls[-1]["kwargs"] == model_kwargs


def test_tabpfn_cpu_limit_override_is_opt_in(monkeypatch) -> None:
    monkeypatch.delenv("TABPFN_ALLOW_CPU_LARGE_DATASET", raising=False)
    monkeypatch.delenv("TABPFN_IGNORE_PRETRAINING_LIMITS", raising=False)
    monkeypatch.setattr(benchmark_models, "torch", _FakeTorch(cuda_available=False))

    assert "ignore_pretraining_limits" not in benchmark_models._make_tabpfn_kwargs(seed=1)

    monkeypatch.setenv("TABPFN_ALLOW_CPU_LARGE_DATASET", "1")
    assert benchmark_models._make_tabpfn_kwargs(seed=1)["ignore_pretraining_limits"] is True


def test_model_pool_preserves_legacy_random_forest_parallelism() -> None:
    model = benchmark_models.make_model_pool(
        model_names=("random_forest",),
        problem_type="regression",
    )["random_forest"]()

    assert model.n_jobs == -1


def test_model_pool_preserves_legacy_lightgbm_parallelism(monkeypatch) -> None:
    class FakeLightGBM:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    monkeypatch.setattr(benchmark_models, "LGBMClassifier", FakeLightGBM)
    monkeypatch.setattr(benchmark_models, "LGBMRegressor", FakeLightGBM)

    model = benchmark_models.make_model_pool(
        model_names=("lightgbm",),
        problem_type="regression",
    )["lightgbm"]()

    assert "n_jobs" not in model.kwargs


def test_model_pool_applies_explicit_worker_limit_to_parallel_models(
    monkeypatch,
) -> None:
    class FakeLightGBM:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    monkeypatch.setattr(benchmark_models, "LGBMClassifier", FakeLightGBM)
    monkeypatch.setattr(benchmark_models, "LGBMRegressor", FakeLightGBM)

    model_pool = benchmark_models.make_model_pool(
        seed=17,
        model_names=("random_forest", "lightgbm"),
        problem_type="regression",
        n_jobs=3,
    )

    assert model_pool["random_forest"]().n_jobs == 3
    assert model_pool["lightgbm"]().kwargs["n_jobs"] == 3


@pytest.mark.parametrize("n_jobs", [0, -2, True, 1.5, "2"])
def test_model_pool_rejects_invalid_worker_limits(n_jobs) -> None:
    with pytest.raises(ValueError, match="n_jobs"):
        benchmark_models.make_model_pool(
            model_names=("ridge",),
            problem_type="regression",
            n_jobs=n_jobs,
        )
