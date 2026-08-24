"""Controlled spiked-data validation for the RMT contraction spectral engine."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sys
from time import perf_counter
from typing import Any, Optional, Sequence

import numpy as np
from tqdm.auto import tqdm


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
BENCHMARK_DIR = Path(__file__).resolve().parent
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

from benchmark_incremental import IncrementalExperimentSaver  # noqa: E402
from benchmark_logging import BenchmarkLogger  # noqa: E402
from spiked_benchmark_reporting import (  # noqa: E402
    SpikedBenchmarkArtifactBuilder,
)
from sampling_zoo.core.experiment.errors import (  # noqa: E402
    UnavailableExperimentDependencyError,
)
from synthetic_benchmark_runtime import (  # noqa: E402
    create_synthetic_incremental_saver,
    resolve_backend_device,
)
from sampling_zoo.core.sampling_strategies.spectral.rmt_contraction_sampler import (  # noqa: E402
    RMTContractionConfig,
    RMTContractionTensorSampler,
)
from sampling_zoo.core.validation.spiked_models import (  # noqa: E402
    NoiseDistribution,
    SpikedDataConfig,
    SpikedDataset,
    SpikedValidationGridPoint,
    build_spiked_validation_grid,
    evaluate_spiked_recovery,
    generate_spiked_dataset,
)


DEFAULT_SPIKED_SNR_VALUES: tuple[float, ...] = (
    0.05,
    0.10,
    0.20,
    0.50,
    1.0,
    2.0,
    5.0,
    10.0,
)
DEFAULT_SPIKED_SEEDS: tuple[int, ...] = (11, 29, 47)
DEFAULT_NOISE_DISTRIBUTIONS: tuple[str, ...] = (
    NoiseDistribution.GAUSSIAN.value,
    NoiseDistribution.STUDENT_T.value,
)
DEFAULT_BACKENDS: tuple[str, ...] = ("numpy", "torch")
DEFAULT_VIEW_STRATEGIES: tuple[str, ...] = ("gaussian",)


@dataclass(frozen=True)
class RMTSpikedExperimentConfig:
    n_samples: int = 512
    n_features: int = 64
    true_rank: int = 3
    snr_values: Sequence[float] = DEFAULT_SPIKED_SNR_VALUES
    seeds: Sequence[int] = DEFAULT_SPIKED_SEEDS
    noise_distributions: Sequence[str] = DEFAULT_NOISE_DISTRIBUTIONS
    backends: Sequence[str] = DEFAULT_BACKENDS
    view_strategies: Sequence[str] = DEFAULT_VIEW_STRATEGIES
    include_null_control: bool = True
    student_t_df: float = 5.0
    min_spike_ratio: float = 0.50
    n_views: int | str = "auto"
    n_views_policy: str = "auto"
    min_views: int = 4
    max_views: int = 32
    initial_rank_fraction: float = 0.25
    explained_variance_threshold: float = 0.95
    null_model_policies: Sequence[str] = (
        "feature_permutation",
        "moment_matched_gaussian",
        "view_resampling",
    )
    null_resamples: int = 8
    null_quantile: float = 0.95
    subspace_resamples: int = 8
    device: str = "auto"
    dtype: str = "float32"
    snapshot_every: int = 10
    show_progress: bool = True
    output_root: str | Path = BENCHMARK_DIR / "results"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "snr_values",
            tuple(float(value) for value in self.snr_values),
        )
        object.__setattr__(
            self,
            "seeds",
            tuple(int(value) for value in self.seeds),
        )
        for field_name in (
            "noise_distributions",
            "backends",
            "view_strategies",
            "null_model_policies",
        ):
            raw = getattr(self, field_name)
            values = (raw,) if isinstance(raw, str) else raw
            object.__setattr__(
                self,
                field_name,
                tuple(str(value).strip().lower() for value in values),
            )
        if isinstance(self.n_views, str):
            object.__setattr__(self, "n_views", self.n_views.strip().lower())
        object.__setattr__(self, "output_root", Path(self.output_root))
        if int(self.null_resamples) < 2:
            raise ValueError("null_resamples must be at least 2")
        if int(self.subspace_resamples) < 2:
            raise ValueError("subspace_resamples must be at least 2")
        if isinstance(self.n_views, str):
            if self.n_views != "auto":
                raise ValueError("n_views must be positive or 'auto'")
        elif int(self.n_views) < 1:
            raise ValueError("n_views must be positive or 'auto'")
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be float32 or float64")
        if int(self.snapshot_every) < 1:
            raise ValueError("snapshot_every must be positive")
        self.build_grid()
        SpikedDataConfig(
            n_samples=self.n_samples,
            n_features=self.n_features,
            true_rank=self.true_rank,
            snr=float(self.snr_values[0]),
            noise_distribution=self.noise_distributions[0],
            min_spike_ratio=self.min_spike_ratio,
            student_t_df=self.student_t_df,
            random_state=int(self.seeds[0]),
        )

    def build_grid(self) -> tuple[SpikedValidationGridPoint, ...]:
        return build_spiked_validation_grid(
            snr_values=self.snr_values,
            seeds=self.seeds,
            noise_distributions=self.noise_distributions,
            backends=self.backends,
            view_strategies=self.view_strategies,
            true_rank=self.true_rank,
            include_null_control=self.include_null_control,
        )


class RMTSpikedExperimentOrchestrator:
    """Thin runtime shell for the pure spiked-data validation core."""

    def __init__(self, config: RMTSpikedExperimentConfig) -> None:
        self.config = config
        self.artifact_builder = SpikedBenchmarkArtifactBuilder()
        self.incremental_saver: Optional[IncrementalExperimentSaver] = None

    def run(self) -> Path:
        with tqdm(
            total=4,
            desc="RMT spiked validation",
            disable=not self.config.show_progress,
            unit="stage",
        ) as stages:
            logger = self._create_logger()
            stages.set_postfix_str("build deterministic grid")
            grid = self._build_grid()
            stages.update(1)

            stages.set_postfix_str("initialize incremental artifacts")
            saver = self._create_incremental_saver(logger, len(grid))
            saver.start()
            stages.update(1)

            try:
                stages.set_postfix_str("run spectral controls")
                self._run_grid(grid, saver)
                stages.update(1)
                stages.set_postfix_str("finalize report and manifest")
                saver.persist_snapshot(status="running")
                saver.finalize()
                stages.update(1)
            except Exception as exc:
                saver.mark_failed(exc)
                raise
        print(f"RMT spiked validation completed. Artifacts: {logger.paths.root}")
        return logger.paths.root

    def _create_logger(self) -> BenchmarkLogger:
        run_id = (
            "run_rmt_spiked_synthetic_" f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        return BenchmarkLogger(
            run_id=run_id,
            artifacts_root=self.config.output_root,
        )

    def _build_grid(self) -> tuple[SpikedValidationGridPoint, ...]:
        return self.config.build_grid()

    def _create_incremental_saver(
        self,
        logger: BenchmarkLogger,
        grid_size: int,
    ) -> IncrementalExperimentSaver:
        saver = create_synthetic_incremental_saver(
            logger=logger,
            config=self.config,
            grid_size=grid_size,
            experiment_name="rmt_spiked_synthetic_validation",
            records_filename="rmt_spiked_runs.jsonl",
            artifact_builder=self.artifact_builder,
            snapshot_every=self.config.snapshot_every,
            repo_root=ROOT_DIR,
        )
        self.incremental_saver = saver
        return saver

    def _run_grid(
        self,
        grid: Sequence[SpikedValidationGridPoint],
        saver: IncrementalExperimentSaver,
    ) -> None:
        for point in tqdm(
            grid,
            desc="Spiked SNR grid",
            disable=not self.config.show_progress,
            leave=False,
            unit="run",
        ):
            try:
                record = self._run_grid_point(point)
            except Exception as exc:
                record = self._failed_record(point, exc)
            saver.record(record)

    def _run_grid_point(
        self,
        point: SpikedValidationGridPoint,
    ) -> dict[str, Any]:
        device = self._resolve_backend_device(point)
        dataset = self._generate_dataset(point)
        sampler = RMTContractionTensorSampler(self._make_sampler_config(point, device))
        started = perf_counter()
        sampler.fit(dataset.X)
        fit_time = perf_counter() - started
        if sampler.left_basis_ is None:
            raise RuntimeError("RMT sampler did not expose a fitted left basis")
        evaluation = evaluate_spiked_recovery(
            true_left_subspace=dataset.true_left_subspace,
            estimated_left_subspace=sampler.left_basis_,
            sampler_diagnostics=sampler.diagnostics_,
        )
        return {
            **point.to_dict(),
            **dataset.summary(),
            **evaluation.to_flat_dict(),
            "status": "completed",
            "device": device if point.backend == "torch" else None,
            "fit_time_sec": float(fit_time),
            "initial_rank": sampler.diagnostics_.get("initial_rank"),
            "selected_n_views": sampler.diagnostics_.get("n_views"),
            "null_model_status": sampler.diagnostics_.get("null_model_status"),
            "subspace_stability_status": sampler.diagnostics_.get(
                "subspace_stability_status"
            ),
            "diagnostics": sampler.diagnostics_,
            "extra": {
                "problem_type": "spectral_validation",
                "split_label": "synthetic_control",
                "grid_key": point.key,
            },
        }

    def _generate_dataset(
        self,
        point: SpikedValidationGridPoint,
    ) -> SpikedDataset:
        return generate_spiked_dataset(
            SpikedDataConfig(
                n_samples=self.config.n_samples,
                n_features=self.config.n_features,
                true_rank=point.true_rank,
                snr=point.snr,
                noise_distribution=point.noise_distribution,
                min_spike_ratio=self.config.min_spike_ratio,
                student_t_df=self.config.student_t_df,
                random_state=point.seed,
            )
        )

    def _make_sampler_config(
        self,
        point: SpikedValidationGridPoint,
        device: str,
    ) -> RMTContractionConfig:
        return RMTContractionConfig(
            n_partitions=1,
            partition_selection_method="fixed",
            n_views=self.config.n_views,
            n_views_policy=self.config.n_views_policy,
            min_views=self.config.min_views,
            max_views=self.config.max_views,
            view_strategy=point.view_strategy,
            initial_rank_fraction=self.config.initial_rank_fraction,
            explained_variance_threshold=(self.config.explained_variance_threshold),
            null_diagnostic_enabled=True,
            null_model_policies=tuple(self.config.null_model_policies),
            null_resamples=self.config.null_resamples,
            null_quantile=self.config.null_quantile,
            subspace_diagnostic_enabled=True,
            subspace_resamples=self.config.subspace_resamples,
            backend=point.backend,
            device=device,
            dtype=self.config.dtype,
            selection_method="all",
            max_encoded_features=None,
            show_progress=self.config.show_progress,
            random_state=point.seed,
        )

    def _resolve_backend_device(
        self,
        point: SpikedValidationGridPoint,
    ) -> str:
        return resolve_backend_device(
            backend=point.backend,
            requested_device=self.config.device,
            error_scope="rmt_spiked.backend",
        )

    @staticmethod
    def _failed_record(
        point: SpikedValidationGridPoint,
        error: BaseException,
    ) -> dict[str, Any]:
        unavailable = isinstance(
            error,
            UnavailableExperimentDependencyError,
        )
        error_code = (
            error.code
            if isinstance(error, UnavailableExperimentDependencyError)
            else error.__class__.__name__
        )
        return {
            **point.to_dict(),
            "dataset": (
                f"spiked_{point.noise_distribution.value}" f"_rank_{point.true_rank}"
            ),
            "status": "skipped" if unavailable else "failed",
            "error_code": error_code,
            "error": str(error),
            "error_details": (
                dict(error.details)
                if isinstance(error, UnavailableExperimentDependencyError)
                else {}
            ),
            "diagnostics": {},
            "extra": {
                "problem_type": "spectral_validation",
                "split_label": "synthetic_control",
                "grid_key": point.key,
            },
        }


def run_rmt_spiked_synthetic_experiment(
    *,
    snr_values: Sequence[float] = DEFAULT_SPIKED_SNR_VALUES,
    seeds: Sequence[int] = DEFAULT_SPIKED_SEEDS,
    noise_distributions: Sequence[str] = DEFAULT_NOISE_DISTRIBUTIONS,
    backends: Sequence[str] = DEFAULT_BACKENDS,
    view_strategies: Sequence[str] = DEFAULT_VIEW_STRATEGIES,
    n_samples: int = 512,
    n_features: int = 64,
    true_rank: int = 3,
    show_progress: bool = True,
    output_root: str | Path = BENCHMARK_DIR / "results",
) -> Path:
    config = RMTSpikedExperimentConfig(
        n_samples=n_samples,
        n_features=n_features,
        true_rank=true_rank,
        snr_values=tuple(snr_values),
        seeds=tuple(seeds),
        noise_distributions=tuple(noise_distributions),
        backends=tuple(backends),
        view_strategies=tuple(view_strategies),
        show_progress=show_progress,
        output_root=output_root,
    )
    return RMTSpikedExperimentOrchestrator(config).run()


def make_server_spiked_config(
    *,
    output_root: str | Path = BENCHMARK_DIR / "results",
    show_progress: bool = True,
) -> RMTSpikedExperimentConfig:
    """Return the 20-SNR, 10-seed validation grid from the RMT roadmap."""

    return RMTSpikedExperimentConfig(
        snr_values=tuple(map(float, np.geomspace(0.03, 30.0, 20))),
        seeds=tuple(range(10)),
        backends=("numpy", "torch"),
        noise_distributions=("gaussian", "student_t"),
        null_resamples=16,
        subspace_resamples=16,
        show_progress=show_progress,
        output_root=output_root,
    )


def make_smoke_spiked_config(
    *,
    output_root: str | Path = BENCHMARK_DIR / "results",
    show_progress: bool = True,
) -> RMTSpikedExperimentConfig:
    return RMTSpikedExperimentConfig(
        n_samples=96,
        n_features=16,
        true_rank=2,
        snr_values=(0.2, 2.0),
        seeds=(7,),
        noise_distributions=("gaussian",),
        backends=("numpy",),
        n_views=2,
        min_views=2,
        max_views=2,
        null_model_policies=(
            "feature_permutation",
            "view_resampling",
        ),
        null_resamples=2,
        subspace_resamples=2,
        snapshot_every=1,
        show_progress=show_progress,
        output_root=output_root,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run controlled RMT spiked synthetic validation."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--smoke",
        action="store_true",
        help="Run a tiny local validation grid.",
    )
    mode.add_argument(
        "--server-grid",
        action="store_true",
        help="Run the 20-SNR, 10-seed roadmap grid.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=BENCHMARK_DIR / "results",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars.",
    )
    return parser.parse_args()


def _main() -> None:
    args = _parse_args()
    common = {
        "output_root": args.output_root,
        "show_progress": not args.no_progress,
    }
    if args.server_grid:
        config = make_server_spiked_config(**common)
    elif args.smoke:
        config = make_smoke_spiked_config(**common)
    else:
        config = RMTSpikedExperimentConfig(**common)
    RMTSpikedExperimentOrchestrator(config).run()


if __name__ == "__main__":
    _main()
