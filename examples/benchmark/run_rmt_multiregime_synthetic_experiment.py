"""Controlled multi-regime validation of RMT partition selection."""

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
from multiregime_benchmark_reporting import (  # noqa: E402
    MultiRegimeBenchmarkArtifactBuilder,
)
from sampling_zoo.core.experiment.errors import (  # noqa: E402
    UnavailableExperimentDependencyError,
)
from sampling_zoo.core.sampling_strategies.spectral.rmt_contraction_sampler import (  # noqa: E402
    RMTContractionConfig,
    RMTContractionTensorSampler,
)
from sampling_zoo.core.validation.multi_regime_models import (  # noqa: E402
    MultiRegimeDataConfig,
    MultiRegimeDataset,
    MultiRegimeValidationGridPoint,
    NoiseDistribution,
    PartitionProbePolicy,
    build_multi_regime_validation_grid,
    evaluate_cluster_recovery,
    generate_multi_regime_dataset,
)
from sampling_zoo.core.validation.spiked_models import (  # noqa: E402
    evaluate_subspace_recovery,
)
from synthetic_benchmark_runtime import (  # noqa: E402
    create_synthetic_incremental_saver,
    resolve_backend_device,
)


DEFAULT_MULTIREGIME_SNR_VALUES: tuple[float, ...] = (0.1, 0.3, 1.0, 3.0, 10.0)
DEFAULT_MULTIREGIME_SEEDS: tuple[int, ...] = (11, 29, 47)
DEFAULT_N_REGIMES: tuple[int, ...] = (3,)
DEFAULT_NOISE_DISTRIBUTIONS: tuple[str, ...] = (
    NoiseDistribution.GAUSSIAN.value,
    NoiseDistribution.STUDENT_T.value,
)
DEFAULT_REGIME_PROFILES: tuple[str, ...] = (
    "balanced",
    "imbalanced_4_to_1",
)
DEFAULT_BACKENDS: tuple[str, ...] = ("numpy", "torch")
DEFAULT_VIEW_STRATEGIES: tuple[str, ...] = ("gaussian",)
DEFAULT_PARTITION_POLICIES: tuple[str, ...] = tuple(
    policy.value for policy in PartitionProbePolicy
)
DEFAULT_CLUSTER_ALGORITHMS: tuple[str, ...] = (
    "kmeans",
    "bisecting_kmeans",
    "gmm",
    "hdbscan",
)


@dataclass(frozen=True)
class RMTMultiRegimeExperimentConfig:
    """Typed grid and sampler controls for cluster-recovery validation."""

    n_samples: int = 512
    n_features: int = 64
    local_rank: int = 2
    n_regimes_values: Sequence[int] = DEFAULT_N_REGIMES
    snr_values: Sequence[float] = DEFAULT_MULTIREGIME_SNR_VALUES
    seeds: Sequence[int] = DEFAULT_MULTIREGIME_SEEDS
    noise_distributions: Sequence[str] = DEFAULT_NOISE_DISTRIBUTIONS
    regime_profiles: Sequence[str] = DEFAULT_REGIME_PROFILES
    backends: Sequence[str] = DEFAULT_BACKENDS
    view_strategies: Sequence[str] = DEFAULT_VIEW_STRATEGIES
    partition_policies: Sequence[str] = DEFAULT_PARTITION_POLICIES
    cluster_algorithms: Sequence[str] = DEFAULT_CLUSTER_ALGORITHMS
    regime_separation: float = 2.0
    local_signal_scale: float = 0.50
    student_t_df: float = 5.0
    max_partitions_floor: int = 6
    production_min_auto_partition_size: int = 256
    unrestricted_min_auto_partition_size: int = 1
    partition_selection_sample_size: int = 5000
    max_cluster_imbalance_ratio: float = 5.0
    min_cluster_fraction: float = 0.05
    n_views: int | str = "auto"
    n_views_policy: str = "auto"
    min_views: int = 4
    max_views: int = 32
    initial_rank_fraction: float = 0.25
    explained_variance_threshold: float = 0.95
    device: str = "auto"
    dtype: str = "float32"
    snapshot_every: int = 10
    show_progress: bool = True
    output_root: str | Path = BENCHMARK_DIR / "results"

    def __post_init__(self) -> None:
        self._normalize_numeric_sequences()
        self._normalize_string_sequences()
        if isinstance(self.n_views, str):
            object.__setattr__(self, "n_views", self.n_views.strip().lower())
        object.__setattr__(self, "output_root", Path(self.output_root))
        self._validate_runtime_controls()
        self.build_grid()
        self._validate_data_configs()

    def _normalize_numeric_sequences(self) -> None:
        object.__setattr__(
            self,
            "n_regimes_values",
            tuple(int(value) for value in self.n_regimes_values),
        )
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

    def _normalize_string_sequences(self) -> None:
        for field_name in (
            "noise_distributions",
            "regime_profiles",
            "backends",
            "view_strategies",
            "partition_policies",
            "cluster_algorithms",
        ):
            raw = getattr(self, field_name)
            values = (raw,) if isinstance(raw, str) else raw
            object.__setattr__(
                self,
                field_name,
                tuple(str(value).strip().lower() for value in values),
            )

    def _validate_runtime_controls(self) -> None:
        if isinstance(self.n_views, str):
            if self.n_views != "auto":
                raise ValueError("n_views must be positive or 'auto'")
        elif int(self.n_views) < 1:
            raise ValueError("n_views must be positive or 'auto'")
        if int(self.max_partitions_floor) < 2:
            raise ValueError("max_partitions_floor must be at least 2")
        if int(self.production_min_auto_partition_size) < 1:
            raise ValueError("production_min_auto_partition_size must be positive")
        if int(self.unrestricted_min_auto_partition_size) < 1:
            raise ValueError("unrestricted_min_auto_partition_size must be positive")
        if int(self.snapshot_every) < 1:
            raise ValueError("snapshot_every must be positive")
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be float32 or float64")

    def _validate_data_configs(self) -> None:
        for n_regimes in self.n_regimes_values:
            MultiRegimeDataConfig(
                n_samples=self.n_samples,
                n_features=self.n_features,
                n_regimes=n_regimes,
                local_rank=self.local_rank,
                snr=float(self.snr_values[0]),
                regime_profile=self.regime_profiles[0],
                noise_distribution=self.noise_distributions[0],
                regime_separation=self.regime_separation,
                local_signal_scale=self.local_signal_scale,
                student_t_df=self.student_t_df,
                random_state=int(self.seeds[0]),
            )

    def build_grid(self) -> tuple[MultiRegimeValidationGridPoint, ...]:
        return build_multi_regime_validation_grid(
            snr_values=self.snr_values,
            n_regimes_values=self.n_regimes_values,
            seeds=self.seeds,
            noise_distributions=self.noise_distributions,
            regime_profiles=self.regime_profiles,
            backends=self.backends,
            view_strategies=self.view_strategies,
            partition_policies=self.partition_policies,
        )


class RMTMultiRegimeExperimentOrchestrator:
    """Thin effectful shell around deterministic multi-regime controls."""

    def __init__(self, config: RMTMultiRegimeExperimentConfig) -> None:
        self.config = config
        self.artifact_builder = MultiRegimeBenchmarkArtifactBuilder()
        self.incremental_saver: Optional[IncrementalExperimentSaver] = None

    def run(self) -> Path:
        with tqdm(
            total=4,
            desc="RMT multi-regime validation",
            disable=not self.config.show_progress,
            unit="stage",
        ) as stages:
            logger = self._create_logger()
            stages.set_postfix_str("build deterministic grid")
            grid = self.config.build_grid()
            stages.update(1)

            stages.set_postfix_str("initialize incremental artifacts")
            saver = self._create_incremental_saver(logger, len(grid))
            saver.start()
            stages.update(1)

            try:
                stages.set_postfix_str("run cluster-recovery controls")
                self._run_grid(grid, saver)
                stages.update(1)
                stages.set_postfix_str("finalize report and manifest")
                saver.persist_snapshot(status="running")
                saver.finalize()
                stages.update(1)
            except Exception as exc:
                saver.mark_failed(exc)
                raise
        print(f"RMT multi-regime validation completed. Artifacts: {logger.paths.root}")
        return logger.paths.root

    def _create_logger(self) -> BenchmarkLogger:
        run_id = (
            "run_rmt_multiregime_synthetic_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        return BenchmarkLogger(
            run_id=run_id,
            artifacts_root=self.config.output_root,
        )

    def _create_incremental_saver(
        self,
        logger: BenchmarkLogger,
        grid_size: int,
    ) -> IncrementalExperimentSaver:
        saver = create_synthetic_incremental_saver(
            logger=logger,
            config=self.config,
            grid_size=grid_size,
            experiment_name="rmt_multiregime_synthetic_validation",
            records_filename="rmt_multiregime_runs.jsonl",
            artifact_builder=self.artifact_builder,
            snapshot_every=self.config.snapshot_every,
            repo_root=ROOT_DIR,
        )
        self.incremental_saver = saver
        return saver

    def _run_grid(
        self,
        grid: Sequence[MultiRegimeValidationGridPoint],
        saver: IncrementalExperimentSaver,
    ) -> None:
        for point in tqdm(
            grid,
            desc="Multi-regime grid",
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
        point: MultiRegimeValidationGridPoint,
    ) -> dict[str, Any]:
        device = resolve_backend_device(
            backend=point.backend,
            requested_device=self.config.device,
            error_scope="rmt_multiregime.backend",
        )
        dataset = self._generate_dataset(point)
        sampler = RMTContractionTensorSampler(self._make_sampler_config(point, device))
        started = perf_counter()
        sampler.fit(dataset.X)
        fit_time = perf_counter() - started
        return self._completed_record(
            point=point,
            dataset=dataset,
            sampler=sampler,
            device=device,
            fit_time=fit_time,
        )

    def _generate_dataset(
        self,
        point: MultiRegimeValidationGridPoint,
    ) -> MultiRegimeDataset:
        return generate_multi_regime_dataset(
            MultiRegimeDataConfig(
                n_samples=self.config.n_samples,
                n_features=self.config.n_features,
                n_regimes=point.n_regimes,
                local_rank=self.config.local_rank,
                snr=point.snr,
                regime_profile=point.regime_profile,
                noise_distribution=point.noise_distribution,
                regime_separation=self.config.regime_separation,
                local_signal_scale=self.config.local_signal_scale,
                student_t_df=self.config.student_t_df,
                random_state=point.seed,
            )
        )

    def _make_sampler_config(
        self,
        point: MultiRegimeValidationGridPoint,
        device: str,
    ) -> RMTContractionConfig:
        policy = point.partition_policy
        fixed = policy is PartitionProbePolicy.FIXED_ORACLE
        min_auto_size = (
            self.config.production_min_auto_partition_size
            if policy is PartitionProbePolicy.AUTO_PRODUCTION
            else self.config.unrestricted_min_auto_partition_size
        )
        return RMTContractionConfig(
            n_partitions=point.n_regimes,
            partition_selection_method="fixed" if fixed else "auto",
            cluster_algorithms=("kmeans",)
            if fixed
            else tuple(self.config.cluster_algorithms),
            cluster_selection_metric="balanced_silhouette",
            cluster_ensemble_method="weighted_vote",
            min_partitions=2,
            max_partitions=max(
                int(self.config.max_partitions_floor),
                int(point.n_regimes) + 2,
            ),
            min_auto_partition_size=int(min_auto_size),
            partition_selection_sample_size=(
                self.config.partition_selection_sample_size
            ),
            max_cluster_imbalance_ratio=(self.config.max_cluster_imbalance_ratio),
            min_cluster_fraction=self.config.min_cluster_fraction,
            n_views=self.config.n_views,
            n_views_policy=self.config.n_views_policy,
            min_views=self.config.min_views,
            max_views=self.config.max_views,
            view_strategy=point.view_strategy,
            embedding_mode="sv_scaled",
            initial_rank_fraction=self.config.initial_rank_fraction,
            explained_variance_threshold=(self.config.explained_variance_threshold),
            null_diagnostic_enabled=False,
            subspace_diagnostic_enabled=False,
            selection_method="all",
            chunk_fraction=1.0,
            chunks_percent=100.0,
            max_encoded_features=None,
            backend=point.backend,
            device=device,
            dtype=self.config.dtype,
            show_progress=self.config.show_progress,
            random_state=point.seed,
        )

    @staticmethod
    def _completed_record(
        *,
        point: MultiRegimeValidationGridPoint,
        dataset: MultiRegimeDataset,
        sampler: RMTContractionTensorSampler,
        device: str,
        fit_time: float,
    ) -> dict[str, Any]:
        if sampler.cluster_labels_ is None:
            raise RuntimeError("RMT sampler did not expose fitted cluster labels")
        if sampler.left_basis_ is None:
            raise RuntimeError("RMT sampler did not expose a fitted left basis")
        cluster_metrics = evaluate_cluster_recovery(
            dataset.regime_labels,
            sampler.cluster_labels_,
        )
        subspace_metrics = evaluate_subspace_recovery(
            dataset.true_left_subspace,
            sampler.left_basis_,
        )
        diagnostics = sampler.diagnostics_
        candidates = tuple(
            int(value)
            for value in diagnostics.get("partition_selection_candidates", ())
        )
        candidate_details = diagnostics.get(
            "partition_selection_candidate_details",
            (),
        )
        candidate_plan = diagnostics.get(
            "partition_selection_candidate_plan",
            {},
        )
        candidate_failures = diagnostics.get(
            "partition_selection_candidate_failures",
            (),
        )
        count_based_candidates = (
            candidates
            if point.partition_policy is PartitionProbePolicy.FIXED_ORACLE
            else tuple(
                sorted(
                    {
                        int(candidate["n_clusters"])
                        for candidate in candidate_details
                        if candidate.get("algorithm") != "hdbscan"
                    }
                )
            )
        )
        density_candidates = tuple(
            sorted(
                {
                    int(candidate["n_clusters"])
                    for candidate in candidate_details
                    if candidate.get("algorithm") == "hdbscan"
                }
            )
        )
        planned_count_candidates = tuple(
            int(value)
            for value in candidate_plan.get(
                "eligible_count_candidates",
                count_based_candidates,
            )
        )
        rejected_counts = {
            int(rejection["n_clusters"])
            for rejection in candidate_plan.get("size_guard_rejections", ())
        }
        selected_k = int(cluster_metrics.predicted_n_clusters)
        return {
            **point.to_dict(),
            **dataset.summary(),
            **cluster_metrics.to_dict(),
            **(subspace_metrics.to_dict() if subspace_metrics is not None else {}),
            "status": "completed",
            "device": device if point.backend == "torch" else None,
            "fit_time_sec": float(fit_time),
            "initial_rank": diagnostics.get("initial_rank"),
            "selected_rank": diagnostics.get("selected_rank"),
            "selected_n_views": diagnostics.get("n_views"),
            "selected_cluster_algorithm": diagnostics.get("selected_cluster_algorithm"),
            "candidate_set": list(candidates),
            "candidate_set_contains_true_n": point.n_regimes in candidates,
            "count_based_candidate_set": list(count_based_candidates),
            "count_based_candidate_set_contains_true_n": (
                point.n_regimes in count_based_candidates
            ),
            "planned_count_candidate_set": list(planned_count_candidates),
            "planned_count_candidate_set_contains_true_n": (
                point.n_regimes in planned_count_candidates
            ),
            "size_guard_rejected_true_n": point.n_regimes in rejected_counts,
            "size_guard_fallback_applied": bool(
                candidate_plan.get("size_guard_fallback_applied", False)
            ),
            "density_candidate_set": list(density_candidates),
            "density_candidate_rescued_true_n": (
                point.n_regimes not in count_based_candidates
                and point.n_regimes in density_candidates
            ),
            "candidate_failure_count": len(candidate_failures),
            "candidate_failure_codes": sorted(
                {str(failure.get("code")) for failure in candidate_failures}
            ),
            "selected_candidate_valid": _selected_candidate_value(
                diagnostics,
                selected_k,
                "valid",
            ),
            "selected_candidate_score": _selected_candidate_value(
                diagnostics,
                selected_k,
                "score",
            ),
            "diagnostics": diagnostics,
            "extra": {
                "problem_type": "cluster_validation",
                "split_label": "synthetic_control",
                "grid_key": point.key,
            },
        }

    @staticmethod
    def _failed_record(
        point: MultiRegimeValidationGridPoint,
        error: BaseException,
    ) -> dict[str, Any]:
        unavailable = isinstance(error, UnavailableExperimentDependencyError)
        return {
            **point.to_dict(),
            "dataset": (
                f"multi_regime_{point.noise_distribution.value}"
                f"_{point.regime_profile.value}_k{point.n_regimes}"
            ),
            "status": "skipped" if unavailable else "failed",
            "error_code": (error.code if unavailable else error.__class__.__name__),
            "error": str(error),
            "error_details": dict(error.details) if unavailable else {},
            "diagnostics": {},
            "extra": {
                "problem_type": "cluster_validation",
                "split_label": "synthetic_control",
                "grid_key": point.key,
            },
        }


def _selected_candidate_value(
    diagnostics: dict[str, Any],
    selected_k: int,
    key: str,
) -> Any:
    candidates = diagnostics.get("partition_selection_candidate_details", ())
    algorithm = diagnostics.get("selected_cluster_algorithm")
    matched = [
        candidate
        for candidate in candidates
        if int(candidate.get("n_clusters", -1)) == selected_k
        and candidate.get("algorithm") == algorithm
    ]
    if not matched:
        return None
    return matched[0].get(key)


def run_rmt_multiregime_synthetic_experiment(
    *,
    snr_values: Sequence[float] = DEFAULT_MULTIREGIME_SNR_VALUES,
    n_regimes_values: Sequence[int] = DEFAULT_N_REGIMES,
    seeds: Sequence[int] = DEFAULT_MULTIREGIME_SEEDS,
    noise_distributions: Sequence[str] = DEFAULT_NOISE_DISTRIBUTIONS,
    regime_profiles: Sequence[str] = DEFAULT_REGIME_PROFILES,
    backends: Sequence[str] = DEFAULT_BACKENDS,
    view_strategies: Sequence[str] = DEFAULT_VIEW_STRATEGIES,
    partition_policies: Sequence[str] = DEFAULT_PARTITION_POLICIES,
    n_samples: int = 512,
    n_features: int = 64,
    local_rank: int = 2,
    show_progress: bool = True,
    output_root: str | Path = BENCHMARK_DIR / "results",
) -> Path:
    config = RMTMultiRegimeExperimentConfig(
        n_samples=n_samples,
        n_features=n_features,
        local_rank=local_rank,
        n_regimes_values=tuple(n_regimes_values),
        snr_values=tuple(snr_values),
        seeds=tuple(seeds),
        noise_distributions=tuple(noise_distributions),
        regime_profiles=tuple(regime_profiles),
        backends=tuple(backends),
        view_strategies=tuple(view_strategies),
        partition_policies=tuple(partition_policies),
        show_progress=show_progress,
        output_root=output_root,
    )
    return RMTMultiRegimeExperimentOrchestrator(config).run()


def make_server_multiregime_config(
    *,
    output_root: str | Path = BENCHMARK_DIR / "results",
    show_progress: bool = True,
) -> RMTMultiRegimeExperimentConfig:
    return RMTMultiRegimeExperimentConfig(
        n_regimes_values=(2, 3, 5),
        snr_values=tuple(map(float, np.geomspace(0.03, 30.0, 16))),
        seeds=tuple(range(10)),
        noise_distributions=("gaussian", "student_t"),
        regime_profiles=("balanced", "imbalanced_4_to_1"),
        backends=("numpy", "torch"),
        partition_policies=DEFAULT_PARTITION_POLICIES,
        snapshot_every=20,
        show_progress=show_progress,
        output_root=output_root,
    )


def make_smoke_multiregime_config(
    *,
    output_root: str | Path = BENCHMARK_DIR / "results",
    show_progress: bool = True,
) -> RMTMultiRegimeExperimentConfig:
    return RMTMultiRegimeExperimentConfig(
        n_samples=96,
        n_features=18,
        local_rank=1,
        n_regimes_values=(3,),
        snr_values=(0.3, 5.0),
        seeds=(7,),
        noise_distributions=("gaussian",),
        regime_profiles=("balanced",),
        backends=("numpy",),
        partition_policies=DEFAULT_PARTITION_POLICIES,
        cluster_algorithms=("kmeans", "gmm"),
        n_views=2,
        min_views=2,
        max_views=2,
        max_partitions_floor=5,
        partition_selection_sample_size=96,
        snapshot_every=1,
        show_progress=show_progress,
        output_root=output_root,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run controlled RMT multi-regime cluster validation."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--server-grid", action="store_true")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=BENCHMARK_DIR / "results",
    )
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def _main() -> None:
    args = _parse_args()
    common = {
        "output_root": args.output_root,
        "show_progress": not args.no_progress,
    }
    if args.server_grid:
        config = make_server_multiregime_config(**common)
    elif args.smoke:
        config = make_smoke_multiregime_config(**common)
    else:
        config = RMTMultiRegimeExperimentConfig(**common)
    RMTMultiRegimeExperimentOrchestrator(config).run()


if __name__ == "__main__":
    _main()
