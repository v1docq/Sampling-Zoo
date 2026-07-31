from __future__ import annotations

import json

import numpy as np
import pytest

from examples.benchmark.run_rmt_multiregime_synthetic_experiment import (
    RMTMultiRegimeExperimentConfig,
    RMTMultiRegimeExperimentOrchestrator,
)
from sampling_zoo.core.sampling_strategies.spectral.cluster_selection import (
    SpectralClusterSelector,
)
from sampling_zoo.core.validation.multi_regime_models import (
    MultiRegimeDataConfig,
    NoiseDistribution,
    build_multi_regime_validation_grid,
    evaluate_cluster_recovery,
    generate_multi_regime_dataset,
)


@pytest.mark.parametrize(
    "noise_distribution",
    [NoiseDistribution.GAUSSIAN, NoiseDistribution.STUDENT_T],
)
def test_multi_regime_generator_is_deterministic_with_exact_empirical_snr(
    noise_distribution: NoiseDistribution,
) -> None:
    config = MultiRegimeDataConfig(
        n_samples=120,
        n_features=24,
        n_regimes=3,
        local_rank=2,
        snr=2.5,
        noise_distribution=noise_distribution,
        random_state=17,
    )

    first = generate_multi_regime_dataset(config)
    second = generate_multi_regime_dataset(config)

    assert np.array_equal(first.X, second.X)
    assert np.array_equal(first.regime_labels, second.regime_labels)
    assert first.regime_sizes == (40, 40, 40)
    assert first.empirical_snr == pytest.approx(2.5, rel=1e-12)
    assert first.true_left_subspace.shape == (120, 8)
    assert first.true_left_subspace.T @ first.true_left_subspace == pytest.approx(
        np.eye(8),
        abs=1e-12,
    )
    assert np.mean(first.signal, axis=0) == pytest.approx(
        np.zeros(config.n_features),
        abs=1e-12,
    )
    assert first.X.flags.writeable is False
    assert first.regime_labels.flags.writeable is False


def test_imbalanced_profile_has_bounded_four_to_one_regime_ratio() -> None:
    dataset = generate_multi_regime_dataset(
        MultiRegimeDataConfig(
            n_samples=401,
            n_features=24,
            n_regimes=4,
            local_rank=1,
            snr=1.0,
            regime_profile="imbalanced_4_to_1",
            random_state=5,
        )
    )

    ratio = max(dataset.regime_sizes) / min(dataset.regime_sizes)

    assert sum(dataset.regime_sizes) == 401
    assert 3.8 <= ratio <= 4.2
    assert all(size >= 3 for size in dataset.regime_sizes)


def test_cluster_recovery_is_permutation_invariant() -> None:
    true_labels = np.asarray([0, 0, 1, 1, 2, 2])
    permuted_labels = np.asarray([2, 2, 0, 0, 1, 1])

    metrics = evaluate_cluster_recovery(true_labels, permuted_labels)

    assert metrics.cluster_count_exact is True
    assert metrics.adjusted_rand_index == pytest.approx(1.0)
    assert metrics.normalized_mutual_information == pytest.approx(1.0)
    assert metrics.aligned_accuracy == pytest.approx(1.0)
    assert metrics.purity == pytest.approx(1.0)
    assert metrics.min_true_regime_recall == pytest.approx(1.0)


def test_cluster_recovery_exposes_under_partitioning() -> None:
    true_labels = np.asarray([0, 0, 1, 1, 2, 2])
    merged_labels = np.asarray([0, 0, 0, 0, 1, 1])

    metrics = evaluate_cluster_recovery(true_labels, merged_labels)

    assert metrics.predicted_n_clusters == 2
    assert metrics.cluster_count_signed_error == -1
    assert metrics.cluster_count_absolute_error == 1
    assert metrics.cluster_count_exact is False
    assert metrics.adjusted_rand_index < 1.0
    assert metrics.aligned_accuracy == pytest.approx(4.0 / 6.0)
    assert metrics.min_true_regime_recall == pytest.approx(0.0)


def test_multi_regime_grid_is_stable_and_rejects_duplicate_leaf_runs() -> None:
    kwargs = {
        "snr_values": (0.1, 1.0),
        "n_regimes_values": (2, 3),
        "seeds": (3,),
        "noise_distributions": ("gaussian",),
        "regime_profiles": ("balanced", "imbalanced_4_to_1"),
        "backends": ("numpy", "torch"),
        "view_strategies": ("gaussian",),
        "partition_policies": ("fixed_oracle", "auto_production"),
    }

    first = build_multi_regime_validation_grid(**kwargs)
    second = build_multi_regime_validation_grid(**kwargs)

    assert first == second
    assert len(first) == 32
    assert len({point.key for point in first}) == len(first)

    with pytest.raises(ValueError, match="duplicate leaf runs"):
        build_multi_regime_validation_grid(**{**kwargs, "seeds": (3, 3)})


def test_production_candidate_size_guard_is_measurable() -> None:
    production = SpectralClusterSelector(
        min_partitions=2,
        max_partitions=6,
        min_auto_partition_size=256,
        show_progress=False,
    )
    unrestricted = SpectralClusterSelector(
        min_partitions=2,
        max_partitions=6,
        min_auto_partition_size=1,
        show_progress=False,
    )

    assert production._candidate_counts(512) == [2]
    assert unrestricted._candidate_counts(512) == [2, 3, 4, 5, 6]
    assert production._candidate_counts(96) == [2, 3, 4, 5, 6]


def test_experiment_config_normalizes_sequences_once(tmp_path) -> None:
    config = RMTMultiRegimeExperimentConfig(
        n_samples=96,
        n_features=18,
        local_rank=1,
        n_regimes_values=[3],
        snr_values=[0.3, 3],
        seeds=[7],
        noise_distributions="gaussian",
        regime_profiles="balanced",
        backends="numpy",
        view_strategies="gaussian",
        partition_policies="fixed_oracle",
        cluster_algorithms="kmeans",
        output_root=str(tmp_path),
    )

    assert config.n_regimes_values == (3,)
    assert config.snr_values == (0.3, 3.0)
    assert config.noise_distributions == ("gaussian",)
    assert config.partition_policies == ("fixed_oracle",)
    assert config.cluster_algorithms == ("kmeans",)
    assert config.output_root == tmp_path
    assert config.build_grid() == config.build_grid()


def test_tiny_multi_regime_experiment_persists_incremental_artifacts(
    tmp_path,
) -> None:
    config = RMTMultiRegimeExperimentConfig(
        n_samples=72,
        n_features=18,
        local_rank=1,
        n_regimes_values=(3,),
        snr_values=(5.0,),
        seeds=(13,),
        noise_distributions=("gaussian",),
        regime_profiles=("balanced",),
        backends=("numpy",),
        view_strategies=("gaussian",),
        partition_policies=("fixed_oracle", "auto_unrestricted"),
        cluster_algorithms=("kmeans",),
        n_views=2,
        min_views=2,
        max_views=2,
        max_partitions_floor=4,
        partition_selection_sample_size=72,
        snapshot_every=1,
        show_progress=False,
        output_root=tmp_path,
    )

    output_dir = RMTMultiRegimeExperimentOrchestrator(config).run()

    records_path = output_dir / "metrics" / "rmt_multiregime_runs.jsonl"
    records = [
        json.loads(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(records) == 2
    assert {record["status"] for record in records} == {"completed"}
    assert {record["partition_policy"] for record in records} == {
        "fixed_oracle",
        "auto_unrestricted",
    }
    assert all("adjusted_rand_index" in record for record in records)
    assert all("selected_subspace_recall" in record for record in records)
    assert all("count_based_candidate_set" in record for record in records)
    assert all("density_candidate_rescued_true_n" in record for record in records)
    assert (output_dir / "metrics" / "rmt_multiregime_raw_runs.csv").exists()
    assert (output_dir / "metrics" / "rmt_multiregime_summary_by_snr.csv").exists()
    assert (output_dir / "metrics" / "rmt_multiregime_policy_regret.csv").exists()
    assert (output_dir / "report.md").exists()

    manifest = json.loads(
        (output_dir / "artifact_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "completed"
    assert manifest["record_count"] == 2
