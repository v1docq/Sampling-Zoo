from __future__ import annotations

import json

import numpy as np
import pytest

from examples.benchmark.run_rmt_spiked_synthetic_experiment import (
    RMTSpikedExperimentConfig,
    RMTSpikedExperimentOrchestrator,
)
from sampling_zoo.core.validation.spiked_models import (
    NoiseDistribution,
    SpikedDataConfig,
    build_spiked_validation_grid,
    evaluate_rank_recovery,
    evaluate_spiked_recovery,
    evaluate_subspace_recovery,
    generate_spiked_dataset,
)


@pytest.mark.parametrize(
    "noise_distribution",
    [NoiseDistribution.GAUSSIAN, NoiseDistribution.STUDENT_T],
)
def test_spiked_generator_is_deterministic_with_exact_empirical_snr(
    noise_distribution: NoiseDistribution,
) -> None:
    config = SpikedDataConfig(
        n_samples=96,
        n_features=24,
        true_rank=3,
        snr=2.5,
        noise_distribution=noise_distribution,
        random_state=17,
    )

    first = generate_spiked_dataset(config)
    second = generate_spiked_dataset(config)

    assert np.array_equal(first.X, second.X)
    assert first.empirical_snr == pytest.approx(2.5, rel=1e-12)
    assert np.isfinite(first.standardized_empirical_snr)
    assert first.standardized_empirical_snr > 0.0
    assert np.mean(first.signal, axis=0) == pytest.approx(
        np.zeros(config.n_features),
        abs=1e-12,
    )
    assert first.true_left_subspace.T @ first.true_left_subspace == pytest.approx(
        np.eye(config.true_rank),
        abs=1e-12,
    )
    assert first.true_right_subspace.T @ first.true_right_subspace == pytest.approx(
        np.eye(config.true_rank),
        abs=1e-12,
    )
    assert first.X.flags.writeable is False


def test_rank_zero_control_contains_no_signal_and_unit_rms_noise() -> None:
    dataset = generate_spiked_dataset(
        SpikedDataConfig(
            n_samples=50,
            n_features=10,
            true_rank=0,
            snr=0.0,
            random_state=5,
        )
    )

    assert dataset.empirical_snr == 0.0
    assert dataset.standardized_empirical_snr == 0.0
    assert dataset.true_left_subspace.shape == (50, 0)
    assert np.count_nonzero(dataset.signal) == 0
    assert np.linalg.norm(dataset.noise) == pytest.approx(np.sqrt(500))


def test_spiked_validation_grid_is_stable_and_rejects_duplicate_leaf_runs() -> None:
    kwargs = {
        "snr_values": (0.1, 1.0),
        "seeds": (3, 7),
        "noise_distributions": ("gaussian",),
        "backends": ("numpy", "torch"),
        "view_strategies": ("gaussian",),
        "true_rank": 2,
        "include_null_control": True,
    }

    first = build_spiked_validation_grid(**kwargs)
    second = build_spiked_validation_grid(**kwargs)

    assert first == second
    assert len(first) == 12
    assert len({point.key for point in first}) == len(first)
    assert sum(point.true_rank == 0 for point in first) == 4

    with pytest.raises(ValueError, match="duplicate leaf runs"):
        build_spiked_validation_grid(
            **{**kwargs, "seeds": (3, 3)}
        )


def test_experiment_config_normalizes_transport_sequences_once(
    tmp_path,
) -> None:
    config = RMTSpikedExperimentConfig(
        snr_values=[0.1, 1],
        seeds=[3],
        noise_distributions="gaussian",
        backends="numpy",
        view_strategies="subsample",
        null_model_policies="feature_permutation",
        output_root=str(tmp_path),
    )

    assert config.snr_values == (0.1, 1.0)
    assert config.seeds == (3,)
    assert config.noise_distributions == ("gaussian",)
    assert config.backends == ("numpy",)
    assert config.view_strategies == ("subsample",)
    assert config.null_model_policies == ("feature_permutation",)
    assert config.output_root == tmp_path
    assert config.build_grid() == config.build_grid()


def test_subspace_recovery_is_rotation_invariant_and_penalizes_missing_rank() -> None:
    true_basis = np.eye(6)[:, :3]
    rotation, _ = np.linalg.qr(
        np.asarray(
            [
                [1.0, 2.0, 0.0],
                [0.0, 1.0, 1.0],
                [1.0, 0.0, 1.0],
            ]
        )
    )

    exact = evaluate_subspace_recovery(true_basis, true_basis @ rotation)
    under_ranked = evaluate_subspace_recovery(true_basis, true_basis[:, :2])

    assert exact is not None
    assert exact.recall == pytest.approx(1.0)
    assert exact.precision == pytest.approx(1.0)
    assert exact.max_principal_angle_degrees == pytest.approx(0.0, abs=1e-6)
    assert under_ranked is not None
    assert under_ranked.recall == pytest.approx(2.0 / 3.0)
    assert under_ranked.precision == pytest.approx(1.0)
    assert under_ranked.missed_subspace_distance == pytest.approx(
        np.sqrt(1.0 / 3.0)
    )


def test_rank_recovery_marks_null_false_positives() -> None:
    null_exact = evaluate_rank_recovery(0, 0)
    null_false_positive = evaluate_rank_recovery(0, 2)
    signal = evaluate_rank_recovery(3, 2)

    assert null_exact.exact is True
    assert null_exact.f1 == 1.0
    assert null_false_positive.false_positive is True
    assert null_false_positive.f1 == 0.0
    assert signal.recall == pytest.approx(2.0 / 3.0)
    assert signal.precision == 1.0


def test_spiked_recovery_flattens_all_sampler_rank_diagnostics() -> None:
    true_basis = np.eye(8)[:, :2]
    diagnostics = {
        "rank_by_explained_variance": 4,
        "rank_by_null_edge": 2,
        "rank_by_stability": 1,
        "rank_by_subspace_stability": None,
    }

    result = evaluate_spiked_recovery(
        true_left_subspace=true_basis,
        estimated_left_subspace=np.eye(8)[:, :4],
        sampler_diagnostics=diagnostics,
    ).to_flat_dict()

    assert result["explained_rank_absolute_error"] == 2
    assert result["null_edge_rank_exact"] is True
    assert result["view_stability_rank_recall"] == 0.5
    assert result["subspace_stability_estimated_rank"] is None
    assert result["selected_subspace_recall"] == pytest.approx(1.0)
    assert result["selected_subspace_precision"] == pytest.approx(0.5)


def test_tiny_spiked_experiment_persists_incremental_artifacts(
    tmp_path,
) -> None:
    config = RMTSpikedExperimentConfig(
        n_samples=64,
        n_features=12,
        true_rank=2,
        snr_values=(0.2, 2.0),
        seeds=(13,),
        noise_distributions=("gaussian",),
        backends=("numpy",),
        view_strategies=("gaussian",),
        include_null_control=True,
        n_views=2,
        min_views=2,
        max_views=2,
        null_model_policies=(
            "feature_permutation",
            "view_resampling",
        ),
        null_resamples=2,
        subspace_resamples=2,
        show_progress=False,
        output_root=tmp_path,
    )

    output_dir = RMTSpikedExperimentOrchestrator(config).run()

    records_path = output_dir / "metrics" / "rmt_spiked_runs.jsonl"
    records = [
        json.loads(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(records) == 3
    assert {record["status"] for record in records} == {"completed"}
    assert (output_dir / "metrics" / "rmt_spiked_raw_runs.csv").exists()
    assert (output_dir / "metrics" / "rmt_spiked_summary_by_snr.csv").exists()
    assert (output_dir / "metrics" / "rmt_spiked_null_controls.csv").exists()
    assert (output_dir / "report.md").exists()

    manifest = json.loads(
        (output_dir / "artifact_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "completed"
    assert manifest["record_count"] == 3
    assert any(
        artifact["role"] == "raw_runs"
        for artifact in manifest["artifacts"]
    )
