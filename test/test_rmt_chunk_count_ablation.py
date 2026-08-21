from __future__ import annotations

import json
import sys

import pytest

from examples.benchmark import rmt_chunk_count_ablation_medium_datasets as chunk_module

from examples.benchmark.rmt_chunk_count_ablation_medium_datasets import (
    ChunkCountAblationReportBuilder,
    RMTChunkCountAblationConfig,
    RMTChunkCountAblationMediumOrchestrator,
    make_chunk_count_ablation_strategy_configs,
)
from benchmark_runner import EnsembleFoldBenchmarkExecutor
from sampling_zoo.core.utils.sampling_ensemble import SamplingEnsemble


def test_grid_forces_each_k_and_keeps_kmeans_gmm_selection() -> None:
    config = RMTChunkCountAblationConfig(
        models=("lightgbm",),
        budget_ratios=(0.1, 0.3),
        chunk_counts=(2, 4, 8, 16),
        show_progress=False,
    )

    configs = make_chunk_count_ablation_strategy_configs(config)

    assert len(configs) == 16
    assert "full_dataset" not in configs
    assert {value["strategy"] for value in configs.values()} == {
        "rmt_contraction"
    }
    assert {value["n_partitions"] for value in configs.values()} == {
        2,
        4,
        8,
        16,
    }
    for name, strategy_config in configs.items():
        requested_k = strategy_config["n_partitions"]
        assert f"__k_{requested_k:02d}" in name
        assert strategy_config["min_partitions"] == requested_k
        assert strategy_config["max_partitions"] == requested_k
        assert strategy_config["partition_selection_method"] == "auto"
        assert strategy_config["cluster_algorithms"] == ["kmeans", "gmm"]
        assert strategy_config["cluster_ensemble_method"] == "best_score"
        assert strategy_config["chunks_percent"] == 100.0
        if strategy_config["ensemble_method"] == "routed_weighted":
            assert strategy_config["router"] == "spectral"
        else:
            assert "router" not in strategy_config
            assert "validation_pruning" not in strategy_config


def test_grid_can_add_matched_voting_no_pruning_control() -> None:
    config = RMTChunkCountAblationConfig(
        models=("lightgbm",),
        budget_ratios=(0.3,),
        chunk_counts=(2, 4),
        voting_pruning_modes=(True, False),
        show_progress=False,
    )

    configs = make_chunk_count_ablation_strategy_configs(config)

    assert len(configs) == 6
    voting = [
        value for value in configs.values()
        if value["ensemble_method"] == "voting"
    ]
    assert {
        value.get("validation_pruning", True) for value in voting
    } == {True, False}


def test_quick_foundation_cli_resolves_small_matched_profile(
    monkeypatch,
    tmp_path,
) -> None:
    captured = {}
    monkeypatch.setattr(
        chunk_module,
        "run_rmt_chunk_count_ablation_medium",
        lambda **kwargs: captured.update(kwargs) or tmp_path,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rmt_chunk_count_ablation_medium_datasets.py",
            "--quick-foundation",
            "--output-dir",
            str(tmp_path),
        ],
    )

    chunk_module._run_cli(chunk_module._parse_args())

    assert tuple(captured["regression_tasks"]) == (
        "Brazilian_houses",
        "pol",
        "elevators",
        "house_16H",
        "house_sales",
        "OnlineNewsPopularity",
        "diamonds",
    )
    assert tuple(captured["models"]) == ("tabpfn", "tabicl")
    assert tuple(captured["budget_ratios"]) == (0.3,)
    assert tuple(captured["chunk_counts"]) == (2, 4)
    assert tuple(captured["voting_pruning_modes"]) == (True, False)


def test_config_requires_k2_reference_and_rejects_duplicate_k() -> None:
    with pytest.raises(ValueError, match="k=2"):
        RMTChunkCountAblationConfig(chunk_counts=(4, 8), show_progress=False)
    with pytest.raises(ValueError, match="duplicates"):
        RMTChunkCountAblationConfig(chunk_counts=(2, 4, 4), show_progress=False)


def test_runner_respects_explicit_all_chunks_override() -> None:
    runner = object.__new__(EnsembleFoldBenchmarkExecutor)

    default_plan = runner._build_execution_plan(
        {"n_partitions": 16, "force_chunking": True},
        train_size=10_000,
    )
    ablation_plan = runner._build_execution_plan(
        {
            "n_partitions": 16,
            "force_chunking": True,
            "chunks_percent": 100.0,
        },
        train_size=10_000,
    )

    assert default_plan.chunks_percent == pytest.approx(62.5)
    assert ablation_plan.chunks_percent == pytest.approx(100.0)


def _record(
    *,
    k: int,
    rmse: float,
    ensemble: str = "routed_weighted",
    selected_rows: int = 100,
    active_experts: int | None = None,
    score: float | None = None,
    algorithm: str = "kmeans",
    pruning: bool = True,
    validation_before: float | None = None,
    validation_after: float | None = None,
) -> dict:
    return {
        "dataset": "medium_demo",
        "strategy_params": {
            "strategy": "rmt_contraction",
            "model": "lightgbm",
            "split_label": "split_1",
            "budget_ratio": 0.1,
            "ensemble_method": ensemble,
            "n_partitions": k,
            **({"validation_pruning": pruning} if ensemble == "voting" else {}),
            **({"router": "spectral"} if ensemble == "routed_weighted" else {}),
        },
        "model_metrics": {"rmse": rmse},
        "timings_sec": {"fit": 2.0, "inference": 0.2},
        "sample_stats": {
            "selected_rows": selected_rows,
            "selected_partition_count": k,
            "model_fit_rows_total": selected_rows,
            "active_model_rows": selected_rows,
            "active_model_count": (
                k if active_experts is None else active_experts
            ),
        },
        "extra": {
            "seed": 42,
            "runtime_diagnostics": {"partitioning": {"total": 1.0}},
            "sampler_diagnostics": {
                "selected_n_partitions": k,
                "selected_cluster_algorithm": algorithm,
                "post_budget_chunk_sizes": {
                    f"chunk_{index}": selected_rows // k
                    for index in range(k)
                },
                "partition_selection_selected_candidate": {
                    "score": score,
                    "valid": True,
                    "components": {
                        "counts": [selected_rows // k] * k,
                        "imbalance_ratio": 1.0,
                        "min_cluster_fraction": 1.0 / k,
                        "constraint_violations": [],
                    },
                },
            },
            "cache_usage": {
                "structure": "miss",
                "base_partitions": "miss",
                "trained_models": "miss",
            },
            "validation_diagnostics": {
                "selection_policy": (
                    "forward_pruning" if pruning else "all_experts_no_pruning"
                ),
                "full_ensemble_metrics_before_pruning": {
                    "rmse": validation_before
                },
                "ensemble_metrics_after_pruning": {
                    "rmse": validation_after
                },
            },
        },
    }


def test_report_finds_downstream_best_k_and_silhouette_disagreement(tmp_path) -> None:
    records = [
        _record(k=2, rmse=10.0, score=0.60),
        _record(k=4, rmse=9.0, score=0.55, algorithm="gmm"),
        _record(k=8, rmse=9.5, score=0.50),
        _record(k=16, rmse=11.0, score=0.45),
    ]

    tables = ChunkCountAblationReportBuilder().build(records, tmp_path)

    compared_k4 = tables["comparisons"].query("requested_k == 4").iloc[0]
    assert bool(compared_k4["row_budget_matches_k2"])
    assert compared_k4["rmse_change_vs_k2_pct"] == pytest.approx(-10.0)
    assert compared_k4["outcome_vs_k2"] == "better"
    best = tables["best_k"].iloc[0]
    assert best["best_downstream_k"] == 4
    assert best["best_improvement_vs_k2_pct"] == pytest.approx(10.0)
    assert best["silhouette_best_k"] == 2
    assert not bool(best["silhouette_matches_downstream"])
    assert (tmp_path / "chunk_count_vs_k2.csv").exists()
    assert (tmp_path / "chunk_count_best_k.csv").exists()
    assert (tmp_path / "chunk_count_ablation_report.md").exists()


def test_report_tracks_voting_pruning_separately_from_created_chunks(tmp_path) -> None:
    tables = ChunkCountAblationReportBuilder().build(
        [
            _record(k=2, rmse=10.0, ensemble="voting", active_experts=1, score=0.5),
            _record(k=4, rmse=9.9, ensemble="voting", active_experts=2, score=0.4),
        ],
        tmp_path,
    )

    run_k4 = tables["runs"].query("requested_k == 4").iloc[0]
    assert run_k4["selected_partition_count"] == 4
    assert run_k4["active_expert_count"] == 2
    assert run_k4["active_fraction_of_requested"] == pytest.approx(0.5)
    assert run_k4["min_expert_train_rows"] == 25
    assert bool(run_k4["foundation_min_rows_compatible"])


def test_report_builds_matched_voting_pruning_effect(tmp_path) -> None:
    tables = ChunkCountAblationReportBuilder().build(
        [
            _record(
                k=2,
                rmse=10.0,
                ensemble="voting",
                active_experts=1,
                pruning=True,
                validation_before=12.0,
                validation_after=10.0,
            ),
            _record(
                k=2,
                rmse=12.5,
                ensemble="voting",
                active_experts=2,
                pruning=False,
                validation_before=12.0,
                validation_after=12.0,
            ),
        ],
        tmp_path,
    )

    effect = tables["pruning_effect"].iloc[0]
    assert bool(effect["pair_complete"])
    assert effect["experts_removed"] == 1
    assert effect["pruning_rmse_change_vs_all_experts_pct"] == pytest.approx(-20.0)
    assert effect["validation_rmse_all_experts"] == pytest.approx(12.0)
    assert effect["validation_rmse_pruned"] == pytest.approx(10.0)
    assert (tmp_path / "chunk_count_pruning_effect.csv").exists()


def test_integrity_rejects_routed_expert_loss(tmp_path) -> None:
    config = RMTChunkCountAblationConfig(
        regression_tasks=("medium_demo",),
        models=("lightgbm",),
        budget_ratios=(0.1,),
        chunk_counts=(2, 4),
        ensemble_methods=("routed_weighted",),
        show_progress=False,
    )
    tables = ChunkCountAblationReportBuilder().build(
        [
            _record(k=2, rmse=10.0, active_experts=2, score=0.5),
            _record(k=4, rmse=9.0, active_experts=3, score=0.4),
        ],
        tmp_path,
    )

    with pytest.raises(RuntimeError, match="wrong_routed_active_k=1"):
        RMTChunkCountAblationMediumOrchestrator(
            config
        )._validate_completed_ablation(tables)


def test_protocol_quantifies_structure_and_expert_fit_work() -> None:
    config = RMTChunkCountAblationConfig(
        models=("lightgbm",),
        budget_ratios=(0.1, 0.3),
        chunk_counts=(2, 4, 8, 16),
        show_progress=False,
    )

    protocol = RMTChunkCountAblationMediumOrchestrator(
        config
    )._comparison_protocol()

    assert protocol["leaf_runs_per_dataset_model"] == 16
    assert protocol["expensive_structure_fits_per_dataset_split"] == 4
    assert protocol["trained_expert_fits_per_dataset_model"] == 60
    json.dumps(protocol)


def test_protocol_counts_pruning_control_without_extra_expert_fits() -> None:
    config = RMTChunkCountAblationConfig(
        models=("lightgbm",),
        budget_ratios=(0.1, 0.3),
        chunk_counts=(2, 4, 8, 16),
        voting_pruning_modes=(True, False),
        show_progress=False,
    )

    protocol = RMTChunkCountAblationMediumOrchestrator(
        config
    )._comparison_protocol()

    assert protocol["leaf_runs_per_dataset_model"] == 24
    assert protocol["trained_expert_fits_per_dataset_model"] == 60


def test_voting_can_keep_all_experts_when_validation_pruning_is_disabled(
    monkeypatch,
) -> None:
    ensemble = SamplingEnsemble(
        problem="regression",
        partitioner_config={
            "strategy": "rmt_contraction",
            "validation_pruning": False,
        },
        model_factory=lambda: object(),
        ensemble_method="voting",
        show_progress=False,
    )
    ensemble.models = [
        {"name": f"chunk_{index}", "data_size": 10}
        for index in range(4)
    ]
    monkeypatch.setattr(
        ensemble,
        "_evaluate_current_ensemble",
        lambda X_val, y_val: {"rmse": 1.25},
    )
    monkeypatch.setattr(
        ensemble,
        "_build_validation_diagnostics",
        lambda **kwargs: kwargs,
    )

    ensemble._finalize_partition_training(
        partitions={},
        X_val=None,
        y_val=None,
        metric_is_better=lambda current, best: current < best,
        validation_metric="rmse",
    )

    assert len(ensemble.models) == 4
    assert ensemble.validation_diagnostics_["selection_policy"] == (
        "all_experts_no_pruning"
    )


def test_routed_local_calibration_is_rejected_when_validation_worsens(
    monkeypatch,
) -> None:
    ensemble = SamplingEnsemble(
        problem="regression",
        partitioner_config={"strategy": "rmt_contraction", "router": "spectral"},
        model_factory=lambda: object(),
        ensemble_method="routed_weighted",
        show_progress=False,
    )
    ensemble.models = [
        {
            "name": "chunk_0",
            "metrics": {"rmse": 1.0},
            "val_predictions": [0.0, 0.0],
        },
        {
            "name": "chunk_1",
            "metrics": {"rmse": 2.0},
            "val_predictions": [1.0, 1.0],
        },
    ]
    evaluations = iter(({"rmse": 1.0}, {"rmse": 5.0}))
    monkeypatch.setattr(
        ensemble,
        "_evaluate_current_ensemble",
        lambda X_val, y_val: next(evaluations),
    )
    monkeypatch.setattr(
        ensemble,
        "_build_local_validation_metrics",
        lambda *args, **kwargs: {
            "chunk_0": {"assigned_count": 1, "metrics": {"rmse": 10.0}},
            "chunk_1": {"assigned_count": 1, "metrics": {"rmse": 20.0}},
        },
    )
    monkeypatch.setattr(
        ensemble,
        "_run_routing_refinement",
        lambda **kwargs: {"mode": "none", "status": "disabled"},
    )
    captured = {}
    monkeypatch.setattr(
        ensemble,
        "_build_validation_diagnostics",
        lambda **kwargs: captured.update(kwargs) or kwargs,
    )
    monkeypatch.setattr(ensemble, "_log_active_chunk_summary", lambda *args: None)

    ensemble._finalize_moe_partition_training(
        partitions={},
        X_val=None,
        y_val=None,
        metric_is_better=lambda current, best: current < best,
        validation_metric="rmse",
    )

    assert captured["reduced_metrics"]["rmse"] == 1.0
    assert captured["best_score"] == 1.0
    assert (
        captured["local_calibration_diagnostics"]["status"]
        == "rejected_no_validation_improvement"
    )
    assert all("local_metrics" not in model for model in ensemble.models)
