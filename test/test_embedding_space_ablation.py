from __future__ import annotations

import json
from pathlib import Path

import pytest

from examples.benchmark.embedding_space_ablation_medium_datasets import (
    EmbeddingAblationReportBuilder,
    EmbeddingSpaceAblationConfig,
    EmbeddingSpaceAblationMediumOrchestrator,
    make_embedding_ablation_strategy_configs,
)
from sampling_zoo.core.utils.sampling_ensemble import SamplingEnsemble


def test_controlled_grid_is_matched_and_omits_direct_baseline() -> None:
    config = EmbeddingSpaceAblationConfig(
        models=("lightgbm",),
        budget_ratios=(0.1, 0.5, 0.9),
        show_progress=False,
    )

    configs = make_embedding_ablation_strategy_configs(config)

    assert len(configs) == 12
    assert "full_dataset" not in configs
    assert {
        strategy_config["strategy"] for strategy_config in configs.values()
    } == {"rmt_contraction", "raw_feature_clustering"}
    for strategy_config in configs.values():
        assert strategy_config["partition_selection_method"] == "auto"
        assert strategy_config["n_partitions"] == 2
        assert strategy_config["cluster_algorithms"] == ["kmeans"]
        assert strategy_config["router"] == "spectral" if (
            strategy_config["ensemble_method"] == "routed_weighted"
        ) else "router" not in strategy_config


def test_controlled_grid_accepts_shared_gmm_clusterer() -> None:
    config = EmbeddingSpaceAblationConfig(
        models=("lightgbm",),
        budget_ratios=(0.1,),
        ensemble_methods=("voting",),
        controlled_cluster_algorithms=("gmm",),
        show_progress=False,
    )

    configs = make_embedding_ablation_strategy_configs(config)

    assert len(configs) == 2
    assert {
        tuple(strategy_config["cluster_algorithms"])
        for strategy_config in configs.values()
    } == {("gmm",)}
    assert all(
        strategy_config["partition_selection_method"] == "auto"
        for strategy_config in configs.values()
    )


def test_paper_profile_preserves_auto_cluster_selection() -> None:
    config = EmbeddingSpaceAblationConfig(
        models=("lightgbm",),
        budget_ratios=(0.5,),
        ensemble_methods=("voting",),
        cluster_profile="paper",
        show_progress=False,
    )

    configs = make_embedding_ablation_strategy_configs(config)

    assert len(configs) == 2
    assert all(
        strategy_config["partition_selection_method"] == "auto"
        for strategy_config in configs.values()
    )
    assert all(
        "hdbscan" in strategy_config["cluster_algorithms"]
        for strategy_config in configs.values()
    )


def test_raw_feature_sampler_receives_same_budget_and_progress_contract() -> None:
    ensemble = SamplingEnsemble(
        problem="regression",
        partitioner_config={
            "strategy": "raw_feature_clustering",
            "n_partitions": 2,
            "budget_ratio": 0.5,
        },
        model_factory=lambda: object(),
        show_progress=False,
    )

    kwargs = ensemble._build_partitioner_kwargs(
        "raw_feature_clustering",
        random_state=42,
    )

    assert kwargs["sampling_budget_ratio"] == 0.5
    assert kwargs["show_progress"] is False


def test_report_builder_creates_exact_matched_budget_pair(tmp_path) -> None:
    def record(strategy: str, rmse: float) -> dict:
        return {
            "dataset": "tiny",
            "strategy_params": {
                "strategy": strategy,
                "model": "lightgbm",
                "split_label": "split_1",
                "budget_ratio": 0.5,
                "ensemble_method": "voting",
                "cluster_algorithms": ["kmeans", "gmm"],
            },
            "model_metrics": {"rmse": rmse},
            "timings_sec": {"fit": 2.0, "inference": 0.2},
            "extra": {
                "seed": 42,
                "runtime_diagnostics": {"partitioning": {"total": 1.0}},
                "partition_size_contract": {"selected_rows": 100},
                "sampler_diagnostics": {
                    "selected_n_partitions": 2,
                    "selected_cluster_algorithm": "kmeans",
                    "cluster_algorithms": ["kmeans", "gmm"],
                    "cluster_selection_metric": "balanced_silhouette",
                    "cluster_ensemble_method": "best_score",
                },
                "cache_usage": {
                    "structure": "miss",
                    "base_partitions": "miss",
                    "trained_models": "miss",
                },
            },
        }

    tables = EmbeddingAblationReportBuilder().build(
        [
            record("rmt_contraction", 9.0),
            record("raw_feature_clustering", 10.0),
        ],
        tmp_path,
    )

    pair = tables["pairs"].iloc[0]
    assert bool(pair["pair_complete"])
    assert bool(pair["row_budget_match"])
    assert pair["selected_rows_raw_minus_rmt"] == 0
    assert pair["rmse_raw_minus_rmt"] == pytest.approx(1.0)
    assert pair["rmt_improvement_vs_raw_pct"] == pytest.approx(10.0)
    assert pair["winner"] == "rmt_embedding"
    assert set(tables["runs"]["configured_cluster_algorithms"]) == {
        "kmeans|gmm"
    }
    assert set(tables["runs"]["evaluated_cluster_algorithms"]) == {
        "kmeans|gmm"
    }
    summary = tables["summary"].iloc[0]
    assert summary["rmt_wins"] == 1
    assert summary["raw_wins"] == 0
    dataset_summary = tables["dataset_summary"].iloc[0]
    assert dataset_summary["dataset"] == "tiny"
    assert dataset_summary["matched_budgets"] == 1
    assert (tmp_path / "embedding_ablation_by_dataset.csv").exists()
    assert (tmp_path / "embedding_ablation_report.md").exists()


def test_report_marks_row_budget_mismatch_incomplete(tmp_path) -> None:
    def record(strategy: str, rmse: float, selected_rows: int) -> dict:
        return {
            "dataset": "tiny",
            "strategy_params": {
                "strategy": strategy,
                "model": "lightgbm",
                "split_label": "split_1",
                "budget_ratio": 0.5,
                "ensemble_method": "voting",
            },
            "model_metrics": {"rmse": rmse},
            "extra": {
                "seed": 42,
                "partition_size_contract": {"selected_rows": selected_rows},
            },
        }

    pair = EmbeddingAblationReportBuilder().build(
        [
            record("rmt_contraction", 9.0, 100),
            record("raw_feature_clustering", 10.0, 101),
        ],
        tmp_path,
    )["pairs"].iloc[0]

    assert not bool(pair["row_budget_match"])
    assert not bool(pair["pair_complete"])
    assert pair["winner"] == "incomplete"


def test_final_integrity_check_rejects_row_budget_mismatch(tmp_path) -> None:
    def record(strategy: str, selected_rows: int) -> dict:
        return {
            "dataset": "tiny",
            "strategy_params": {
                "strategy": strategy,
                "model": "lightgbm",
                "split_label": "split_1",
                "budget_ratio": 0.5,
                "ensemble_method": "voting",
            },
            "model_metrics": {"rmse": 1.0},
            "extra": {
                "seed": 42,
                "partition_size_contract": {"selected_rows": selected_rows},
            },
        }

    tables = EmbeddingAblationReportBuilder().build(
        [
            record("rmt_contraction", 100),
            record("raw_feature_clustering", 99),
        ],
        tmp_path,
    )

    with pytest.raises(RuntimeError, match="complete_pairs=0"):
        EmbeddingSpaceAblationMediumOrchestrator._validate_completed_comparison(
            tables
        )


def test_protocol_records_expected_cache_reuse() -> None:
    config = EmbeddingSpaceAblationConfig(
        models=("lightgbm",),
        show_progress=False,
    )
    protocol = EmbeddingSpaceAblationMediumOrchestrator(
        config
    )._comparison_protocol()

    assert protocol["leaf_runs_per_dataset_model"] == 12
    assert protocol["expensive_structure_fits_per_dataset_split"] == 2
    assert protocol["unique_trained_chunk_sets_per_dataset_model"] == 6
    json.dumps(protocol)


def test_explicit_output_directory_must_be_empty(tmp_path: Path) -> None:
    output_dir = tmp_path / "existing_run"
    output_dir.mkdir()
    (output_dir / "marker.txt").write_text("existing", encoding="utf-8")
    config = EmbeddingSpaceAblationConfig(
        regression_tasks=("synthetic",),
        models=("lightgbm",),
        budget_ratios=(0.5,),
        output_dir=output_dir,
        synthetic_smoke=True,
        show_progress=False,
    )

    orchestrator = EmbeddingSpaceAblationMediumOrchestrator(config)
    with pytest.raises(FileExistsError, match="--resume-from"):
        orchestrator._create_logger()
