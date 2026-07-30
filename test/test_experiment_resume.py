from __future__ import annotations

import json
from pathlib import Path

import pytest

from examples.benchmark.benchmark_incremental import load_jsonl_records
from examples.benchmark.benchmark_logging import BenchmarkLogger
import examples.benchmark.rmt_regression_medium_datasets as rmt_runner
from sampling_zoo.core.experiment.artifact_manifest import (
    AcceleratorIdentity,
    ExperimentConfigRef,
    RunIdentity,
)
from sampling_zoo.core.experiment.artifact_runtime import (
    canonical_payload_sha256,
    materialize_experiment_artifact_manifest,
)
from sampling_zoo.core.experiment.errors import ResumeCompatibilityError
from sampling_zoo.core.experiment.resume import (
    ResumePolicy,
    build_resume_plan,
    leaf_run_key_from_components,
    leaf_run_key_from_record,
    legacy_resume_config,
    scientific_experiment_config,
)
from sampling_zoo.core.experiment.resume_runtime import load_resume_session


def _record(
    *,
    dataset: str = "diamonds",
    split: str = "split_1",
    model: str = "ridge",
    strategy: str = "random__voting__budget_10",
    budget_ratio: float = 0.1,
    seed: int = 42,
    failed: bool = False,
) -> dict:
    strategy_config = {
        "strategy": "random",
        "budget_ratio": budget_ratio,
        "random_state": seed,
    }
    leaf_run = leaf_run_key_from_components(
        dataset=dataset,
        split=split,
        model=model,
        strategy=strategy,
        strategy_config=strategy_config,
        seed=seed,
    )
    return {
        "dataset": dataset,
        "strategy": f"{strategy}__{model}__{split}",
        "strategy_params": {
            **strategy_config,
            "model": model,
            "split_label": split,
        },
        "model_metrics": {} if failed else {"rmse": 1.0},
        "extra": {
            "strategy": strategy,
            "model": model,
            "split_label": split,
            "leaf_run_key": leaf_run.key,
            "leaf_run": leaf_run.to_dict(),
            **({"error": "forced failure"} if failed else {}),
        },
    }


def _identity(run_id: str, config_hash: str) -> RunIdentity:
    return RunIdentity(
        project="sampling-zoo",
        run_id=run_id,
        created_at_utc="2026-07-30T10:00:00+00:00",
        git_commit="a" * 40,
        git_dirty=False,
        python_version="3.11.9",
        platform="test-platform",
        packages=(),
        accelerator=AcceleratorIdentity(
            torch_version=None,
            cuda_runtime=None,
            cuda_available=False,
            device_count=0,
            device_name=None,
        ),
        config=ExperimentConfigRef(sha256=config_hash, seed=42),
    )


def _materialize_run(
    tmp_path: Path,
    *,
    records: list[dict],
    status: str,
    config_hash: str,
) -> Path:
    run_dir = tmp_path / "run_resume"
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True)
    (run_dir / "run_meta.json").write_text(
        json.dumps({"status": status}),
        encoding="utf-8",
    )
    raw_path = metrics_dir / "rmt_regression_runs.jsonl"
    raw_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    materialize_experiment_artifact_manifest(
        run_dir=run_dir,
        run_identity=_identity(run_dir.name, config_hash),
        status=status,
        records=records,
    )
    return run_dir


def test_leaf_run_key_is_deterministic_and_covers_every_axis() -> None:
    base = {
        "dataset": "diamonds",
        "split": "split_1",
        "model": "ridge",
        "strategy": "rmt__routed__budget_10",
        "strategy_config": {
            "budget_ratio": 0.1,
            "router": "spectral",
            "view_strategy": "gaussian",
        },
        "seed": 42,
    }
    first = leaf_run_key_from_components(**base)
    second = leaf_run_key_from_components(**base)

    assert first == second
    assert first.key == second.key
    variants = [
        {**base, "dataset": "elevators"},
        {**base, "split": "split_2"},
        {**base, "model": "lightgbm"},
        {**base, "strategy": "random"},
        {
            **base,
            "strategy_config": {
                **base["strategy_config"],
                "budget_ratio": 0.3,
            },
        },
        {
            **base,
            "strategy_config": {
                **base["strategy_config"],
                "router": "constrained_gating",
            },
        },
        {
            **base,
            "strategy_config": {
                **base["strategy_config"],
                "view_strategy": "subsample",
            },
        },
        {**base, "seed": 43},
    ]
    assert len({first.key, *(leaf_run_key_from_components(**item).key for item in variants)}) == 9


def test_leaf_run_key_supports_records_without_embedded_identity() -> None:
    record = _record()
    expected = leaf_run_key_from_record(record, default_seed=42)
    record["extra"].pop("leaf_run")
    record["extra"].pop("leaf_run_key")

    assert leaf_run_key_from_record(record, default_seed=42) == expected


def test_resume_plan_retries_failed_and_completed_record_wins() -> None:
    failed = _record(failed=True)
    completed = _record(failed=False)
    records = [failed, completed, failed]

    plan = build_resume_plan(
        records,
        policy=ResumePolicy.RETRY_FAILED,
        default_seed=42,
    )

    key = leaf_run_key_from_record(completed, default_seed=42).key
    assert plan.completed_keys == frozenset({key})
    assert plan.failed_keys == frozenset()
    assert plan.skipped_keys == frozenset({key})
    assert len(plan.retained_records) == 1
    assert plan.duplicate_records_dropped == 2
    rebuilt = build_resume_plan(
        plan.retained_records,
        policy=ResumePolicy.RETRY_FAILED,
        default_seed=42,
    )
    assert rebuilt.retained_records == plan.retained_records
    assert rebuilt.skipped_keys == plan.skipped_keys


def test_resume_plan_marks_isolated_failure_for_retry() -> None:
    record = _record(failed=True)

    plan = build_resume_plan(
        [record],
        policy=ResumePolicy.RETRY_FAILED,
        default_seed=42,
    )

    key = leaf_run_key_from_record(record, default_seed=42).key
    assert plan.failed_keys == frozenset({key})
    assert plan.retry_keys == frozenset({key})
    assert plan.skipped_keys == frozenset()
    assert plan.retained_records == ()


def test_resume_plan_skip_existing_retains_failed_record() -> None:
    record = _record(failed=True)

    plan = build_resume_plan(
        [record],
        policy=ResumePolicy.SKIP_EXISTING,
        default_seed=42,
    )

    key = leaf_run_key_from_record(record, default_seed=42).key
    assert plan.skipped_keys == frozenset({key})
    assert plan.retry_keys == frozenset()
    assert plan.retained_records == (record,)


def test_resume_runtime_rejects_incompatible_config(tmp_path) -> None:
    persisted_hash = canonical_payload_sha256({"seed": 42})
    run_dir = _materialize_run(
        tmp_path,
        records=[_record()],
        status="completed",
        config_hash=persisted_hash,
    )

    with pytest.raises(ResumeCompatibilityError) as exc_info:
        load_resume_session(
            run_dir,
            expected_config_hashes=(
                canonical_payload_sha256({"seed": 43}),
            ),
            policy="retry_failed",
            default_seed=42,
        )

    assert exc_info.value.code == "resume_config_mismatch"


def test_resume_runtime_accepts_legacy_pre_resume_config_hash(
    tmp_path,
) -> None:
    current_config = {
        "seed": 42,
        "show_progress": False,
        "resume_from": "run_resume",
        "resume_policy": "retry_failed",
    }
    legacy_hash = canonical_payload_sha256(
        legacy_resume_config(current_config)
    )
    run_dir = _materialize_run(
        tmp_path,
        records=[_record()],
        status="completed",
        config_hash=legacy_hash,
    )

    session = load_resume_session(
        run_dir,
        expected_config_hashes=(
            canonical_payload_sha256(
                scientific_experiment_config(current_config)
            ),
            legacy_hash,
        ),
        policy="retry_failed",
        default_seed=42,
    )

    assert session.accepted_config_hash == legacy_hash


def test_resume_runtime_recovers_trailing_partial_line_for_failed_run(
    tmp_path,
) -> None:
    config_hash = canonical_payload_sha256({"seed": 42})
    run_dir = _materialize_run(
        tmp_path,
        records=[_record()],
        status="failed",
        config_hash=config_hash,
    )
    raw_path = run_dir / "metrics" / "rmt_regression_runs.jsonl"
    with raw_path.open("a", encoding="utf-8") as handle:
        handle.write('{"dataset": "partial"')

    session = load_resume_session(
        run_dir,
        expected_config_hashes=(config_hash,),
        policy="retry_failed",
        default_seed=42,
    )

    assert session.trailing_partial_line_ignored is True
    assert session.raw_hash_matches_manifest is False
    assert len(session.plan.retained_records) == 1


def test_completed_run_rejects_raw_hash_mismatch(tmp_path) -> None:
    config_hash = canonical_payload_sha256({"seed": 42})
    run_dir = _materialize_run(
        tmp_path,
        records=[_record()],
        status="completed",
        config_hash=config_hash,
    )
    with (
        run_dir / "metrics" / "rmt_regression_runs.jsonl"
    ).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_record(split="split_2")) + "\n")

    with pytest.raises(ResumeCompatibilityError) as exc_info:
        load_resume_session(
            run_dir,
            expected_config_hashes=(config_hash,),
            policy="retry_failed",
            default_seed=42,
        )

    assert exc_info.value.code == "resume_raw_records_hash_mismatch"


def test_interrupted_synthetic_run_resumes_without_duplicate_leaf_runs(
    tmp_path,
) -> None:
    run_id = "run_resume_smoke"

    class FixedRunOrchestrator(
        rmt_runner.RMTRegressionExperimentOrchestrator
    ):
        def _create_logger(self):
            if self.config.resume_from is not None:
                return super()._create_logger()
            return BenchmarkLogger(
                run_id=run_id,
                artifacts_root=tmp_path,
            )

    class InterruptingOrchestrator(FixedRunOrchestrator):
        def _create_incremental_recorder(self, logger):
            record = super()._create_incremental_recorder(logger)
            calls = 0

            def _record_then_interrupt(payload):
                nonlocal calls
                record(payload)
                calls += 1
                if calls == 2:
                    raise RuntimeError("forced interruption")

            return _record_then_interrupt

    base_config = dict(
        regression_suite=None,
        regression_tasks=None,
        strategies=("random",),
        models=("ridge",),
        ensemble_methods=("voting",),
        budget_ratios=(0.1,),
        view_strategies=("gaussian",),
        router_modes=("spectral",),
        n_partitions=2,
        max_train_rows=None,
        seed=42,
        show_progress=False,
        synthetic_smoke=True,
    )
    with pytest.raises(RuntimeError, match="forced interruption"):
        InterruptingOrchestrator(
            rmt_runner.RMTRegressionExperimentConfig(**base_config)
        ).run()

    run_dir = tmp_path / run_id
    raw_path = run_dir / "metrics" / "rmt_regression_runs.jsonl"
    interrupted_records = load_jsonl_records(raw_path)
    assert len(interrupted_records) == 2
    assert json.loads(
        (run_dir / "artifact_manifest.json").read_text(encoding="utf-8")
    )["status"] == "failed"

    result = FixedRunOrchestrator(
        rmt_runner.RMTRegressionExperimentConfig(
            **base_config,
            resume_from=run_dir,
        )
    ).run()

    final_records = load_jsonl_records(raw_path)
    final_keys = [
        leaf_run_key_from_record(record, default_seed=42).key
        for record in final_records
    ]
    assert result == run_dir
    assert len(final_records) == 4
    assert len(final_keys) == len(set(final_keys))
    assert {
        leaf_run_key_from_record(record, default_seed=42).key
        for record in interrupted_records
    }.issubset(final_keys)
    manifest = json.loads(
        (run_dir / "artifact_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "completed"
    assert manifest["record_count"] == 4
