from __future__ import annotations

import json

from examples.benchmark.benchmark_incremental import IncrementalExperimentSaver, load_jsonl_records


def test_incremental_saver_appends_records_and_refreshes_metadata(tmp_path) -> None:
    snapshot_calls: list[int] = []

    def _snapshot(records):
        snapshot_calls.append(len(records))
        (tmp_path / "snapshot.txt").write_text(str(len(records)), encoding="utf-8")

    def _metadata(records, status):
        return {"run_id": "demo", "status_seen": status, "records_seen": len(records)}

    saver = IncrementalExperimentSaver(
        records_path=tmp_path / "metrics" / "runs.jsonl",
        metadata_path=tmp_path / "run_meta.json",
        snapshot_hooks=(_snapshot,),
        metadata_builder=_metadata,
    )

    saver.start()
    saver.record({"dataset": "a", "metric": 1.0})
    saver.record({"dataset": "b", "metric": 2.0})
    saver.finalize()

    assert load_jsonl_records(tmp_path / "metrics" / "runs.jsonl") == [
        {"dataset": "a", "metric": 1.0},
        {"dataset": "b", "metric": 2.0},
    ]
    assert snapshot_calls == [1, 2]
    assert (tmp_path / "snapshot.txt").read_text(encoding="utf-8") == "2"

    meta = json.loads((tmp_path / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["status"] == "completed"
    assert meta["records"] == 2
    assert meta["status_seen"] == "completed"
    assert meta["records_seen"] == 2
    assert "updated_utc" in meta


def test_incremental_saver_keeps_jsonl_when_snapshot_hook_fails(tmp_path) -> None:
    def _broken_snapshot(records):
        raise RuntimeError("snapshot failed")

    saver = IncrementalExperimentSaver(
        records_path=tmp_path / "runs.jsonl",
        metadata_path=tmp_path / "run_meta.json",
        snapshot_hooks=(_broken_snapshot,),
    )

    saver.start()
    saver.record({"dataset": "a"})

    assert load_jsonl_records(tmp_path / "runs.jsonl") == [{"dataset": "a"}]
    error_lines = (tmp_path / "incremental_saver_errors.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(error_lines) == 1
    assert json.loads(error_lines[0])["error"] == "snapshot failed"


def test_incremental_saver_runs_lifecycle_hooks_after_metadata_write(
    tmp_path,
) -> None:
    calls: list[tuple[str, int, str]] = []

    def _lifecycle(records, status):
        metadata = json.loads(
            (tmp_path / "run_meta.json").read_text(encoding="utf-8")
        )
        calls.append((status, len(records), metadata["status"]))

    saver = IncrementalExperimentSaver(
        records_path=tmp_path / "metrics" / "runs.jsonl",
        metadata_path=tmp_path / "run_meta.json",
        lifecycle_hooks=(_lifecycle,),
    )

    saver.start()
    saver.record({"dataset": "a"})
    saver.finalize()

    assert calls == [
        ("running", 0, "running"),
        ("running", 1, "running"),
        ("completed", 1, "completed"),
    ]


def test_incremental_saver_logs_lifecycle_hook_failure(tmp_path) -> None:
    def _broken_lifecycle(records, status):
        raise RuntimeError(f"manifest failed at {status}")

    saver = IncrementalExperimentSaver(
        records_path=tmp_path / "metrics" / "runs.jsonl",
        metadata_path=tmp_path / "run_meta.json",
        lifecycle_hooks=(_broken_lifecycle,),
    )

    saver.start()

    error_lines = (
        tmp_path / "metrics" / "incremental_saver_errors.jsonl"
    ).read_text(encoding="utf-8").splitlines()
    assert len(error_lines) == 1
    error = json.loads(error_lines[0])
    assert error["event"] == "lifecycle_hook:_broken_lifecycle"
    assert error["error"] == "manifest failed at running"


def test_incremental_saver_restores_and_atomically_rewrites_records(
    tmp_path,
) -> None:
    records_path = tmp_path / "metrics" / "runs.jsonl"
    saver = IncrementalExperimentSaver(
        records_path=records_path,
        metadata_path=tmp_path / "run_meta.json",
    )
    saver.record({"leaf": "completed"})
    saver.record({"leaf": "failed"})

    saver.restore_records(
        [{"leaf": "completed"}],
        rewrite_file=True,
    )

    assert saver.records == [{"leaf": "completed"}]
    assert load_jsonl_records(records_path) == [{"leaf": "completed"}]
    assert not records_path.with_suffix(".jsonl.tmp").exists()
