from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


JsonReady = Callable[[Any], Any]
SnapshotHook = Callable[[Sequence[Mapping[str, Any]]], Any]
MetadataBuilder = Callable[[Sequence[Mapping[str, Any]], str], Mapping[str, Any]]
LifecycleHook = Callable[[Sequence[Mapping[str, Any]], str], Any]


def default_json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): default_json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [default_json_ready(item) for item in value]
    return value


def load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


@dataclass
class IncrementalExperimentSaver:
    records_path: Path
    metadata_path: Path
    snapshot_hooks: Sequence[SnapshotHook] = ()
    lifecycle_hooks: Sequence[LifecycleHook] = ()
    metadata_builder: MetadataBuilder | None = None
    json_ready: JsonReady = default_json_ready
    rebuild_every: int = 1
    error_log_path: Path | None = None
    fsync_records: bool = True
    records: list[Mapping[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.rebuild_every < 1:
            raise ValueError("rebuild_every must be positive")
        self.records_path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        if self.error_log_path is None:
            self.error_log_path = self.records_path.with_name("incremental_saver_errors.jsonl")

    def start(self) -> None:
        self.write_metadata(status="running")
        self._run_lifecycle_hooks(status="running")

    def add_lifecycle_hook(
        self,
        hook: LifecycleHook,
    ) -> "IncrementalExperimentSaver":
        self.lifecycle_hooks = (*self.lifecycle_hooks, hook)
        return self

    def restore_records(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        rewrite_file: bool = False,
    ) -> None:
        self.records = [dict(record) for record in records]
        if rewrite_file:
            self._atomic_write_jsonl(self.records_path, self.records)

    def record(self, record: Mapping[str, Any]) -> None:
        normalized = dict(record)
        self._append_record_jsonl(normalized)
        self.records.append(normalized)
        if len(self.records) % self.rebuild_every == 0:
            self.persist_snapshot(status="running")

    def persist_snapshot(
        self,
        records: Sequence[Mapping[str, Any]] | None = None,
        status: str = "running",
    ) -> None:
        if records is not None:
            self.records = [dict(record) for record in records]
        for hook in self.snapshot_hooks:
            self._run_snapshot_hook(hook)
        self.write_metadata(status=status)
        self._run_lifecycle_hooks(status=status)

    def finalize(self, records: Sequence[Mapping[str, Any]] | None = None) -> None:
        if records is not None:
            self.records = [dict(record) for record in records]
        self.write_metadata(status="completed")
        self._run_lifecycle_hooks(status="completed")

    def mark_failed(self, error: BaseException) -> None:
        self._append_error("run_failed", error)
        self.write_metadata(status="failed")
        self._run_lifecycle_hooks(status="failed")

    def write_metadata(self, status: str) -> None:
        if self.metadata_builder is None:
            payload: Mapping[str, Any] = {"status": status, "records": len(self.records)}
        else:
            payload = self.metadata_builder(self.records, status)
        payload = {
            **dict(payload),
            "status": status,
            "records": len(self.records),
            "updated_utc": datetime.utcnow().isoformat(timespec="seconds"),
        }
        self._atomic_write_json(self.metadata_path, payload)

    def _append_record_jsonl(self, record: Mapping[str, Any]) -> None:
        line = json.dumps(self.json_ready(dict(record)), ensure_ascii=False)
        with self.records_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            if self.fsync_records:
                os.fsync(handle.fileno())

    def _run_snapshot_hook(self, hook: SnapshotHook) -> None:
        try:
            hook(tuple(self.records))
        except Exception as ex:  # pragma: no cover - defensive logging path
            self._append_error(f"snapshot_hook:{getattr(hook, '__name__', hook.__class__.__name__)}", ex)

    def _run_lifecycle_hooks(self, status: str) -> None:
        for hook in self.lifecycle_hooks:
            try:
                hook(tuple(self.records), status)
            except Exception as ex:  # pragma: no cover - defensive logging path
                hook_name = getattr(hook, "__name__", hook.__class__.__name__)
                self._append_error(f"lifecycle_hook:{hook_name}", ex)

    def _append_error(self, event: str, error: BaseException) -> None:
        if self.error_log_path is None:
            return
        payload = {
            "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds"),
            "event": event,
            "error_type": error.__class__.__name__,
            "error": str(error),
            "records": len(self.records),
        }
        with self.error_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(self.json_ready(payload), ensure_ascii=False) + "\n")

    def _atomic_write_json(self, path: Path, payload: Mapping[str, Any]) -> None:
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(
            json.dumps(self.json_ready(dict(payload)), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(tmp_path, path)

    def _atomic_write_jsonl(
        self,
        path: Path,
        records: Sequence[Mapping[str, Any]],
    ) -> None:
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(
                    json.dumps(
                        self.json_ready(dict(record)),
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            handle.flush()
            if self.fsync_records:
                os.fsync(handle.fileno())
        os.replace(tmp_path, path)
