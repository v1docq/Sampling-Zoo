from __future__ import annotations

import json
import re
import sys

from examples.benchmark.benchmark_logging import BenchmarkLogger


def test_console_capture_and_structured_stage_log(tmp_path) -> None:
    logger = BenchmarkLogger(run_id="logging_test", artifacts_root=tmp_path)
    capture = logger.start_console_capture()
    try:
        print("stdout marker")
        print("stderr marker", file=sys.stderr)
        with logger.stage("test.stage", dataset="tiny"):
            print("inside stage")
    finally:
        capture.close()

    console_lines = logger.console_log_path.read_text(encoding="utf-8").splitlines()
    assert any("[stdout] stdout marker" in line for line in console_lines)
    assert any("[stderr] stderr marker" in line for line in console_lines)
    assert all(
        re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}[+-]\d{2}:\d{2}", line)
        for line in console_lines
    )

    events = [
        json.loads(line)
        for line in logger.events_path.read_text(encoding="utf-8").splitlines()
    ]
    stage_events = [event for event in events if event["event"] == "test.stage"]
    assert [event["status"] for event in stage_events] == ["started", "completed"]
    assert stage_events[-1]["duration_sec"] >= 0.0
    assert stage_events[-1]["details"]["dataset"] == "tiny"
    assert stage_events[-1]["resources"]["max_rss_bytes"] > 0

    environment = json.loads(logger.environment_path.read_text(encoding="utf-8"))
    assert environment["logical_cpu_count"]
    assert environment["total_memory_bytes"] > 0
    assert "lightgbm" in environment["packages"]
