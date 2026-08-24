"""Run a small real-data smoke test for the gated RMT research program."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Optional, Sequence

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = Path(__file__).resolve().parent
for module_path in (ROOT_DIR, BENCHMARK_DIR):
    if str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))

from rmt_bulk_spike_topology_real_experiment import (  # noqa: E402
    run_rmt_bulk_spike_topology_real_experiment,
)
from rmt_classification_geometry_guard_analysis import (  # noqa: E402
    build_classification_geometry_gate,
)
from rmt_cross_fitted_geometry_selector_experiment import (  # noqa: E402
    run_rmt_classification_guarded_geometry_selector_experiment,
)
from rmt_experiment_utils import json_ready  # noqa: E402


SMOKE_CLASSIFICATION_TASKS: tuple[str, ...] = ("adult", "car")
SMOKE_TOPOLOGY_REGRESSION_TASKS: tuple[str, ...] = ("diamonds",)
SMOKE_TOPOLOGY_CLASSIFICATION_TASKS: tuple[str, ...] = ("bank-marketing",)
SMOKE_EXPECTED_GEOMETRY_RECORDS = 8
SMOKE_EXPECTED_SELECTOR_RECORDS = 2
SMOKE_EXPECTED_TOPOLOGY_RECORDS = 8


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_output_root() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return ROOT_DIR / "examples" / "benchmark" / "results" / f"run_rmt_research_smoke_{stamp}"


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(
        json.dumps(json_ready(value), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _load_existing_runs(output_dir: Path) -> Optional[pd.DataFrame]:
    replay_path = output_dir / "routing_geometry_replay.csv"
    if not replay_path.exists():
        return None
    return pd.read_csv(replay_path)


def _topology_operational_gate(
    runs: pd.DataFrame,
    scientific_gate: dict[str, Any],
) -> dict[str, Any]:
    scientific_checks = scientific_gate.get("checks", {})
    checks = {
        "expected_record_count": len(runs) == SMOKE_EXPECTED_TOPOLOGY_RECORDS,
        "all_records_completed": bool(
            len(runs) == SMOKE_EXPECTED_TOPOLOGY_RECORDS
            and runs["status"].eq("completed").all()
        ),
        "full_dataset_references_completed": bool(
            scientific_checks.get("full_dataset_references_completed", False)
        ),
        "specialized_topologies_preserve_exact_budget": bool(
            scientific_checks.get(
                "specialized_topologies_preserve_exact_budget",
                False,
            )
        ),
        "regression_tail_contract": bool(
            scientific_checks.get("regression_tail_noninferiority", False)
        ),
        "classification_probability_and_balance_contract": bool(
            scientific_checks.get(
                "classification_probability_and_balance_noninferiority",
                False,
            )
        ),
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "scientific_gate_status": scientific_gate.get("status"),
        "scientific_gate_is_informative": False,
        "reason": (
            "A one-seed smoke validates execution and invariants; specialization "
            "effectiveness is evaluated by the multi-seed pilot."
        ),
    }


def run_rmt_bulk_spike_research_smoke(
    *,
    output_root: Optional[str | Path] = None,
    models: Sequence[str] = ("lightgbm",),
    max_train_rows: Optional[int] = 10_000,
    show_progress: bool = True,
    resume: bool = True,
) -> dict[str, Any]:
    """Validate classification safety and bulk/spike budget invariants."""

    root = _default_output_root() if output_root is None else Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    metadata_path = root / "smoke_meta.json"
    previous_attempt = None
    if resume and metadata_path.exists():
        previous_attempt = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata: dict[str, Any] = {
        "status": "running",
        "started_at": _utc_now(),
        "completed_at": None,
        "output_root": str(root.resolve()),
        "stages": {},
        "previous_attempt": previous_attempt,
    }
    _write_json_atomic(metadata_path, metadata)

    try:
        classification_dir = root / "01_classification_geometry_guard"
        classification_runs = (
            _load_existing_runs(classification_dir) if resume else None
        )
        if classification_runs is None:
            classification_runs = run_rmt_classification_guarded_geometry_selector_experiment(
                classification_tasks=SMOKE_CLASSIFICATION_TASKS,
                models=models,
                budget_ratios=(0.10,),
                seeds=(42,),
                max_train_rows=max_train_rows,
                output_dir=classification_dir,
                show_progress=show_progress,
            )
        classification_gate = build_classification_geometry_gate(
            classification_dir,
            expected_records=SMOKE_EXPECTED_GEOMETRY_RECORDS,
            expected_selector_records=SMOKE_EXPECTED_SELECTOR_RECORDS,
        )
        metadata["stages"]["classification_geometry_guard"] = {
            "status": classification_gate["status"],
            "record_count": len(classification_runs),
            "output_dir": str(classification_dir.resolve()),
        }
        _write_json_atomic(metadata_path, metadata)
        if classification_gate["status"] != "passed":
            raise RuntimeError("classification geometry smoke gate failed")

        topology_dir = root / "02_bulk_spike_topology"
        topology_runs = _load_existing_runs(topology_dir) if resume else None
        if topology_runs is None:
            topology_runs = run_rmt_bulk_spike_topology_real_experiment(
                regression_tasks=SMOKE_TOPOLOGY_REGRESSION_TASKS,
                classification_tasks=SMOKE_TOPOLOGY_CLASSIFICATION_TASKS,
                models=models,
                budget_ratios=(0.10,),
                seeds=(42,),
                max_train_rows=max_train_rows,
                output_dir=topology_dir,
                show_progress=show_progress,
            )
        topology_gate = json.loads(
            (topology_dir / "bulk_spike_gate.json").read_text(encoding="utf-8")
        )
        operational_gate = _topology_operational_gate(
            topology_runs,
            topology_gate,
        )
        _write_json_atomic(
            topology_dir / "bulk_spike_smoke_gate.json",
            operational_gate,
        )
        metadata["stages"]["bulk_spike_topology"] = {
            "status": operational_gate["status"],
            "record_count": len(topology_runs),
            "output_dir": str(topology_dir.resolve()),
            "scientific_gate_status": topology_gate["status"],
        }
        if operational_gate["status"] != "passed":
            raise RuntimeError("bulk/spike topology operational smoke gate failed")

        metadata["status"] = "completed"
    except Exception as error:
        metadata["status"] = "failed"
        metadata["error_type"] = type(error).__name__
        metadata["error"] = str(error)
        raise
    finally:
        metadata["completed_at"] = _utc_now()
        _write_json_atomic(metadata_path, metadata)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--max-train-rows", type=int, default=10_000)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    result = run_rmt_bulk_spike_research_smoke(
        output_root=args.output_root,
        max_train_rows=args.max_train_rows,
        show_progress=not args.no_progress,
        resume=not args.no_resume,
    )
    print(json.dumps(json_ready(result), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
