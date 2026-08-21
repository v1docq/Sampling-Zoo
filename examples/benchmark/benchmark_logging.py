from __future__ import annotations

import json
import os
import platform
import resource
import sys
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import matplotlib

matplotlib.use("Agg")


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _compact_details(details: Mapping[str, Any]) -> str:
    return " ".join(
        f"{key}={value}"
        for key, value in details.items()
        if value is not None
    )


def _process_resource_snapshot() -> Dict[str, Any]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    max_rss = float(usage.ru_maxrss)
    # Linux reports KiB while macOS reports bytes.
    max_rss_bytes = int(max_rss if sys.platform == "darwin" else max_rss * 1024.0)
    return {
        "pid": os.getpid(),
        "max_rss_bytes": max_rss_bytes,
        "user_cpu_sec": float(usage.ru_utime),
        "system_cpu_sec": float(usage.ru_stime),
    }


def _total_memory_bytes() -> Optional[int]:
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        page_count = int(os.sysconf("SC_PHYS_PAGES"))
        return page_size * page_count
    except (AttributeError, OSError, TypeError, ValueError):
        return None


def _linux_cpu_model() -> Optional[str]:
    cpuinfo = Path("/proc/cpuinfo")
    if not cpuinfo.exists():
        return None
    try:
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name") and ":" in line:
                return line.split(":", 1)[1].strip()
    except OSError:
        return None
    return None


def _package_versions(names: Sequence[str]) -> Dict[str, Optional[str]]:
    versions: Dict[str, Optional[str]] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _accelerator_snapshot() -> Dict[str, Any]:
    try:
        import torch
    except Exception:
        return {
            "torch_version": None,
            "cuda_runtime": None,
            "cuda_available": False,
            "device_count": 0,
            "device_names": [],
        }
    try:
        cuda_available = bool(torch.cuda.is_available())
        device_count = int(torch.cuda.device_count()) if cuda_available else 0
        device_names = [
            str(torch.cuda.get_device_name(index))
            for index in range(device_count)
        ]
    except Exception:
        cuda_available = False
        device_count = 0
        device_names = []
    return {
        "torch_version": str(torch.__version__),
        "cuda_runtime": None if torch.version.cuda is None else str(torch.version.cuda),
        "cuda_available": cuda_available,
        "device_count": device_count,
        "device_names": device_names,
    }


@dataclass
class ArtifactPaths:
    root: Path
    logs: Path
    metrics: Path
    plots: Path
    samples: Path


class _TimestampedTee:
    """Mirror a text stream while writing a timestamped, line-oriented log."""

    def __init__(self, original: Any, log_handle: Any, stream_name: str, lock: threading.RLock) -> None:
        self._original = original
        self._log_handle = log_handle
        self._stream_name = stream_name
        self._lock = lock
        self._buffer = ""

    def write(self, value: Any) -> int:
        text = str(value)
        with self._lock:
            written = self._original.write(text)
            self._original.flush()
            # tqdm uses carriage returns. Treat them as line boundaries in the
            # persistent log so an interrupted run still shows its last state.
            normalized = text.replace("\r\n", "\n").replace("\r", "\n")
            self._buffer += normalized
            while "\n" in self._buffer:
                line, self._buffer = self._buffer.split("\n", 1)
                self._write_log_line(line)
            self._log_handle.flush()
        return len(text) if written is None else int(written)

    def flush(self) -> None:
        with self._lock:
            self._original.flush()
            self._log_handle.flush()

    def close_log_buffer(self) -> None:
        with self._lock:
            if self._buffer:
                self._write_log_line(self._buffer)
                self._buffer = ""
            self._log_handle.flush()

    def _write_log_line(self, line: str) -> None:
        timestamp = datetime.now().astimezone().isoformat(timespec="milliseconds")
        self._log_handle.write(f"{timestamp} [{self._stream_name}] {line}\n")

    def isatty(self) -> bool:
        return bool(getattr(self._original, "isatty", lambda: False)())

    def fileno(self) -> int:
        return self._original.fileno()

    @property
    def encoding(self) -> str:
        return getattr(self._original, "encoding", "utf-8")

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)


class ConsoleLogCapture:
    """Active stdout/stderr tee owned by a benchmark logger."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._handle: Any = None
        self._stdout: Any = None
        self._stderr: Any = None
        self._stdout_tee: Optional[_TimestampedTee] = None
        self._stderr_tee: Optional[_TimestampedTee] = None
        self._closed = False

    def start(self) -> "ConsoleLogCapture":
        if self._handle is not None:
            return self
        self._handle = self.path.open("a", encoding="utf-8", buffering=1)
        self._stdout, self._stderr = sys.stdout, sys.stderr
        lock = threading.RLock()
        self._stdout_tee = _TimestampedTee(self._stdout, self._handle, "stdout", lock)
        self._stderr_tee = _TimestampedTee(self._stderr, self._handle, "stderr", lock)
        sys.stdout, sys.stderr = self._stdout_tee, self._stderr_tee
        return self

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._stdout_tee is not None:
            self._stdout_tee.close_log_buffer()
        if self._stderr_tee is not None:
            self._stderr_tee.close_log_buffer()
        if self._stdout is not None and sys.stdout is self._stdout_tee:
            sys.stdout = self._stdout
        if self._stderr is not None and sys.stderr is self._stderr_tee:
            sys.stderr = self._stderr
        if self._handle is not None:
            self._handle.close()

    def __enter__(self) -> "ConsoleLogCapture":
        return self.start()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


class BenchmarkLogger:
    """Utility to persist benchmark artifacts and strategy run logs."""

    def __init__(self, run_id: Optional[str] = None, artifacts_root: str | Path = "artifacts/benchmarks") -> None:
        self.run_id = run_id or self._generate_run_id()
        self.paths = self._init_artifacts(artifacts_root=artifacts_root, run_id=self.run_id)
        self.jsonl_path = self.paths.logs / "strategy_runs.jsonl"
        self.events_path = self.paths.logs / "events.jsonl"
        self.console_log_path = self.paths.logs / "console.log"
        self.environment_path = self.paths.root / "environment.json"
        self._started_at = perf_counter()
        self._write_lock = threading.RLock()
        self.save_environment_snapshot()

    @staticmethod
    def _generate_run_id() -> str:
        return f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

    @staticmethod
    def _init_artifacts(artifacts_root: str | Path, run_id: str) -> ArtifactPaths:
        root = Path(artifacts_root) / run_id
        logs = root / "logs"
        metrics = root / "metrics"
        plots = root / "plots"
        samples = root / "samples"

        for path in (logs, metrics, plots, samples):
            path.mkdir(parents=True, exist_ok=True)

        return ArtifactPaths(root=root, logs=logs, metrics=metrics, plots=plots, samples=samples)

    def log_strategy_run(
        self,
        dataset_name: str,
        strategy_name: str,
        strategy_params: Mapping[str, Any],
        model_metrics: Mapping[str, Any],
        timings: Mapping[str, float],
        sample_stats: Mapping[str, Any],
        extra: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload = {
            "run_id": self.run_id,
            "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds"),
            "dataset": dataset_name,
            "strategy": strategy_name,
            "strategy_params": _json_ready(dict(strategy_params)),
            "model_metrics": _json_ready(dict(model_metrics)),
            "timings_sec": _json_ready(dict(timings)),
            "sample_stats": _json_ready(dict(sample_stats)),
        }
        if extra:
            payload["extra"] = _json_ready(dict(extra))

        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

        self._save_metrics_snapshot(dataset_name, strategy_name, payload)
        return payload

    def start_console_capture(self) -> ConsoleLogCapture:
        """Start mirroring stdout and stderr into ``logs/console.log``."""
        return ConsoleLogCapture(self.console_log_path).start()

    def log_event(
        self,
        event: str,
        *,
        status: Optional[str] = None,
        duration_sec: Optional[float] = None,
        details: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "run_id": self.run_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "timestamp_local": datetime.now().astimezone().isoformat(timespec="milliseconds"),
            "elapsed_run_sec": float(perf_counter() - self._started_at),
            "event": str(event),
            "resources": _process_resource_snapshot(),
        }
        if status is not None:
            payload["status"] = str(status)
        if duration_sec is not None:
            payload["duration_sec"] = float(duration_sec)
        if details:
            payload["details"] = _json_ready(dict(details))
        with self._write_lock:
            with self.events_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return payload

    @contextmanager
    def stage(self, stage_name: str, **details: Any) -> Iterable[None]:
        """Log a stage immediately, including duration and failure details."""
        started = perf_counter()
        print(f"[stage:start] {stage_name} {_compact_details(details)}".rstrip(), flush=True)
        self.log_event(stage_name, status="started", details=details)
        try:
            yield
        except BaseException as exc:
            duration = perf_counter() - started
            failure_details = {
                **details,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            self.log_event(
                stage_name,
                status="failed",
                duration_sec=duration,
                details=failure_details,
            )
            print(
                f"[stage:failed] {stage_name} duration={duration:.3f}s "
                f"error={type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            raise
        else:
            duration = perf_counter() - started
            self.log_event(
                stage_name,
                status="completed",
                duration_sec=duration,
                details=details,
            )
            print(f"[stage:end] {stage_name} duration={duration:.3f}s", flush=True)

    def save_environment_snapshot(self) -> Path:
        """Persist the hardware/software context needed to interpret timings."""
        payload = {
            "captured_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor() or _linux_cpu_model(),
            "logical_cpu_count": os.cpu_count(),
            "total_memory_bytes": _total_memory_bytes(),
            "accelerator": _accelerator_snapshot(),
            "packages": _package_versions(
                ("sampling-zoo", "numpy", "pandas", "scipy", "scikit-learn", "openml", "lightgbm", "tabpfn", "tabicl", "torch")
            ),
            "environment": {
                key: os.environ.get(key)
                for key in (
                    "CUDA_VISIBLE_DEVICES",
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "LOKY_MAX_CPU_COUNT",
                )
                if os.environ.get(key) is not None
            },
        }
        self.environment_path.write_text(
            json.dumps(_json_ready(payload), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return self.environment_path

    def _save_metrics_snapshot(self, dataset_name: str, strategy_name: str, payload: Mapping[str, Any]) -> None:
        file_name = f"{dataset_name}__{strategy_name}.json".replace("/", "_")
        with (self.paths.metrics / file_name).open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def save_sample_dump(self, dataset_name: str, strategy_name: str, sample_indices: Sequence[int]) -> Path:
        file_name = f"{dataset_name}__{strategy_name}_sample_indices.npy".replace("/", "_")
        target_path = self.paths.samples / file_name
        np.save(target_path, np.asarray(sample_indices, dtype=int))
        return target_path

    def save_probability_distribution_plot(
        self,
        values: Sequence[float],
        dataset_name: str,
        strategy_name: str,
        value_name: str = "probability / weight",
    ) -> List[Path]:
        plt = _safe_import_matplotlib()
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(values, bins=30, color="#4C72B0", alpha=0.85)
        ax.set_title(f"{strategy_name}: {value_name} distribution ({dataset_name})")
        ax.set_xlabel(value_name)
        ax.set_ylabel("count")
        return _save_figure(fig, self.paths.plots / f"{dataset_name}__{strategy_name}_probability_distribution")

    def save_class_coverage_plot(
        self,
        class_counts: Mapping[Any, int],
        dataset_name: str,
        strategy_name: str,
    ) -> List[Path]:
        plt = _safe_import_matplotlib()
        labels = list(class_counts.keys())
        values = [class_counts[label] for label in labels]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(range(len(labels)), values, color="#55A868")
        ax.set_title(f"{strategy_name}: class coverage ({dataset_name})")
        ax.set_xlabel("class")
        ax.set_ylabel("samples")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        return _save_figure(fig, self.paths.plots / f"{dataset_name}__{strategy_name}_class_coverage")

    def save_2d_projection_plot(
        self,
        x_sampled: np.ndarray,
        y_sampled: Sequence[Any],
        dataset_name: str,
        strategy_name: str,
        method: str = "pca",
    ) -> List[Path]:
        projection = build_2d_projection(x_sampled, method=method)
        plt = _safe_import_matplotlib()

        fig, ax = plt.subplots(figsize=(7, 6))
        scatter = ax.scatter(projection[:, 0], projection[:, 1], c=np.asarray(y_sampled), cmap="tab10", s=24, alpha=0.8)
        ax.set_title(f"{strategy_name}: 2D projection ({method.upper()}) - {dataset_name}")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        fig.colorbar(scatter, ax=ax, label="class")
        return _save_figure(fig, self.paths.plots / f"{dataset_name}__{strategy_name}_2d_projection_{method}")

    def create_markdown_report(self, run_records: Iterable[Mapping[str, Any]]) -> Path:
        records = list(run_records)
        report_path = self.paths.root / "report.md"

        def _problem_type(record: Mapping[str, Any]) -> str:
            extra = record.get("extra", {}) or {}
            pt = str(extra.get("problem_type", "")).lower()
            if pt in {"classification", "regression"}:
                return pt
            metrics = record.get("model_metrics", {}) or {}
            if any(key in metrics for key in ("rmse", "mse", "r2")):
                return "regression"
            return "classification"

        classification_records = [record for record in records if _problem_type(record) == "classification"]
        regression_records = [record for record in records if _problem_type(record) == "regression"]

        lines = [
            f"# Benchmark report ({self.run_id})",
            "",
            f"- Total records: {len(records)}",
            f"- Classification records: {len(classification_records)}",
            f"- Regression records: {len(regression_records)}",
        ]

        grouped: Dict[str, List[Mapping[str, Any]]] = {}
        if classification_records:
            lines.extend(
                [
                    "",
                    "## Classification",
                    "",
                    "| Dataset | Strategy | ROC-AUC | Log loss | Brier | ECE | F1 macro | F1 weighted | fit(s) | sample(s) | inference(s) |",
                    "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for record in classification_records:
                dataset_name = str(record.get("dataset", "unknown"))
                grouped.setdefault(dataset_name, []).append(record)
                metrics = record.get("model_metrics", {})
                timings = record.get("timings_sec", {})
                lines.append(
                    "| {dataset} | {strategy} | {roc_auc:.4f} | {log_loss:.4f} | {brier:.4f} | {ece:.4f} | {f1_macro:.4f} | {f1_weighted:.4f} | {fit:.4f} | {sample:.4f} | {inference:.4f} |".format(
                        dataset=dataset_name,
                        strategy=record.get("strategy", "-"),
                        roc_auc=float(metrics.get("roc_auc", float("nan"))),
                        log_loss=float(metrics.get("log_loss", float("nan"))),
                        brier=float(metrics.get("brier_score", float("nan"))),
                        ece=float(metrics.get("expected_calibration_error", float("nan"))),
                        f1_macro=float(metrics.get("f1_macro", float("nan"))),
                        f1_weighted=float(metrics.get("f1_weighted", float("nan"))),
                        fit=float(timings.get("fit", 0.0)),
                        sample=float(timings.get("sample", 0.0)),
                        inference=float(timings.get("inference", 0.0)),
                    )
                )

        if regression_records:
            lines.extend(
                [
                    "",
                    "## Regression",
                    "",
                    "| Dataset | Strategy | RMSE | MSE | R2 | fit(s) | sample(s) | inference(s) |",
                    "|---|---|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for record in regression_records:
                dataset_name = str(record.get("dataset", "unknown"))
                grouped.setdefault(dataset_name, []).append(record)
                metrics = record.get("model_metrics", {})
                timings = record.get("timings_sec", {})
                lines.append(
                    "| {dataset} | {strategy} | {rmse:.4f} | {mse:.4f} | {r2:.4f} | {fit:.4f} | {sample:.4f} | {inference:.4f} |".format(
                        dataset=dataset_name,
                        strategy=record.get("strategy", "-"),
                        rmse=float(metrics.get("rmse", float("nan"))),
                        mse=float(metrics.get("mse", float("nan"))),
                        r2=float(metrics.get("r2", float("nan"))),
                        fit=float(timings.get("fit", 0.0)),
                        sample=float(timings.get("sample", 0.0)),
                        inference=float(timings.get("inference", 0.0)),
                    )
                )

        lines.extend(["", "## Best / worst by dataset", ""])
        for dataset_name, dataset_records in grouped.items():
            first_type = _problem_type(dataset_records[0])
            if first_type == "regression":
                best = min(dataset_records, key=lambda r: float(r.get("model_metrics", {}).get("rmse", float("inf"))))
                worst = max(dataset_records, key=lambda r: float(r.get("model_metrics", {}).get("rmse", float("-inf"))))
                lines.append(f"### {dataset_name}")
                lines.append(f"- Best (RMSE): **{best.get('strategy', '-') }** ({float(best.get('model_metrics', {}).get('rmse', float('nan'))):.4f})")
                lines.append(f"- Worst (RMSE): **{worst.get('strategy', '-') }** ({float(worst.get('model_metrics', {}).get('rmse', float('nan'))):.4f})")
                lines.append("")
            else:
                has_binary_auc = any(
                    np.isfinite(
                        float(record.get("model_metrics", {}).get("roc_auc", float("nan")))
                    )
                    for record in dataset_records
                )
                primary_metric = "roc_auc" if has_binary_auc else "log_loss"
                if primary_metric == "roc_auc":
                    best = max(dataset_records, key=lambda r: float(r.get("model_metrics", {}).get(primary_metric, float("-inf"))))
                    worst = min(dataset_records, key=lambda r: float(r.get("model_metrics", {}).get(primary_metric, float("inf"))))
                else:
                    best = min(dataset_records, key=lambda r: float(r.get("model_metrics", {}).get(primary_metric, float("inf"))))
                    worst = max(dataset_records, key=lambda r: float(r.get("model_metrics", {}).get(primary_metric, float("-inf"))))
                lines.append(f"### {dataset_name}")
                lines.append(f"- Best ({primary_metric}): **{best.get('strategy', '-') }** ({float(best.get('model_metrics', {}).get(primary_metric, float('nan'))):.4f})")
                lines.append(f"- Worst ({primary_metric}): **{worst.get('strategy', '-') }** ({float(worst.get('model_metrics', {}).get(primary_metric, float('nan'))):.4f})")
                lines.append("")

        report_path.write_text("\n".join(lines), encoding="utf-8")
        return report_path


def build_sample_stats(
    y_sampled: Sequence[Any],
    total_train_size: int,
    cluster_labels: Optional[Sequence[Any]] = None,
    cell_ids: Optional[Sequence[Any]] = None,
    simplex_ids: Optional[Sequence[Any]] = None,
) -> Dict[str, Any]:
    y_array = np.asarray(y_sampled)
    unique, counts = np.unique(y_array, return_counts=True)

    stats: Dict[str, Any] = {
        "sample_size": int(y_array.shape[0]),
        "coverage_ratio": float(y_array.shape[0] / max(total_train_size, 1)),
        "class_coverage_count": int(unique.shape[0]),
        "class_distribution": {str(cls): int(cnt) for cls, cnt in zip(unique.tolist(), counts.tolist())},
    }

    if cluster_labels is not None:
        stats["cluster_distribution"] = _distribution(cluster_labels)
    if cell_ids is not None:
        stats["cell_distribution"] = _distribution(cell_ids)
    if simplex_ids is not None:
        stats["simplex_distribution"] = _distribution(simplex_ids)

    return stats


def build_2d_projection(x: np.ndarray, method: str = "pca") -> np.ndarray:
    method_normalized = method.lower()
    if method_normalized == "umap":
        try:
            import umap  # type: ignore

            reducer = umap.UMAP(n_components=2, random_state=42)
            return reducer.fit_transform(x)
        except Exception:
            method_normalized = "pca"

    if method_normalized == "pca":
        from sklearn.decomposition import PCA

        return PCA(n_components=2, random_state=42).fit_transform(x)

    raise ValueError(f"Unsupported projection method: {method}")


def _distribution(values: Sequence[Any]) -> Dict[str, int]:
    arr = np.asarray(values)
    unique, counts = np.unique(arr, return_counts=True)
    return {str(key): int(value) for key, value in zip(unique.tolist(), counts.tolist())}


def _safe_import_matplotlib():
    import matplotlib.pyplot as plt

    return plt


def _save_figure(fig: Any, base_path: Path) -> List[Path]:
    png_path = Path(f"{base_path}.png")
    svg_path = Path(f"{base_path}.svg")
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    fig.savefig(svg_path)

    import matplotlib.pyplot as plt

    plt.close(fig)
    return [png_path, svg_path]
