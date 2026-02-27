from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
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
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


@dataclass
class ArtifactPaths:
    root: Path
    logs: Path
    metrics: Path
    plots: Path
    samples: Path


class BenchmarkLogger:
    """Utility to persist benchmark artifacts and strategy run logs."""

    def __init__(self, run_id: Optional[str] = None, artifacts_root: str | Path = "artifacts/benchmarks") -> None:
        self.run_id = run_id or self._generate_run_id()
        self.paths = self._init_artifacts(artifacts_root=artifacts_root, run_id=self.run_id)
        self.jsonl_path = self.paths.logs / "strategy_runs.jsonl"

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

        lines = [
            f"# Benchmark report ({self.run_id})",
            "",
            "## Strategy comparison",
            "",
            "| Dataset | Strategy | ROC-AUC | F1 macro | F1 weighted | fit(s) | sample(s) | inference(s) |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]

        grouped: Dict[str, List[Mapping[str, Any]]] = {}
        for record in records:
            dataset_name = str(record.get("dataset", "unknown"))
            grouped.setdefault(dataset_name, []).append(record)

            metrics = record.get("model_metrics", {})
            timings = record.get("timings_sec", {})
            lines.append(
                "| {dataset} | {strategy} | {roc_auc:.4f} | {f1_macro:.4f} | {f1_weighted:.4f} | {fit:.4f} | {sample:.4f} | {inference:.4f} |".format(
                    dataset=dataset_name,
                    strategy=record.get("strategy", "-"),
                    roc_auc=float(metrics.get("roc_auc", float("nan"))),
                    f1_macro=float(metrics.get("f1_macro", float("nan"))),
                    f1_weighted=float(metrics.get("f1_weighted", float("nan"))),
                    fit=float(timings.get("fit", 0.0)),
                    sample=float(timings.get("sample", 0.0)),
                    inference=float(timings.get("inference", 0.0)),
                )
            )

        lines.extend(["", "## Best / worst by dataset", ""])
        for dataset_name, dataset_records in grouped.items():
            best = max(dataset_records, key=lambda r: float(r.get("model_metrics", {}).get("f1_macro", float("-inf"))))
            worst = min(dataset_records, key=lambda r: float(r.get("model_metrics", {}).get("f1_macro", float("inf"))))

            lines.append(f"### {dataset_name}")
            lines.append(f"- Best (F1 macro): **{best.get('strategy', '-') }** ({float(best.get('model_metrics', {}).get('f1_macro', 0.0)):.4f})")
            lines.append(f"- Worst (F1 macro): **{worst.get('strategy', '-') }** ({float(worst.get('model_metrics', {}).get('f1_macro', 0.0)):.4f})")
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
