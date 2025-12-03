"""Конструкторы и структуры конфигурации AMLB экспериментов."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class SamplingStrategySpec:
    name: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AutoMLModelSpec:
    name: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetSpec:
    name: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentConfig:
    datasets: List[DatasetSpec]
    fedot_config: Dict
    sampling_strategies: List[SamplingStrategySpec]
    automl_models: List[AutoMLModelSpec]
    time_budget_minutes: int = 10
    tracking_uri: str | None = None
    experiment_name: str = "sampling-zoo-amlb"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "datasets": [dataset.__dict__ for dataset in self.datasets],
            "sampling_strategies": [strategy.__dict__ for strategy in self.sampling_strategies],
            "automl_models": [model.__dict__ for model in self.automl_models],
            "time_budget_minutes": self.time_budget_minutes,
            "tracking_uri": self.tracking_uri,
            "experiment_name": self.experiment_name,
        }


class ExperimentConfigBuilder:
    """Простой парсер текстовых запросов в структуру конфигурации."""

    def __init__(self, default_time_budget: int = 10):
        self.default_time_budget = default_time_budget

    def from_text(self, text: str, fedot_config: Dict) -> ExperimentConfig:
        sections = self._parse_sections(text)

        datasets = [DatasetSpec(name=item.strip()) for item in sections.get("datasets", []) if item.strip()]
        sampling = [self._parse_named_entity(SamplingStrategySpec, item) for item in sections.get("sampling", [])]
        models = [self._parse_named_entity(AutoMLModelSpec, item) for item in sections.get("models", [])]

        time_budget = int(sections.get("time_budget", [self.default_time_budget])[0])
        tracking_uri = sections.get("tracking_uri", [None])[0]
        experiment_name = sections.get("experiment_name", ["sampling-zoo-amlb"])[0]

        return ExperimentConfig(
            datasets=datasets,
            sampling_strategies=sampling or [SamplingStrategySpec(name="hierarchical_stratified", params={"n_splits": 5})],
            automl_models=models or [AutoMLModelSpec(name="fedot")],
            time_budget_minutes=time_budget,
            tracking_uri=tracking_uri,
            experiment_name=experiment_name,
            fedot_config=fedot_config
        )

    @staticmethod
    def example_request() -> str:
        return (
            "datasets: kddcup, airline\n"
            "sampling: hierarchical_stratified(n_splits=5)\n"
            "models: fedot(preset=auto)\n"
            "time_budget: 20\n"
            "tracking_uri: file:./mlruns\n"
        )

    def _parse_sections(self, text: str) -> Dict[str, List[str]]:
        sections: Dict[str, List[str]] = {}
        for raw_line in text.splitlines():
            if ":" not in raw_line:
                continue
            key, value = raw_line.split(":", maxsplit=1)
            key = key.strip().lower()
            values = [item.strip() for item in self._split_top_level_commas(value) if item.strip()]
            sections[key] = values
        return sections

    def _parse_named_entity(self, cls, token: str):
        token = token.strip()
        if "(" not in token or not token.endswith(")"):
            return cls(name=token.strip())

        start = token.find("(")
        end = token.rfind(")")
        if start == -1 or end == -1 or end <= start:
            return cls(name=token.strip())

        name = token[:start].strip()
        raw_params = token[start + 1:end].strip()

        params: Dict[str, Any] = {}
        if raw_params:
            pairs = self._split_top_level_commas(raw_params)
            for pair in pairs:
                pair = pair.strip()
                if not pair:
                    continue
                if "=" in pair:
                    param_key, param_value = pair.split("=", maxsplit=1)
                    params[param_key.strip()] = self._coerce_value(param_value.strip())

        return cls(name=name, params=params)

    @staticmethod
    def _split_top_level_commas(s: str) -> List[str]:
        parts: List[str] = []
        buf = []
        level = 0
        for ch in s:
            if ch == "(":
                level += 1
                buf.append(ch)
            elif ch == ")":
                level = max(level - 1, 0)
                buf.append(ch)
            elif ch == "," and level == 0:
                parts.append("".join(buf).strip())
                buf = []
            else:
                buf.append(ch)
        remaining = "".join(buf).strip()
        if remaining:
            parts.append(remaining)
        return parts

    @staticmethod
    def _coerce_value(raw: str) -> Any:
        if raw.isdigit():
            return int(raw)
        try:
            return float(raw)
        except ValueError:
            return raw
