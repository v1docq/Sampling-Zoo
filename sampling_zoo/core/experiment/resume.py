"""Pure contracts and decisions for idempotent experiment resume."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

from .errors import ResumeContractError, ResumeRecordError


class ResumePolicy(str, Enum):
    RETRY_FAILED = "retry_failed"
    SKIP_EXISTING = "skip_existing"

    @classmethod
    def parse(cls, value: str | "ResumePolicy") -> "ResumePolicy":
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value))
        except ValueError as exc:
            raise ResumeContractError(
                scope="experiment.resume.policy",
                code="invalid_resume_policy",
                message=(
                    "resume_policy must be one of: "
                    f"{', '.join(policy.value for policy in cls)}"
                ),
                details={"value": value},
            ) from exc


class LeafRunOutcome(str, Enum):
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class LeafRunKey:
    dataset: str
    split: str
    model: str
    strategy: str
    budget_ratio: float | None
    router: str | None
    view_strategy: str | None
    seed: int

    def __post_init__(self) -> None:
        for field_name in ("dataset", "split", "model", "strategy"):
            value = getattr(self, field_name)
            if not str(value).strip():
                raise ValueError(f"{field_name} must be non-empty")
        if self.budget_ratio is not None:
            if not math.isfinite(self.budget_ratio):
                raise ValueError("budget_ratio must be finite")
            if not 0 < self.budget_ratio <= 1:
                raise ValueError("budget_ratio must be in (0, 1]")

    @property
    def key(self) -> str:
        encoded = json.dumps(
            self.identity_payload(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def identity_payload(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "split": self.split,
            "model": self.model,
            "strategy": self.strategy,
            "budget_ratio": self.budget_ratio,
            "router": self.router,
            "view_strategy": self.view_strategy,
            "seed": self.seed,
        }

    def to_dict(self) -> dict[str, Any]:
        return {"key": self.key, **self.identity_payload()}


@dataclass(frozen=True)
class ResumePlan:
    policy: ResumePolicy
    source_records: tuple[Mapping[str, Any], ...]
    retained_records: tuple[Mapping[str, Any], ...]
    completed_keys: frozenset[str]
    failed_keys: frozenset[str]
    skipped_keys: frozenset[str]
    retry_keys: frozenset[str]
    duplicate_records_dropped: int = 0

    def should_execute(self, leaf_run: LeafRunKey) -> bool:
        return leaf_run.key not in self.skipped_keys

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy.value,
            "source_records": len(self.source_records),
            "retained_records": len(self.retained_records),
            "completed_keys": len(self.completed_keys),
            "failed_keys": len(self.failed_keys),
            "skipped_keys": len(self.skipped_keys),
            "retry_keys": len(self.retry_keys),
            "duplicate_records_dropped": self.duplicate_records_dropped,
        }


def scientific_experiment_config(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    runtime_only_fields = {
        "resume_from",
        "resume_policy",
        "show_progress",
    }
    return {
        str(key): value
        for key, value in config.items()
        if key not in runtime_only_fields
    }


def legacy_resume_config(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in config.items()
        if key not in {"resume_from", "resume_policy"}
    }


def leaf_run_key_from_components(
    *,
    dataset: str,
    split: str,
    model: str,
    strategy: str,
    strategy_config: Mapping[str, Any],
    seed: int,
) -> LeafRunKey:
    return LeafRunKey(
        dataset=str(dataset),
        split=str(split),
        model=str(model),
        strategy=str(strategy),
        budget_ratio=_optional_float(
            strategy_config.get("budget_ratio")
        ),
        router=_optional_text(strategy_config.get("router")),
        view_strategy=_optional_text(
            strategy_config.get("view_strategy")
        ),
        seed=int(strategy_config.get("random_state", seed)),
    )


def leaf_run_key_from_record(
    record: Mapping[str, Any],
    *,
    default_seed: int,
) -> LeafRunKey:
    extra = _mapping(record.get("extra"))
    embedded = _mapping(extra.get("leaf_run"))
    if embedded:
        leaf_run = _leaf_run_from_payload(embedded)
        persisted_key = _optional_text(extra.get("leaf_run_key"))
        if persisted_key is not None and persisted_key != leaf_run.key:
            raise ValueError(
                "Persisted leaf_run_key does not match leaf_run payload"
            )
        return leaf_run

    strategy_params = _mapping(record.get("strategy_params"))
    dataset = _required_text(record.get("dataset"), "dataset")
    split = _required_text(
        extra.get("split_label", strategy_params.get("split_label")),
        "split",
    )
    model = _required_text(
        extra.get("model", strategy_params.get("model")),
        "model",
    )
    strategy = _required_text(
        extra.get("strategy", strategy_params.get("strategy")),
        "strategy",
    )
    return leaf_run_key_from_components(
        dataset=dataset,
        split=split,
        model=model,
        strategy=strategy,
        strategy_config=strategy_params,
        seed=default_seed,
    )


def run_record_outcome(record: Mapping[str, Any]) -> LeafRunOutcome:
    extra = _mapping(record.get("extra"))
    if (
        _optional_text(extra.get("error")) is not None
        or _optional_text(extra.get("error_code")) is not None
        or str(record.get("status", "")).lower() == "failed"
    ):
        return LeafRunOutcome.FAILED
    return LeafRunOutcome.COMPLETED


def build_resume_plan(
    records: Sequence[Mapping[str, Any]],
    *,
    policy: str | ResumePolicy,
    default_seed: int,
) -> ResumePlan:
    normalized_policy = ResumePolicy.parse(policy)
    indexed_by_key: dict[
        str,
        tuple[int, Mapping[str, Any], LeafRunOutcome],
    ] = {}
    duplicate_count = 0
    normalized_source: list[Mapping[str, Any]] = []

    for index, raw_record in enumerate(records):
        record = dict(raw_record)
        normalized_source.append(record)
        try:
            leaf_run = leaf_run_key_from_record(
                record,
                default_seed=default_seed,
            )
        except (TypeError, ValueError) as exc:
            raise ResumeRecordError(
                scope=f"experiment.resume.records.{index}",
                code="invalid_resume_record",
                message=str(exc),
                details={"record_index": index},
            ) from exc
        outcome = run_record_outcome(record)
        previous = indexed_by_key.get(leaf_run.key)
        if previous is not None:
            duplicate_count += 1
        if previous is None or _prefer_record(
            previous_outcome=previous[2],
            current_outcome=outcome,
        ):
            indexed_by_key[leaf_run.key] = (index, record, outcome)

    selected = sorted(
        indexed_by_key.items(),
        key=lambda item: item[1][0],
    )
    completed_keys = frozenset(
        key
        for key, (_, _, outcome) in selected
        if outcome == LeafRunOutcome.COMPLETED
    )
    failed_keys = frozenset(
        key
        for key, (_, _, outcome) in selected
        if outcome == LeafRunOutcome.FAILED
    )

    if normalized_policy == ResumePolicy.RETRY_FAILED:
        retained_records = tuple(
            record
            for _, (_, record, outcome) in selected
            if outcome == LeafRunOutcome.COMPLETED
        )
        skipped_keys = completed_keys
        retry_keys = failed_keys
    else:
        retained_records = tuple(
            record for _, (_, record, _) in selected
        )
        skipped_keys = completed_keys | failed_keys
        retry_keys = frozenset()

    return ResumePlan(
        policy=normalized_policy,
        source_records=tuple(normalized_source),
        retained_records=retained_records,
        completed_keys=completed_keys,
        failed_keys=failed_keys,
        skipped_keys=skipped_keys,
        retry_keys=retry_keys,
        duplicate_records_dropped=duplicate_count,
    )


def _prefer_record(
    *,
    previous_outcome: LeafRunOutcome,
    current_outcome: LeafRunOutcome,
) -> bool:
    if previous_outcome == LeafRunOutcome.COMPLETED:
        return current_outcome == LeafRunOutcome.COMPLETED
    return True


def _leaf_run_from_payload(payload: Mapping[str, Any]) -> LeafRunKey:
    return LeafRunKey(
        dataset=_required_text(payload.get("dataset"), "dataset"),
        split=_required_text(payload.get("split"), "split"),
        model=_required_text(payload.get("model"), "model"),
        strategy=_required_text(payload.get("strategy"), "strategy"),
        budget_ratio=_optional_float(payload.get("budget_ratio")),
        router=_optional_text(payload.get("router")),
        view_strategy=_optional_text(payload.get("view_strategy")),
        seed=int(payload.get("seed")),
    )


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _required_text(value: Any, field_name: str) -> str:
    text = _optional_text(value)
    if text is None:
        raise ValueError(f"{field_name} must be present and non-empty")
    return text


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)
