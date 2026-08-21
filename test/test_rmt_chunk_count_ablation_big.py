from __future__ import annotations

import json

import pytest

from examples.benchmark.rmt_chunk_count_ablation_big_datasets import (
    DEFAULT_BIG_EXPERT_COUNT_BUDGETS,
    DEFAULT_BIG_EXPERT_COUNT_TASKS,
    DEFAULT_BIG_EXPERT_COUNTS,
    _plan_summary,
    make_big_foundation_config,
)
from examples.benchmark.rmt_chunk_count_ablation_medium_datasets import (
    make_chunk_count_ablation_strategy_configs,
)


def test_big_quick_foundation_profile_is_small_and_matched() -> None:
    config = make_big_foundation_config(show_progress=False)
    plan = _plan_summary(config)

    assert tuple(config.regression_tasks or ()) == DEFAULT_BIG_EXPERT_COUNT_TASKS
    assert tuple(config.budget_ratios) == DEFAULT_BIG_EXPERT_COUNT_BUDGETS
    assert tuple(config.chunk_counts) == DEFAULT_BIG_EXPERT_COUNTS
    assert tuple(config.models) == ("tabpfn", "tabicl")
    assert tuple(config.voting_pruning_modes) == (True, False)
    assert tuple(config.regression_tasks or ()) == (
        "black_friday",
        "Allstate_Claims_Severity",
        "Yolanda",
        "Buzzinsocialmedia_Twitter",
        "nyc-taxi-green-dec-2016",
        "Airlines_DepDelay_10M",
    )
    assert plan["leaf_runs"] == 108
    assert plan["actual_expert_fits_with_cache"] == 168
    json.dumps(plan)

    strategies = make_chunk_count_ablation_strategy_configs(config)
    assert len(strategies) == 9
    for strategy in strategies.values():
        assert strategy["chunks_percent"] == pytest.approx(100.0)


def test_big_profile_rejects_medium_task() -> None:
    with pytest.raises(ValueError, match="Unknown big tasks"):
        make_big_foundation_config(
            tasks=("Brazilian_houses",),
            show_progress=False,
        )
