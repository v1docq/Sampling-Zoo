from __future__ import annotations

import random

import pytest

from sampling_zoo.core.experiment.row_policy_selection import (
    CrossFittedRowPolicySelectionSpec,
    CrossFittedRowPolicySelector,
    RowPolicyFoldScore,
    RowSamplingPolicyRequest,
    RowSamplingPolicySpec,
)


UNIFORM = RowSamplingPolicySpec("U0_uniform", "uniform")
CAPPED = RowSamplingPolicySpec("C1_capped_leverage", "capped_leverage")


def _score(
    policy: RowSamplingPolicySpec,
    fold: int,
    *,
    primary: float,
    tail: float | None = None,
    coverage: bool | None = None,
) -> RowPolicyFoldScore:
    return RowPolicyFoldScore(
        policy_name=policy.name,
        fold_id=str(fold),
        primary_metric="rmse",
        primary_value=primary,
        primary_direction="lower",
        tail_metric="tail_mean_absolute_error" if tail is not None else None,
        tail_value=tail,
        tail_direction="lower" if tail is not None else None,
        class_coverage_preserved=coverage,
        evaluation_rows=25,
    )


def _request(scores) -> RowSamplingPolicyRequest:
    return RowSamplingPolicyRequest(
        reference_policy=UNIFORM,
        candidate_policy=CAPPED,
        fold_scores=tuple(scores),
    )


def _selector() -> CrossFittedRowPolicySelector:
    return CrossFittedRowPolicySelector(
        CrossFittedRowPolicySelectionSpec(
            min_folds=3,
            bootstrap_iterations=500,
            random_state=17,
        )
    )


def test_selector_enables_capped_leverage_on_stable_gain() -> None:
    scores = []
    for fold in range(5):
        scores.extend(
            (
                _score(UNIFORM, fold, primary=10.0, tail=20.0),
                _score(CAPPED, fold, primary=9.5, tail=19.5),
            )
        )

    result = _selector().select(_request(scores))

    assert result.status == "selected"
    assert result.selected_policy == CAPPED
    assert result.evidence.positive_fold_fraction == 1.0


def test_selector_falls_back_when_less_than_two_thirds_of_folds_win() -> None:
    candidate = (9.5, 9.5, 10.5, 10.5, 10.5)
    scores = []
    for fold, value in enumerate(candidate):
        scores.extend(
            (
                _score(UNIFORM, fold, primary=10.0),
                _score(CAPPED, fold, primary=value),
            )
        )

    result = _selector().select(_request(scores))

    assert result.status == "fallback_to_reference"
    assert "insufficient_positive_fold_fraction" in result.evidence.rejection_reasons


def test_tail_guard_blocks_primary_only_improvement() -> None:
    scores = []
    for fold in range(5):
        scores.extend(
            (
                _score(UNIFORM, fold, primary=10.0, tail=20.0),
                _score(CAPPED, fold, primary=9.5, tail=21.0),
            )
        )

    result = _selector().select(_request(scores))

    assert result.status == "fallback_to_reference"
    assert any(reason.startswith("tail_") for reason in result.evidence.rejection_reasons)


def test_class_coverage_guard_blocks_candidate() -> None:
    scores = []
    for fold in range(5):
        scores.extend(
            (
                _score(UNIFORM, fold, primary=10.0, coverage=True),
                _score(
                    CAPPED,
                    fold,
                    primary=9.5,
                    coverage=fold != 2,
                ),
            )
        )

    result = _selector().select(_request(scores))

    assert result.status == "fallback_to_reference"
    assert "class_coverage_not_preserved" in result.evidence.rejection_reasons


def test_selection_is_invariant_to_fold_score_order() -> None:
    scores = []
    for fold in range(5):
        scores.extend(
            (
                _score(UNIFORM, fold, primary=10.0),
                _score(CAPPED, fold, primary=9.5),
            )
        )
    shuffled = list(scores)
    random.Random(91).shuffle(shuffled)

    original = _selector().select(_request(scores))
    permuted = _selector().select(_request(shuffled))

    assert original.summary() == permuted.summary()


def test_duplicate_fold_score_is_rejected() -> None:
    score = _score(UNIFORM, 0, primary=10.0)

    with pytest.raises(ValueError, match="Duplicate row policy fold score"):
        _selector().select(_request((score, score)))
