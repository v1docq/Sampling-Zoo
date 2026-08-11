from __future__ import annotations

import random

import pytest

from sampling_zoo.core.experiment.routing_geometry_selection import (
    CrossFittedRoutingGeometrySelectionSpec,
    CrossFittedRoutingGeometrySelector,
    RoutingGeometryFoldScore,
    RoutingGeometryTailRiskScore,
)


REFERENCE = "A2_median_scaled_euclidean"
CANDIDATE = "A5_gmm_posterior"


def _regression_score(
    arm_name: str,
    fold: int,
    *,
    rmse: float,
    mae: float,
    tail_error: float | None = None,
) -> RoutingGeometryFoldScore:
    return RoutingGeometryFoldScore(
        arm_name=arm_name,
        fold_id=str(fold),
        primary_metric="rmse",
        primary_value=rmse,
        primary_direction="lower",
        robust_metric="mae",
        robust_value=mae,
        robust_direction="lower",
        tail_risk=(
            RoutingGeometryTailRiskScore(
                metric="tail_mean_absolute_error",
                value=tail_error,
                direction="lower",
                quantile=0.90,
            )
            if tail_error is not None
            else None
        ),
        evaluation_rows=20,
        selected_temperature=0.5,
    )


def _selector(**overrides) -> CrossFittedRoutingGeometrySelector:
    params = {
        "reference_arm": REFERENCE,
        "candidate_arms": (CANDIDATE,),
        "min_folds": 3,
        "bootstrap_iterations": 500,
        "require_robust_metric": True,
        "random_state": 17,
    }
    params.update(overrides)
    return CrossFittedRoutingGeometrySelector(
        CrossFittedRoutingGeometrySelectionSpec(**params)
    )


def test_selector_chooses_candidate_with_stable_primary_and_robust_gain() -> None:
    scores = []
    for fold in range(5):
        scores.extend(
            (
                _regression_score(REFERENCE, fold, rmse=10.0, mae=8.0),
                _regression_score(CANDIDATE, fold, rmse=9.0, mae=7.5),
            )
        )

    decision = _selector().select(scores)

    assert decision.status == "selected"
    assert decision.selected_arm == CANDIDATE
    assert decision.evidence[0].positive_fold_fraction == 1.0
    assert decision.evidence[0].confidence_interval[0] > 0.0


def test_selector_falls_back_when_primary_gain_changes_sign() -> None:
    candidate_rmse = (9.0, 9.0, 11.0, 11.0, 11.0)
    scores = []
    for fold, rmse in enumerate(candidate_rmse):
        scores.extend(
            (
                _regression_score(REFERENCE, fold, rmse=10.0, mae=8.0),
                _regression_score(CANDIDATE, fold, rmse=rmse, mae=8.0),
            )
        )

    decision = _selector().select(scores)

    assert decision.status == "fallback_to_a2"
    assert decision.selected_arm == REFERENCE
    assert "unstable_primary_gain_sign" in decision.evidence[0].rejection_reasons


def test_selector_robust_gate_blocks_rmse_only_improvement() -> None:
    scores = []
    for fold in range(5):
        scores.extend(
            (
                _regression_score(REFERENCE, fold, rmse=10.0, mae=8.0),
                _regression_score(CANDIDATE, fold, rmse=9.0, mae=9.0),
            )
        )

    decision = _selector().select(scores)

    assert decision.status == "fallback_to_a2"
    assert any(
        reason.startswith("robust_")
        for reason in decision.evidence[0].rejection_reasons
    )


def test_tail_gate_blocks_candidate_that_improves_rmse_and_mae() -> None:
    scores = []
    for fold in range(5):
        scores.extend(
            (
                _regression_score(
                    REFERENCE,
                    fold,
                    rmse=10.0,
                    mae=8.0,
                    tail_error=20.0,
                ),
                _regression_score(
                    CANDIDATE,
                    fold,
                    rmse=9.0,
                    mae=7.5,
                    tail_error=25.0,
                ),
            )
        )

    decision = _selector(tail_guard_primary_metrics=("rmse",)).select(scores)

    assert decision.status == "fallback_to_a2"
    assert decision.selected_arm == REFERENCE
    assert "tail_mean_exceeds_harm_margin" in decision.evidence[0].rejection_reasons
    assert decision.evidence[0].tail_mean_relative_gain == pytest.approx(-0.25)


def test_tail_diagnostics_do_not_change_a8_without_tail_gate() -> None:
    scores = []
    for fold in range(5):
        scores.extend(
            (
                _regression_score(
                    REFERENCE,
                    fold,
                    rmse=10.0,
                    mae=8.0,
                    tail_error=20.0,
                ),
                _regression_score(
                    CANDIDATE,
                    fold,
                    rmse=9.0,
                    mae=7.5,
                    tail_error=25.0,
                ),
            )
        )

    decision = _selector().select(scores)

    assert decision.status == "selected"
    assert decision.selected_arm == CANDIDATE
    assert decision.evidence[0].tail_mean_relative_gain == pytest.approx(-0.25)


def test_required_tail_metric_must_be_present_for_rmse() -> None:
    scores = []
    for fold in range(5):
        scores.extend(
            (
                _regression_score(REFERENCE, fold, rmse=10.0, mae=8.0),
                _regression_score(CANDIDATE, fold, rmse=9.0, mae=7.5),
            )
        )

    decision = _selector(tail_guard_primary_metrics=("rmse",)).select(scores)

    assert decision.status == "fallback_to_a2"
    assert "tail_metric_required" in decision.evidence[0].rejection_reasons


def test_higher_is_better_metric_is_converted_to_positive_gain() -> None:
    scores = []
    for fold in range(3):
        scores.extend(
            (
                RoutingGeometryFoldScore(
                    arm_name=REFERENCE,
                    fold_id=str(fold),
                    primary_metric="roc_auc",
                    primary_value=0.80,
                    primary_direction="higher",
                    evaluation_rows=30,
                    selected_temperature=1.0,
                ),
                RoutingGeometryFoldScore(
                    arm_name=CANDIDATE,
                    fold_id=str(fold),
                    primary_metric="roc_auc",
                    primary_value=0.83,
                    primary_direction="higher",
                    evaluation_rows=30,
                    selected_temperature=1.0,
                ),
            )
        )
    selector = _selector(
        require_robust_metric=False,
        tail_guard_primary_metrics=("rmse",),
    )

    decision = selector.select(scores)

    assert decision.status == "selected"
    assert decision.evidence[0].mean_relative_gain == pytest.approx(0.0375)


def test_selection_is_invariant_to_score_order() -> None:
    scores = []
    for fold in range(5):
        scores.extend(
            (
                _regression_score(REFERENCE, fold, rmse=10.0, mae=8.0),
                _regression_score(CANDIDATE, fold, rmse=9.5, mae=7.8),
            )
        )
    shuffled = list(scores)
    random.Random(91).shuffle(shuffled)
    selector = _selector()

    original = selector.select(scores)
    permuted = selector.select(shuffled)

    assert original.summary() == permuted.summary()


def test_duplicate_fold_score_is_rejected() -> None:
    score = _regression_score(REFERENCE, 0, rmse=10.0, mae=8.0)

    with pytest.raises(ValueError, match="Duplicate fold score"):
        _selector().select((score, score))
