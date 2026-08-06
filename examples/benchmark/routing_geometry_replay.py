"""Replay Phase A routing arms over one fitted RMT chunk ensemble."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from sampling_zoo.core.experiment.routing_replay import (
    RoutingGeometrySelection,
    RoutingGeometrySelectionPolicy,
    RoutingReplayEvaluator,
    RoutingReplayRequest,
    RoutingReplayResult,
    ValidationRoutingGeometrySelector,
)
from sampling_zoo.core.sampling_strategies.spectral.routing_contracts import (
    PartitionGeometrySpec,
    RoutingDistanceContract,
)


DEFAULT_ROUTING_TEMPERATURES: tuple[float, ...] = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0)
DEFAULT_VALIDATION_SELECTOR_ARM_NAMES: tuple[str, ...] = (
    "A2_median_scaled_euclidean",
    "A5_gmm_posterior",
    "A6_cosine_negative_control",
)


@dataclass(frozen=True)
class RoutingGeometryArm:
    name: str
    spec: PartitionGeometrySpec
    temperature_candidates: tuple[float, ...] = DEFAULT_ROUTING_TEMPERATURES
    use_validation_priors: bool = True


def default_routing_geometry_arms() -> tuple[RoutingGeometryArm, ...]:
    """Return the agreed Phase A screening grid in stable execution order."""

    return (
        RoutingGeometryArm(
            name="A0_uniform_voting",
            spec=PartitionGeometrySpec(uniform_shrinkage=1.0),
            temperature_candidates=(1.0,),
            use_validation_priors=False,
        ),
        RoutingGeometryArm(
            name="A1_source_centroid_squared_euclidean",
            spec=PartitionGeometrySpec(),
            temperature_candidates=(1.0,),
        ),
        RoutingGeometryArm(
            name="A2_median_scaled_euclidean",
            spec=PartitionGeometrySpec(metric="median_scaled_euclidean"),
        ),
        RoutingGeometryArm(
            name="A3_diag_shrinkage_mahalanobis",
            spec=PartitionGeometrySpec(metric="diag_shrinkage_mahalanobis"),
        ),
        RoutingGeometryArm(
            name="A4_full_shrinkage_mahalanobis",
            spec=PartitionGeometrySpec(metric="full_shrinkage_mahalanobis"),
        ),
        RoutingGeometryArm(
            name="A5_gmm_posterior",
            spec=PartitionGeometrySpec(
                metric="full_shrinkage_mahalanobis",
                kernel="gmm_posterior",
            ),
        ),
        RoutingGeometryArm(
            name="A6_cosine_negative_control",
            spec=PartitionGeometrySpec(metric="cosine"),
        ),
    )


def default_validation_selector_arms() -> tuple[RoutingGeometryArm, ...]:
    """Return the selector candidates identified by the Phase A screen."""

    by_name = {arm.name: arm for arm in default_routing_geometry_arms()}
    return tuple(by_name[name] for name in DEFAULT_VALIDATION_SELECTOR_ARM_NAMES)


def default_validation_selector_policy(
    *,
    minimum_improvement: float = 0.0,
) -> RoutingGeometrySelectionPolicy:
    return RoutingGeometrySelectionPolicy(
        candidate_arm_names=DEFAULT_VALIDATION_SELECTOR_ARM_NAMES,
        fallback_arm_name="A2_median_scaled_euclidean",
        minimum_improvement=minimum_improvement,
    )


class FittedEnsembleRoutingReplay:
    """Adapter from a fitted SamplingEnsemble to pure routing replay contracts."""

    def __init__(
        self,
        evaluator: Optional[RoutingReplayEvaluator] = None,
        *,
        show_progress: bool = True,
    ) -> None:
        self.evaluator = evaluator or RoutingReplayEvaluator()
        self.show_progress = bool(show_progress)

    def run_validation(
        self,
        *,
        ensemble: Any,
        X_val: pd.DataFrame,
        y_val: Any,
        arms: Optional[Sequence[RoutingGeometryArm]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> list[RoutingReplayResult]:
        context = self._prepare_context(
            ensemble=ensemble,
            features=X_val,
            stage="validation",
        )
        selected_arms = tuple(arms or default_routing_geometry_arms())
        results = []
        for arm in self._iter_arms(selected_arms):
            request = self._build_request(
                ensemble=ensemble,
                target=y_val,
                arm=arm,
                context=context,
                metadata=metadata,
            )
            selection = self.evaluator.calibrate_temperature(
                request,
                arm.temperature_candidates,
            )
            results.append(selection.best)
        return results

    def run_evaluation(
        self,
        *,
        ensemble: Any,
        features: pd.DataFrame,
        target: Any,
        arms: Sequence[RoutingGeometryArm],
        temperatures: Mapping[str, float],
        metadata: Optional[dict[str, Any]] = None,
    ) -> list[RoutingReplayResult]:
        """Evaluate validation-selected arms on held-out rows without recalibration."""

        context = self._prepare_context(
            ensemble=ensemble,
            features=features,
            stage="inference",
        )
        results = []
        for arm in self._iter_arms(tuple(arms)):
            if arm.name not in temperatures:
                raise ValueError(f"Missing selected temperature for arm {arm.name!r}")
            request = self._build_request(
                ensemble=ensemble,
                target=target,
                arm=arm,
                context=context,
                metadata=metadata,
            )
            results.append(
                self.evaluator.evaluate(
                    request,
                    temperature=float(temperatures[arm.name]),
                )
            )
        return results

    @staticmethod
    def select_validation_geometry(
        results: Sequence[RoutingReplayResult],
        *,
        policy: Optional[RoutingGeometrySelectionPolicy] = None,
        selector: Optional[ValidationRoutingGeometrySelector] = None,
    ) -> RoutingGeometrySelection:
        """Select one geometry from held-out validation results."""

        return (selector or ValidationRoutingGeometrySelector()).select(
            results,
            policy or default_validation_selector_policy(),
        )

    def _prepare_context(
        self,
        *,
        ensemble: Any,
        features: pd.DataFrame,
        stage: str,
    ) -> dict[str, Any]:
        partitioner = getattr(ensemble, "partitioner", None)
        if partitioner is None or not hasattr(partitioner, "transform_embedding"):
            raise ValueError("Routing replay requires a fitted RMT partitioner")

        model_names, expert_outputs = ensemble.export_expert_outputs(
            features,
            stage=stage,
        )
        embedding = partitioner.transform_embedding(features)
        validation_priors = ensemble.validation_prior_weights()
        raw_classes = getattr(ensemble, "classes_", None)
        classes = () if raw_classes is None else tuple(np.asarray(raw_classes).tolist())
        return {
            "partitioner": partitioner,
            "model_names": model_names,
            "expert_outputs": expert_outputs,
            "embedding": embedding,
            "validation_priors": validation_priors,
            "classes": classes,
        }

    def _build_request(
        self,
        *,
        ensemble: Any,
        target: Any,
        arm: RoutingGeometryArm,
        context: Mapping[str, Any],
        metadata: Optional[dict[str, Any]],
    ) -> RoutingReplayRequest:
        partitioner = context["partitioner"]
        geometry = partitioner.build_partition_geometry(arm.spec)
        distances, _weights = partitioner.route_embedding(
            context["embedding"],
            geometry=geometry,
        )
        aligned_distances = self._align_distances(
            distances,
            context["model_names"],
        )
        return RoutingReplayRequest(
            arm_name=arm.name,
            geometry_spec=arm.spec,
            distances=aligned_distances,
            target=np.asarray(target),
            expert_outputs=context["expert_outputs"],
            problem_type=str(ensemble.problem),
            classes=context["classes"],
            expert_priors=(
                context["validation_priors"]
                if arm.use_validation_priors
                else None
            ),
            metadata=metadata or {},
        )

    def _iter_arms(self, arms: Sequence[RoutingGeometryArm]):
        return tqdm(
            arms,
            desc="Replay routing geometry",
            disable=not self.show_progress,
            leave=False,
        )

    @staticmethod
    def records(
        results: Iterable[RoutingReplayResult],
        *,
        metadata: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        common = dict(metadata or {})
        return [{**common, **result.summary()} for result in results]

    @staticmethod
    def _align_distances(
        distances: RoutingDistanceContract,
        model_names: Sequence[str],
    ) -> RoutingDistanceContract:
        column_by_name = {
            name: idx for idx, name in enumerate(distances.partition_names)
        }
        missing = [name for name in model_names if name not in column_by_name]
        if missing:
            raise ValueError(f"Expert names are missing from partition geometry: {missing}")
        columns = [column_by_name[name] for name in model_names]
        return RoutingDistanceContract(
            partition_names=tuple(model_names),
            values=distances.values[:, columns],
            kind=distances.kind,
            diagnostics={
                **dict(distances.diagnostics),
                "aligned_from_partition_count": len(distances.partition_names),
            },
        )
