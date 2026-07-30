"""Scientific validation primitives for Sampling Zoo algorithms."""

from .spiked_models import (
    NoiseDistribution,
    RankRecoveryMetrics,
    SpikedDataConfig,
    SpikedDataset,
    SpikedRecoveryEvaluation,
    SpikedValidationGridPoint,
    SubspaceRecoveryMetrics,
    build_spiked_validation_grid,
    evaluate_rank_recovery,
    evaluate_spiked_recovery,
    evaluate_subspace_recovery,
    generate_spiked_dataset,
)

__all__ = [
    "NoiseDistribution",
    "RankRecoveryMetrics",
    "SpikedDataConfig",
    "SpikedDataset",
    "SpikedRecoveryEvaluation",
    "SpikedValidationGridPoint",
    "SubspaceRecoveryMetrics",
    "build_spiked_validation_grid",
    "evaluate_rank_recovery",
    "evaluate_spiked_recovery",
    "evaluate_subspace_recovery",
    "generate_spiked_dataset",
]
