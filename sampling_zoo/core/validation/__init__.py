"""Scientific validation primitives for Sampling Zoo algorithms."""

from .multi_regime_models import (
    ClusterRecoveryMetrics,
    MultiRegimeDataConfig,
    MultiRegimeDataset,
    MultiRegimeValidationGridPoint,
    PartitionProbePolicy,
    RegimeProfile,
    build_multi_regime_validation_grid,
    evaluate_cluster_recovery,
    generate_multi_regime_dataset,
)

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
    "ClusterRecoveryMetrics",
    "MultiRegimeDataConfig",
    "MultiRegimeDataset",
    "MultiRegimeValidationGridPoint",
    "NoiseDistribution",
    "PartitionProbePolicy",
    "RankRecoveryMetrics",
    "RegimeProfile",
    "SpikedDataConfig",
    "SpikedDataset",
    "SpikedRecoveryEvaluation",
    "SpikedValidationGridPoint",
    "SubspaceRecoveryMetrics",
    "build_multi_regime_validation_grid",
    "build_spiked_validation_grid",
    "evaluate_cluster_recovery",
    "evaluate_rank_recovery",
    "evaluate_spiked_recovery",
    "evaluate_subspace_recovery",
    "generate_multi_regime_dataset",
    "generate_spiked_dataset",
]
