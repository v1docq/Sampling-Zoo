"""Scalable numerical representation for cluster consensus."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy import sparse

from .cluster_selection_contracts import ClusterConsensusPlan


def build_weighted_membership_embedding(
    labels_by_source: Sequence[np.ndarray],
    plan: ClusterConsensusPlan,
) -> sparse.csr_matrix:
    """Build H where H H^T is the weighted co-association matrix."""

    labels = tuple(np.asarray(values) for values in labels_by_source)
    if len(labels) != len(plan.sources):
        raise ValueError("labels_by_source must align with consensus plan sources")
    n_samples = _validate_source_labels(labels, plan)
    rows = np.arange(n_samples, dtype=int)
    blocks = []
    for source_labels, source in zip(labels, plan.sources):
        _, inverse = np.unique(source_labels, return_inverse=True)
        block = sparse.csr_matrix(
            (
                np.full(n_samples, np.sqrt(source.weight), dtype=float),
                (rows, inverse),
            ),
            shape=(n_samples, source.n_clusters),
        )
        block.eliminate_zeros()
        blocks.append(block)
    return sparse.hstack(blocks, format="csr")


def _validate_source_labels(
    labels_by_source: Sequence[np.ndarray],
    plan: ClusterConsensusPlan,
) -> int:
    n_samples = int(labels_by_source[0].size)
    for labels, source in zip(labels_by_source, plan.sources):
        if labels.ndim != 1:
            raise ValueError("consensus source labels must be one-dimensional")
        if labels.size != n_samples:
            raise ValueError("all consensus source labels must have equal lengths")
        if np.unique(labels).size != source.n_clusters:
            raise ValueError(
                "consensus source labels do not match the planned cluster count"
            )
    return n_samples
