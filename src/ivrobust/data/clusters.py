"""Cluster specifications and validation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ivrobust.utils.arrays import as_1d_array


@dataclass(frozen=True)
class ClusterSpec:
    """Cluster metadata for one-way clustering."""

    ids: np.ndarray
    codes: np.ndarray
    unique_ids: tuple[object, ...]
    n_clusters: int
    min_size: int
    max_size: int

    @classmethod
    def from_array(cls, clusters: ArrayLike) -> ClusterSpec:
        ids = as_1d_array(clusters, "clusters")
        unique_ids, codes = np.unique(ids, return_inverse=True)
        counts = np.bincount(codes)
        return cls(
            ids=ids,
            codes=codes,
            unique_ids=tuple(unique_ids.tolist()),
            n_clusters=int(unique_ids.size),
            min_size=int(counts.min()) if counts.size > 0 else 0,
            max_size=int(counts.max()) if counts.size > 0 else 0,
        )
