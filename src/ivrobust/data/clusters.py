from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .._typing import IntArray


@dataclass(frozen=True)
class ClusterSpec:
    """
    Normalized cluster labels.

    codes are integer arrays with contiguous labels starting at 0.
    """

    codes: tuple[IntArray, ...]
    n_clusters: tuple[int, ...]
    nobs: int

    @property
    def is_multiway(self) -> bool:
        return len(self.codes) > 1


def _as_1d_int(x: np.ndarray) -> IntArray:
    arr = np.asarray(x)
    if arr.ndim != 1:
        raise ValueError("clusters must be a 1D array.")
    if arr.size == 0:
        raise ValueError("clusters must be non-empty.")
    if not np.isfinite(arr.astype(np.float64)).all():
        raise ValueError("clusters contain NaN or inf.")
    return arr.astype(np.int64)


def normalize_clusters(clusters: Sequence[np.ndarray] | np.ndarray, *, nobs: int) -> ClusterSpec:
    """
    Normalize cluster labels to contiguous integer codes.

    Accepts a 1D array (one-way clustering) or a sequence of 1D arrays.
    """
    if isinstance(clusters, (list, tuple)):
        arrays = [np.asarray(c) for c in clusters]
    else:
        arrays = [np.asarray(clusters)]

    codes: list[IntArray] = []
    n_clusters: list[int] = []
    for arr in arrays:
        c1 = _as_1d_int(arr)
        if c1.shape[0] != nobs:
            raise ValueError(f"clusters must have length n={nobs}.")
        uniq, inv = np.unique(c1, return_inverse=True)
        codes.append(inv.astype(np.int64))
        n_clusters.append(int(uniq.size))

    return ClusterSpec(codes=tuple(codes), n_clusters=tuple(n_clusters), nobs=nobs)


def combine_clusters(spec: ClusterSpec) -> IntArray:
    """
    Combine multiway clusters into a single interaction cluster code.
    """
    if not spec.codes:
        raise ValueError("ClusterSpec has no codes.")
    if len(spec.codes) == 1:
        return spec.codes[0]

    stacked = np.column_stack(spec.codes)
    _, inv = np.unique(stacked, axis=0, return_inverse=True)
    return inv.astype(np.int64)
