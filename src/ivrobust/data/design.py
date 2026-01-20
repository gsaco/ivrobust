from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .._typing import FloatArray
from ..linalg.ops import proj, resid


def add_constant(x: FloatArray) -> FloatArray:
    x2 = np.asarray(x, dtype=np.float64)
    if x2.ndim == 1:
        x2 = x2.reshape(-1, 1)
    n = x2.shape[0]
    ones = np.ones((n, 1), dtype=np.float64)
    return np.hstack([ones, x2])


def stack_columns(arrays: Iterable[FloatArray]) -> FloatArray:
    mats = [np.asarray(a, dtype=np.float64) for a in arrays]
    if not mats:
        return np.zeros((0, 0), dtype=np.float64)
    for i, mat in enumerate(mats):
        if mat.ndim == 1:
            mats[i] = mat.reshape(-1, 1)
    return np.hstack(mats)


def partial_out(x: FloatArray, *args: FloatArray) -> tuple[FloatArray, ...]:
    """
    Residualize each array in args on x using QR-based projections.
    """
    if x.size == 0:
        return args
    return tuple(resid(x, a) for a in args)


def project_on(x: FloatArray, *args: FloatArray) -> tuple[FloatArray, ...]:
    """
    Project each array in args onto the column space of x.
    """
    if x.size == 0:
        return tuple(np.zeros_like(a) for a in args)
    return tuple(proj(x, a) for a in args)


def column_names(prefix: str, n: int) -> list[str]:
    return [f"{prefix}{i + 1}" for i in range(n)]
