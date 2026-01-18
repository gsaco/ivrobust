from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._typing import FloatArray, IntArray


@dataclass(frozen=True)
class Shapes:
    n: int
    p_exog: int
    p_endog: int
    k_instr: int


def _as_2d_float(x: np.ndarray, *, name: str) -> FloatArray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 1D or 2D; got shape {arr.shape}.")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or infinite values.")
    return arr


def _as_1d_int(x: np.ndarray, *, name: str) -> IntArray:
    arr = np.asarray(x)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D; got shape {arr.shape}.")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    if not np.isfinite(arr.astype(np.float64)).all():
        raise ValueError(f"{name} contains NaN or infinite values.")
    return arr.astype(np.int64)


def validate_iv_arrays(
    *,
    y: np.ndarray,
    d: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    clusters: np.ndarray | None = None,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, IntArray | None, Shapes]:
    y2 = _as_2d_float(y, name="y")
    d2 = _as_2d_float(d, name="d")
    x2 = _as_2d_float(x, name="x")
    z2 = _as_2d_float(z, name="z")

    n = y2.shape[0]
    if d2.shape[0] != n or x2.shape[0] != n or z2.shape[0] != n:
        raise ValueError(
            "All inputs must have the same number of rows (observations). "
            f"Got y:{y2.shape[0]}, d:{d2.shape[0]}, x:{x2.shape[0]}, z:{z2.shape[0]}."
        )

    if y2.shape[1] != 1:
        raise ValueError(f"y must be a single column (n x 1); got shape {y2.shape}.")

    if d2.shape[1] < 1:
        raise ValueError("d must have at least one endogenous regressor column.")

    if z2.shape[1] < 1:
        raise ValueError("z must have at least one instrument column.")

    if x2.shape[1] < 1:
        raise ValueError("x must have at least one exogenous column (include intercept).")

    clusters1: IntArray | None = None
    if clusters is not None:
        clusters1 = _as_1d_int(clusters, name="clusters")
        if clusters1.shape[0] != n:
            raise ValueError(
                f"clusters must have length n={n}; got length {clusters1.shape[0]}."
            )

    shapes = Shapes(
        n=n,
        p_exog=x2.shape[1],
        p_endog=d2.shape[1],
        k_instr=z2.shape[1],
    )
    return y2, d2, x2, z2, clusters1, shapes
