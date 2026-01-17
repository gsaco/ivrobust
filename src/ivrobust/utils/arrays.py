"""Array validation helpers."""

from __future__ import annotations

from typing import Optional, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray


def as_1d_array(x: ArrayLike, name: str) -> NDArray[np.float64]:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr[:, 0]
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape {arr.shape}.")
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty.")
    return cast(NDArray[np.float64], arr)


def as_2d_array(
    x: ArrayLike, name: str, n_rows: Optional[int] = None
) -> NDArray[np.float64]:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {arr.shape}.")
    if arr.shape[0] == 0:
        raise ValueError(f"{name} must have at least one row.")
    if n_rows is not None and arr.shape[0] != n_rows:
        raise ValueError(f"{name} must have {n_rows} rows, got {arr.shape[0]}.")
    return cast(NDArray[np.float64], arr)


def check_finite(arr: NDArray[np.float64], name: str) -> None:
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinite values.")
