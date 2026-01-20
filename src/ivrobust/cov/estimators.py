from __future__ import annotations

from typing import cast

import numpy as np

from .._typing import FloatArray
from ..covariance import compute_moment_cov, cov_ols, cov_reduced_form


def hac_meat(
    *, X: FloatArray, resid1: FloatArray, resid2: FloatArray, lags: int, kernel: str
) -> FloatArray:
    _ = kernel
    X2 = np.asarray(X, dtype=np.float64)
    r1 = np.asarray(resid1, dtype=np.float64).reshape(-1, 1)
    r2 = np.asarray(resid2, dtype=np.float64).reshape(-1, 1)
    Xu1 = X2 * r1
    Xu2 = X2 * r2
    meat = Xu1.T @ Xu2
    for lag in range(1, lags + 1):
        weight = 1.0 - lag / (lags + 1)
        if weight <= 0:
            continue
        gamma = Xu1[lag:].T @ Xu2[:-lag]
        meat += weight * (gamma + gamma.T)
    return cast(FloatArray, meat)


__all__ = [
    "compute_moment_cov",
    "cov_ols",
    "cov_reduced_form",
    "hac_meat",
]
