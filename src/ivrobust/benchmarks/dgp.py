"""Data-generating processes for benchmarks."""

from __future__ import annotations

from typing import Optional

import numpy as np

from ivrobust.data.ivdata import IVData


def weak_iv_dgp(
    *,
    n: int = 500,
    k: int = 5,
    strength: float = 0.1,
    beta: float = 1.0,
    rho: float = 0.5,
    seed: Optional[int] = None,
    n_clusters: Optional[int] = None,
) -> tuple[IVData, float]:
    """Generate a weak-IV dataset with a single endogenous regressor."""

    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, k))

    cov = np.array([[1.0, rho], [rho, 1.0]])
    uv = rng.multivariate_normal([0.0, 0.0], cov, size=n)
    u = uv[:, 0]
    v = uv[:, 1]

    pi = np.full(k, strength / np.sqrt(k))
    x_endog = z @ pi + v
    y = beta * x_endog + u

    x_exog = np.ones((n, 1))

    clusters = None
    if n_clusters is not None:
        if n_clusters <= 1:
            raise ValueError("n_clusters must be at least 2.")
        clusters = np.repeat(np.arange(n_clusters), np.ceil(n / n_clusters))[:n]

    data = IVData.from_arrays(
        y=y,
        X_endog=x_endog,
        X_exog=x_exog,
        Z=z,
        clusters=clusters,
    )
    return data, beta
