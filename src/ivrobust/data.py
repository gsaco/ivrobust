from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ._typing import FloatArray, IntArray
from ._validation import Shapes, validate_iv_arrays


@dataclass(frozen=True)
class IVData:
    """
    Container for a linear IV dataset with a fixed matrix layout.

    Parameters
    ----------
    y
        Outcome vector with shape (n, 1).
    d
        Endogenous regressors with shape (n, p_endog).
        The current weak-IV robust inference in ivrobust targets the scalar case
        p_endog=1.
    x
        Exogenous regressors with shape (n, p_exog). Include an intercept column
        if desired.
    z
        Excluded instruments with shape (n, k_instr).
    clusters
        Optional 1D array of length n with cluster identifiers for cluster-robust
        covariance.

    Notes
    -----
    - ivrobust expects the intercept (if used) to be included explicitly in `x`.
    - Input validation rejects NaN/inf and inconsistent shapes.
    """

    y: FloatArray
    d: FloatArray
    x: FloatArray
    z: FloatArray
    clusters: IntArray | None = None
    shapes: Shapes | None = None

    def __post_init__(self) -> None:
        y2, d2, x2, z2, c1, shapes = validate_iv_arrays(
            y=self.y, d=self.d, x=self.x, z=self.z, clusters=self.clusters
        )
        object.__setattr__(self, "y", y2)
        object.__setattr__(self, "d", d2)
        object.__setattr__(self, "x", x2)
        object.__setattr__(self, "z", z2)
        object.__setattr__(self, "clusters", c1)
        object.__setattr__(self, "shapes", shapes)

    @property
    def nobs(self) -> int:
        assert self.shapes is not None
        return self.shapes.n

    @property
    def k_instr(self) -> int:
        assert self.shapes is not None
        return self.shapes.k_instr

    @property
    def p_endog(self) -> int:
        assert self.shapes is not None
        return self.shapes.p_endog

    @property
    def p_exog(self) -> int:
        assert self.shapes is not None
        return self.shapes.p_exog

    def with_clusters(self, clusters: np.ndarray) -> "IVData":
        return IVData(y=self.y, d=self.d, x=self.x, z=self.z, clusters=clusters)

    def as_dict(self) -> dict[str, Any]:
        return {
            "y": self.y,
            "d": self.d,
            "x": self.x,
            "z": self.z,
            "clusters": self.clusters,
        }


def weak_iv_dgp(
    *,
    n: int,
    k: int,
    strength: float,
    beta: float,
    seed: int | None = None,
    rho: float = 0.5,
) -> tuple[IVData, float]:
    """
    Generate a synthetic linear IV dataset with one endogenous regressor.

    Model
    -----
    d = Z pi + v
    y = beta d + u

    where (u, v) are correlated to induce endogeneity.

    Parameters
    ----------
    n
        Number of observations.
    k
        Number of excluded instruments.
    strength
        Controls first-stage strength via pi magnitude (roughly scales the
        concentration).
    beta
        True structural coefficient on d.
    seed
        Random seed for reproducibility.
    rho
        Corr(u, v). Must satisfy |rho| < 1.

    Returns
    -------
    data, beta_true
        IVData with intercept-only exogenous regressors and the true beta.

    Notes
    -----
    This DGP is intended for teaching, testing, and Monte Carlo diagnostics. It
    is not a substitute for application-specific simulation design.
    """
    if n <= 5:
        raise ValueError("n must be > 5.")
    if k < 1:
        raise ValueError("k must be >= 1.")
    if not np.isfinite(strength) or strength <= 0:
        raise ValueError("strength must be a positive finite number.")
    if not np.isfinite(beta):
        raise ValueError("beta must be finite.")
    if not np.isfinite(rho) or abs(rho) >= 1:
        raise ValueError("rho must be finite with |rho| < 1.")

    rng = np.random.default_rng(seed)

    z = rng.standard_normal(size=(n, k))
    x = np.ones((n, 1), dtype=np.float64)

    # Scale pi so that increasing k does not mechanically inflate the first stage.
    pi = (strength / np.sqrt(k)) * np.ones((k, 1), dtype=np.float64)

    e1 = rng.standard_normal(size=(n, 1))
    e2 = rng.standard_normal(size=(n, 1))

    # Correlated structural shocks (u, v)
    u = e1
    v = rho * e1 + np.sqrt(1.0 - rho**2) * e2

    d = z @ pi + v
    y = beta * d + u

    data = IVData(y=y, d=d, x=x, z=z)
    return data, float(beta)
