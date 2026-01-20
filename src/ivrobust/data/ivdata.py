from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .._typing import FloatArray, IntArray
from .._validation import Shapes, validate_iv_arrays
from .clusters import normalize_clusters
from .design import add_constant


@dataclass(frozen=True)
class IVData:
    """
    Canonical container for linear IV data with a fixed matrix layout.

    Array shapes
    -----------
    y : (n, 1)
        Outcome vector (single column).
    d : (n, p_endog)
        Endogenous regressors. Weak-IV robust tests target p_endog=1.
    x : (n, p_exog)
        Exogenous regressors. Include an intercept column if desired.
    z : (n, k_instr)
        Excluded instruments.
    clusters : (n,)
        Optional cluster labels for cluster-robust covariance.

    Validation
    ----------
    - Arrays are coerced to float64 (or int64 for clusters).
    - NaN/inf values are rejected.
    - All inputs must have the same number of rows.
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
        if c1 is not None:
            c1 = normalize_clusters(c1, nobs=shapes.n).codes[0]
        object.__setattr__(self, "y", y2)
        object.__setattr__(self, "d", d2)
        object.__setattr__(self, "x", x2)
        object.__setattr__(self, "z", z2)
        object.__setattr__(self, "clusters", c1)
        object.__setattr__(self, "shapes", shapes)

    @classmethod
    def from_arrays(
        cls,
        *,
        y: FloatArray,
        d: FloatArray,
        z: FloatArray,
        x: FloatArray | None = None,
        add_const: bool = True,
        clusters: IntArray | None = None,
    ) -> IVData:
        """
        Construct IVData from raw arrays with explicit keyword-only inputs.
        """
        y2 = np.asarray(y, dtype=np.float64)
        d2 = np.asarray(d, dtype=np.float64)
        z2 = np.asarray(z, dtype=np.float64)

        if y2.ndim == 1:
            y2 = y2.reshape(-1, 1)
        if d2.ndim == 1:
            d2 = d2.reshape(-1, 1)
        if z2.ndim == 1:
            z2 = z2.reshape(-1, 1)

        if x is None:
            if not add_const:
                raise ValueError("x is None and add_const=False.")
            x2 = np.ones((y2.shape[0], 1), dtype=np.float64)
        else:
            x2 = np.asarray(x, dtype=np.float64)
            if x2.ndim == 1:
                x2 = x2.reshape(-1, 1)
            if add_const:
                x2 = add_constant(x2)

        return cls(y=y2, d=d2, x=x2, z=z2, clusters=clusters)

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

    def with_clusters(self, clusters: np.ndarray) -> IVData:
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

    pi = (strength / np.sqrt(k)) * np.ones((k, 1), dtype=np.float64)

    e1 = rng.standard_normal(size=(n, 1))
    e2 = rng.standard_normal(size=(n, 1))

    u = e1
    v = rho * e1 + np.sqrt(1.0 - rho**2) * e2

    d = z @ pi + v
    y = beta * d + u

    data = IVData(y=y, d=d, x=x, z=z)
    return data, float(beta)
