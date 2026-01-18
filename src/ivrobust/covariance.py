from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from ._typing import FloatArray, IntArray


CovType = Literal["unadjusted", "HC0", "HC1", "cluster"]


@dataclass(frozen=True)
class CovarianceResult:
    cov: FloatArray
    cov_type: str
    df_resid: int
    nobs: int
    n_clusters: int | None = None


def _pinv_sym(a: FloatArray) -> FloatArray:
    # Symmetric pseudo-inverse for numerical stability.
    vals, vecs = np.linalg.eigh(a)
    tol = np.max(vals) * 1e-12 if vals.size else 0.0
    inv_vals = np.where(vals > tol, 1.0 / vals, 0.0)
    return (vecs * inv_vals) @ vecs.T


def cov_ols(
    *,
    X: FloatArray,
    resid: FloatArray,
    cov_type: CovType,
    clusters: IntArray | None = None,
) -> CovarianceResult:
    """
    Sandwich covariance for OLS coefficients.

    Parameters
    ----------
    X
        Design matrix (n x p).
    resid
        Residual vector (n x 1) or (n,).
    cov_type
        One of {"unadjusted", "HC0", "HC1", "cluster"}.
    clusters
        Required when cov_type="cluster". Integer cluster labels length n.

    Returns
    -------
    CovarianceResult
        Contains covariance matrix for OLS coefficients.
    """
    X2 = np.asarray(X, dtype=np.float64)
    r = np.asarray(resid, dtype=np.float64).reshape(-1, 1)

    n, p = X2.shape
    if r.shape[0] != n:
        raise ValueError("resid must have the same number of rows as X.")
    if n <= p:
        raise ValueError("Need n > p for covariance estimation.")

    bread = _pinv_sym(X2.T @ X2)
    df_resid = n - p

    if cov_type == "unadjusted":
        sigma2 = float(((r.T @ r) / df_resid).item())
        cov = sigma2 * bread
        return CovarianceResult(cov=cov, cov_type="unadjusted", df_resid=df_resid, nobs=n)

    if cov_type in ("HC0", "HC1"):
        w = r**2
        meat = X2.T @ (X2 * w)
        cov = bread @ meat @ bread
        if cov_type == "HC1":
            cov *= n / df_resid
        return CovarianceResult(cov=cov, cov_type=cov_type, df_resid=df_resid, nobs=n)

    if cov_type == "cluster":
        if clusters is None:
            raise ValueError("clusters must be provided when cov_type='cluster'.")
        g = np.asarray(clusters)
        if g.ndim != 1 or g.shape[0] != n:
            raise ValueError("clusters must be a 1D array of length n.")

        uniq, inv = np.unique(g, return_inverse=True)
        G = int(uniq.size)
        if G < 2:
            raise ValueError("cluster covariance requires at least 2 clusters.")

        meat = np.zeros((p, p), dtype=np.float64)
        for gi in range(G):
            idx = inv == gi
            Xg = X2[idx, :]
            rg = r[idx, :]
            sg = Xg.T @ rg  # p x 1
            meat += sg @ sg.T

        cov = bread @ meat @ bread

        # Finite-sample correction (common in practice)
        cov *= (G / (G - 1)) * ((n - 1) / df_resid)

        return CovarianceResult(
            cov=cov,
            cov_type="cluster",
            df_resid=df_resid,
            nobs=n,
            n_clusters=G,
        )

    raise ValueError(f"Unknown cov_type: {cov_type}")
