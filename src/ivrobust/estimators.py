from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._typing import FloatArray
from .covariance import CovType
from .data import IVData


@dataclass(frozen=True)
class TSLSResult:
    """
    Two-stage least squares (2SLS) result.

    Notes
    -----
    2SLS standard errors are not weak-instrument robust. Use AR-based inference
    for weak-IV robust tests and confidence sets.
    """

    params: FloatArray
    stderr: FloatArray
    vcov: FloatArray
    cov_type: str
    nobs: int
    df_resid: int

    @property
    def beta(self) -> float:
        # By convention: params = [x params..., d param]
        return float(self.params[-1, 0])


def _pinv_sym(a: FloatArray) -> FloatArray:
    vals, vecs = np.linalg.eigh(a)
    tol = np.max(vals) * 1e-12 if vals.size else 0.0
    inv_vals = np.where(vals > tol, 1.0 / vals, 0.0)
    return (vecs * inv_vals) @ vecs.T


def tsls(
    data: IVData,
    *,
    cov_type: CovType = "HC1",
) -> TSLSResult:
    """
    Two-stage least squares estimation for a single endogenous regressor.

    Parameters
    ----------
    data
        IVData.
    cov_type
        "HC0", "HC1", or "cluster". Cluster uses one-way cluster robust covariance.

    Returns
    -------
    TSLSResult
        Parameter estimates and covariance.

    Notes
    -----
    This function is provided for workflow support. For weak-IV robust inference
    on the coefficient of the endogenous regressor, use `ar_test` or
    `ar_confidence_set`.
    """
    if data.p_endog != 1:
        raise NotImplementedError("tsls currently supports p_endog=1.")

    y = data.y  # n x 1
    X = np.hstack([data.x, data.d])  # n x (p_exog + 1)
    Z = np.hstack([data.x, data.z])  # n x (p_exog + k)

    n, p = X.shape
    df_resid = n - p
    if df_resid <= 0:
        raise ValueError("Need n > number of regressors for 2SLS.")

    ZTZ_inv = _pinv_sym(Z.T @ Z)

    # 2SLS point estimate: (X'Pz X)^(-1) X'Pz y
    XTPZX = X.T @ Z @ ZTZ_inv @ Z.T @ X
    XTPZy = X.T @ Z @ ZTZ_inv @ Z.T @ y
    beta = np.linalg.solve(XTPZX, XTPZy)

    resid = y - X @ beta

    # Robust covariance for 2SLS (sandwich):
    # V = A^{-1} B A^{-1}, where
    # A = X' Z (Z'Z)^{-1} Z' X
    # B = X' Z (Z'Z)^{-1} (sum z_i z_i' e_i^2) (Z'Z)^{-1} Z' X
    A_inv = _pinv_sym(XTPZX)

    if cov_type in ("HC0", "HC1"):
        e2 = resid**2  # n x 1
        # sum z_i z_i' e_i^2 is Z' diag(e^2) Z
        S = Z.T @ (Z * e2)
        B = X.T @ Z @ ZTZ_inv @ S @ ZTZ_inv @ Z.T @ X
        V = A_inv @ B @ A_inv
        if cov_type == "HC1":
            V *= n / df_resid

    elif cov_type == "cluster":
        if data.clusters is None:
            raise ValueError("Cluster covariance requested but data.clusters is None.")
        g = np.asarray(data.clusters)
        uniq, inv = np.unique(g, return_inverse=True)
        G = int(uniq.size)
        if G < 2:
            raise ValueError("cluster covariance requires at least 2 clusters.")

        # Cluster meat: sum_g (Z_g' e_g)(Z_g' e_g)'
        S = np.zeros((Z.shape[1], Z.shape[1]), dtype=np.float64)
        for gi in range(G):
            idx = inv == gi
            Zg = Z[idx, :]
            eg = resid[idx, :]
            sg = Zg.T @ eg
            S += sg @ sg.T

        B = X.T @ Z @ ZTZ_inv @ S @ ZTZ_inv @ Z.T @ X
        V = A_inv @ B @ A_inv
        V *= (G / (G - 1)) * ((n - 1) / df_resid)

    else:
        raise ValueError(f"Unknown cov_type: {cov_type}")

    se = np.sqrt(np.clip(np.diag(V), 0.0, np.inf)).reshape(-1, 1)

    return TSLSResult(
        params=beta,
        stderr=se,
        vcov=V,
        cov_type=str(cov_type),
        nobs=n,
        df_resid=df_resid,
    )
