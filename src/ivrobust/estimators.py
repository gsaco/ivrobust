from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg

from ._typing import FloatArray
from .covariance import CovType, _pinv_sym
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
    method: str = "2sls"

    @property
    def beta(self) -> float:
        # By convention: params = [x params..., d param]
        return float(self.params[-1, 0])


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
    y = data.y  # n x 1
    X = np.hstack([data.x, data.d])  # n x (p_exog + p_endog)
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


def _generalized_eigvals(A: FloatArray, B: FloatArray) -> FloatArray:
    try:
        vals = scipy.linalg.eigvalsh(A, B)
    except Exception:
        vals = np.linalg.eigvals(np.linalg.pinv(B) @ A)
    vals = np.real(vals)
    return np.sort(np.clip(vals, 0.0, np.inf))


def _kappa_liml(X: FloatArray, y: FloatArray, Z: FloatArray) -> float:
    Xy = np.hstack([X, y])
    ZTZ_inv = _pinv_sym(Z.T @ Z)
    Xy_proj = Z @ ZTZ_inv @ Z.T @ Xy
    Xy_orth = Xy - Xy_proj
    A = Xy_proj.T @ Xy_proj
    B = Xy_orth.T @ Xy_orth
    eigvals = _generalized_eigvals(A, B)
    ar_min = float(eigvals[0]) if eigvals.size else 0.0
    return 1.0 + ar_min


def _kclass(
    data: IVData,
    *,
    kappa: float,
    cov_type: CovType = "HC1",
) -> TSLSResult:
    y = data.y
    X = np.hstack([data.x, data.d])
    Z = np.hstack([data.x, data.z])

    n, p = X.shape
    df_resid = n - p
    if df_resid <= 0:
        raise ValueError("Need n > number of regressors for k-class estimation.")

    ZTZ_inv = _pinv_sym(Z.T @ Z)
    X_proj = Z @ ZTZ_inv @ Z.T @ X
    y_proj = Z @ ZTZ_inv @ Z.T @ y

    X_k = (1.0 - kappa) * X + kappa * X_proj
    beta = np.linalg.solve(X_k.T @ X, X_k.T @ y)

    resid = y - X @ beta
    resid_proj = y_proj - X_proj @ beta

    if cov_type in ("HC0", "HC1", "HC2", "HC3"):
        u2 = resid**2
        meat = X_k.T @ (X_k * u2)
        V = _pinv_sym(X_k.T @ X) @ meat @ _pinv_sym(X_k.T @ X)
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
        meat = np.zeros((p, p), dtype=np.float64)
        for gi in range(G):
            idx = inv == gi
            Xg = X_k[idx, :]
            eg = resid[idx, :]
            sg = Xg.T @ eg
            meat += sg @ sg.T
        V = _pinv_sym(X_k.T @ X) @ meat @ _pinv_sym(X_k.T @ X)
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
        method=f"kclass({kappa:.4f})",
    )


def liml(
    data: IVData,
    *,
    cov_type: CovType = "HC1",
) -> TSLSResult:
    """
    Limited-information maximum likelihood (LIML) estimator.
    """
    X = np.hstack([data.x, data.d])
    Z = np.hstack([data.x, data.z])
    kappa = _kappa_liml(X, data.y, Z)
    res = _kclass(data, kappa=kappa, cov_type=cov_type)
    return TSLSResult(
        params=res.params,
        stderr=res.stderr,
        vcov=res.vcov,
        cov_type=res.cov_type,
        nobs=res.nobs,
        df_resid=res.df_resid,
        method="liml",
    )


def fuller(
    data: IVData,
    *,
    alpha: float = 1.0,
    cov_type: CovType = "HC1",
) -> TSLSResult:
    """
    Fuller estimator with tuning parameter alpha (default 1.0).
    """
    X = np.hstack([data.x, data.d])
    Z = np.hstack([data.x, data.z])
    kappa_liml = _kappa_liml(X, data.y, Z)
    dof = data.nobs - Z.shape[1]
    if dof <= 0:
        raise ValueError("Need n > number of instruments for Fuller estimator.")
    kappa = kappa_liml - alpha / dof
    res = _kclass(data, kappa=kappa, cov_type=cov_type)
    return TSLSResult(
        params=res.params,
        stderr=res.stderr,
        vcov=res.vcov,
        cov_type=res.cov_type,
        nobs=res.nobs,
        df_resid=res.df_resid,
        method=f"fuller({alpha})",
    )
