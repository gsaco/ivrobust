from __future__ import annotations

from typing import cast

import numpy as np
import scipy.linalg

from ..covariance import (
    CovSpec,
    CovType,
    _default_hac_lags,
    _hac_meat,
    _leverage,
    _pinv_sym,
)
from ..data import IVData
from ..data.clusters import normalize_clusters
from .results import IVResults


def _generalized_eigvals(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    try:
        vals = scipy.linalg.eigvalsh(A, B)
    except Exception:
        vals = np.linalg.eigvals(np.linalg.pinv(B) @ A)
    vals = np.real(vals)
    return np.sort(np.clip(vals, 0.0, np.inf))


def _kappa_liml(X: np.ndarray, y: np.ndarray, Z: np.ndarray) -> float:
    Xy = np.hstack([X, y])
    ZTZ_inv = _pinv_sym(Z.T @ Z)
    Xy_proj = Z @ ZTZ_inv @ Z.T @ Xy
    Xy_orth = Xy - Xy_proj
    A = Xy_proj.T @ Xy_proj
    B = Xy_orth.T @ Xy_orth
    eigvals = _generalized_eigvals(A, B)
    ar_min = float(eigvals[0]) if eigvals.size else 0.0
    return 1.0 + ar_min


def _kclass_cov(
    Xk: np.ndarray,
    resid: np.ndarray,
    *,
    cov_type: CovType,
    clusters: np.ndarray | None,
) -> np.ndarray:
    X2 = np.asarray(Xk, dtype=np.float64)
    r = np.asarray(resid, dtype=np.float64).reshape(-1, 1)
    n, p = X2.shape
    if cov_type in ("HC2", "HC3"):
        h = _leverage(X2).reshape(-1, 1)
        scale = np.clip(1.0 - h, 1e-12, None)
        r = r / np.sqrt(scale) if cov_type == "HC2" else r / scale

    if cov_type == "cluster":
        if clusters is None:
            raise ValueError("clusters must be provided when cov_type='cluster'.")
        spec = normalize_clusters(clusters, nobs=n)
        g = spec.codes[0]
        uniq, inv = np.unique(g, return_inverse=True)
        G = int(uniq.size)
        if G < 2:
            raise ValueError("cluster covariance requires at least 2 clusters.")
        meat = np.zeros((p, p), dtype=np.float64)
        for gi in range(G):
            idx = inv == gi
            Xg = X2[idx, :]
            rg = r[idx, :]
            sg = Xg.T @ rg
            meat += sg @ sg.T
        return meat

    if cov_type == "HAC":
        lags = _default_hac_lags(n)
        return _hac_meat(X=X2, resid1=r, resid2=r, lags=lags, kernel="bartlett")

    w = r**2
    return cast(np.ndarray, X2.T @ (X2 * w))


def kclass(
    data: IVData,
    *,
    kappa: float,
    cov: CovSpec | str | None = None,
    cov_type: CovType = "HC1",
    clusters: np.ndarray | None = None,
) -> IVResults:
    _ = cov
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
    _ = resid_proj

    A_inv = _pinv_sym(X_k.T @ X)
    meat = _kclass_cov(
        X_k,
        resid,
        cov_type=cov_type,
        clusters=clusters if clusters is not None else data.clusters,
    )
    V = A_inv @ meat @ A_inv
    if cov_type == "HC1":
        V *= n / df_resid
    if cov_type == "cluster":
        if clusters is None and data.clusters is None:
            raise ValueError("clusters must be provided when cov_type='cluster'.")
        g = clusters if clusters is not None else data.clusters
        assert g is not None
        uniq = np.unique(g)
        G = int(uniq.size)
        V *= (G / (G - 1)) * ((n - 1) / df_resid)

    se = np.sqrt(np.clip(np.diag(V), 0.0, np.inf)).reshape(-1, 1)
    return IVResults(
        params=beta,
        stderr=se,
        vcov=V,
        cov_type=str(cov_type),
        cov_config={},
        nobs=n,
        df_resid=df_resid,
        k_endog=data.p_endog,
        k_instr=data.k_instr,
        k_exog=data.p_exog,
        method=f"kclass({kappa:.4f})",
        data=data,
        diagnostics={},
    )


def liml(
    data: IVData,
    *,
    cov: CovSpec | str | None = None,
    cov_type: CovType = "HC1",
    clusters: np.ndarray | None = None,
) -> IVResults:
    X = np.hstack([data.x, data.d])
    Z = np.hstack([data.x, data.z])
    kappa = _kappa_liml(X, data.y, Z)
    res = kclass(data, kappa=kappa, cov=cov, cov_type=cov_type, clusters=clusters)
    return IVResults(
        params=res.params,
        stderr=res.stderr,
        vcov=res.vcov,
        cov_type=res.cov_type,
        cov_config=res.cov_config,
        nobs=res.nobs,
        df_resid=res.df_resid,
        k_endog=data.p_endog,
        k_instr=data.k_instr,
        k_exog=data.p_exog,
        method="liml",
        data=data,
        diagnostics=res.diagnostics,
    )


def fuller(
    data: IVData,
    *,
    alpha: float = 1.0,
    cov: CovSpec | str | None = None,
    cov_type: CovType = "HC1",
    clusters: np.ndarray | None = None,
) -> IVResults:
    X = np.hstack([data.x, data.d])
    Z = np.hstack([data.x, data.z])
    kappa_liml = _kappa_liml(X, data.y, Z)
    dof = data.nobs - Z.shape[1]
    if dof <= 0:
        raise ValueError("Need n > number of instruments for Fuller estimator.")
    kappa = kappa_liml - alpha / dof
    res = kclass(data, kappa=kappa, cov=cov, cov_type=cov_type, clusters=clusters)
    return IVResults(
        params=res.params,
        stderr=res.stderr,
        vcov=res.vcov,
        cov_type=res.cov_type,
        cov_config=res.cov_config,
        nobs=res.nobs,
        df_resid=res.df_resid,
        k_endog=data.p_endog,
        k_instr=data.k_instr,
        k_exog=data.p_exog,
        method=f"fuller({alpha})",
        data=data,
        diagnostics=res.diagnostics,
    )


__all__ = ["fuller", "kclass", "liml"]
