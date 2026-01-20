from __future__ import annotations

from typing import Any

import numpy as np

from ..covariance import CovSpec, CovType, _pinv_sym, compute_moment_cov
from ..data import IVData
from .results import IVResults, TSLSResult


def tsls(
    data: IVData,
    *,
    cov: CovSpec | str | None = None,
    cov_type: CovType = "HC1",
    clusters: np.ndarray | None = None,
) -> TSLSResult:
    """
    Two-stage least squares estimation for a single endogenous regressor.
    """
    y = data.y
    X = np.hstack([data.x, data.d])
    Z = np.hstack([data.x, data.z])

    if cov_type == "cluster" and clusters is None and data.clusters is None:
        raise ValueError("Cluster covariance requested but data.clusters is None.")

    n, p = X.shape
    df_resid = n - p
    if df_resid <= 0:
        raise ValueError("Need n > number of regressors for 2SLS.")

    ZTZ_inv = _pinv_sym(Z.T @ Z)

    XTPZX = X.T @ Z @ ZTZ_inv @ Z.T @ X
    XTPZy = X.T @ Z @ ZTZ_inv @ Z.T @ y
    beta = np.linalg.solve(XTPZX, XTPZy)

    resid = y - X @ beta

    A_inv = _pinv_sym(XTPZX)
    cov_z = compute_moment_cov(
        X=Z,
        resid=resid,
        cov_type=cov_type,
        clusters=clusters if clusters is not None else data.clusters,
        cov=cov,
    )
    B = X.T @ Z @ cov_z.cov @ Z.T @ X
    V = A_inv @ B @ A_inv

    se = np.sqrt(np.clip(np.diag(V), 0.0, np.inf)).reshape(-1, 1)
    cov_config: dict[str, Any] = {}
    return IVResults(
        params=beta,
        stderr=se,
        vcov=V,
        cov_type=str(cov_type),
        cov_config=cov_config,
        nobs=n,
        df_resid=df_resid,
        k_endog=data.p_endog,
        k_instr=data.k_instr,
        k_exog=data.p_exog,
        method="2sls",
        data=data,
        diagnostics={},
    )


__all__ = ["TSLSResult", "tsls"]
