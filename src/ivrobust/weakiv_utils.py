from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._typing import FloatArray, IntArray
from .covariance import CovSpec, CovType, cov_reduced_form
from .data import IVData
from .linalg.ops import proj as _proj
from .linalg.ops import resid as _resid


@dataclass(frozen=True)
class ReducedFormResult:
    z: FloatArray
    y: FloatArray
    d: FloatArray
    pi_y: FloatArray
    pi_d: FloatArray
    resid_y: FloatArray
    resid_d: FloatArray
    cov: FloatArray
    cov_type: str
    df_resid: int
    nobs: int
    k_instr: int
    warnings: tuple[str, ...]


def partial_out(x: FloatArray, *args: FloatArray) -> tuple[FloatArray, ...]:
    """
    Residualize each array in args on x using least squares.
    """
    if x.size == 0:
        return args
    return tuple(_resid(x, a) for a in args)


def proj(z: FloatArray, *args: FloatArray) -> tuple[FloatArray, ...]:
    """
    Project each array in args onto the column space of z.
    """
    if z.size == 0:
        return tuple(np.zeros_like(a) for a in args)
    return tuple(_proj(z, a) for a in args)


def reduced_form(
    data: IVData,
    *,
    cov_type: CovType = "HC1",
    clusters: IntArray | None = None,
    cov: CovSpec | str | None = None,
    hac_lags: int | None = None,
    kernel: str = "bartlett",
) -> ReducedFormResult:
    """
    Compute reduced-form coefficients and covariance for scalar endogenous regressor.
    """
    if data.p_endog != 1:
        raise NotImplementedError("reduced_form currently supports p_endog=1.")

    y = data.y.reshape(-1, 1)
    d = data.d.reshape(-1, 1)
    z = data.z
    x = data.x

    y_tilde, d_tilde, z_tilde = partial_out(x, y, d, z)

    Z = np.asarray(z_tilde, dtype=np.float64)
    YD = np.hstack([y_tilde, d_tilde])
    coef, *_ = np.linalg.lstsq(Z, YD, rcond=None)
    resid = YD - Z @ coef

    pi_y = coef[:, [0]].astype(np.float64)
    pi_d = coef[:, [1]].astype(np.float64)
    resid_y = resid[:, [0]].astype(np.float64)
    resid_d = resid[:, [1]].astype(np.float64)

    clusters_use = clusters if clusters is not None else data.clusters
    cov_res = cov_reduced_form(
        X=Z,
        resid_y=resid_y,
        resid_d=resid_d,
        cov_type=cov_type,
        clusters=clusters_use,
        cov=cov,
        hac_lags=hac_lags,
        kernel=kernel,
        df_resid_adj=data.nobs - data.k_instr - data.p_exog,
    )

    return ReducedFormResult(
        z=Z,
        y=y_tilde,
        d=d_tilde,
        pi_y=pi_y,
        pi_d=pi_d,
        resid_y=resid_y,
        resid_d=resid_d,
        cov=cov_res.cov,
        cov_type=cov_res.cov_type,
        df_resid=cov_res.df_resid,
        nobs=data.nobs,
        k_instr=data.k_instr,
        warnings=cov_res.warnings,
    )


def md_optimal_pi(
    beta: float,
    *,
    V_inv: np.ndarray,
    k: int,
    pi_y: np.ndarray,
    pi_d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Minimum-distance projection for reduced-form coefficients under beta.
    """
    V11 = V_inv[:k, :k]
    V12 = V_inv[:k, k:]
    V21 = V_inv[k:, :k]
    V22 = V_inv[k:, k:]

    A = (beta**2) * V11 + beta * (V12 + V21) + V22
    B = beta * (V11 @ pi_y + V12 @ pi_d) + (V21 @ pi_y + V22 @ pi_d)

    try:
        pi_hat = np.linalg.solve(A, B)
    except np.linalg.LinAlgError:
        pi_hat = np.linalg.pinv(A) @ B

    r = np.vstack([pi_y - beta * pi_hat, pi_d - pi_hat])
    q = float((r.T @ V_inv @ r).ravel()[0])
    return pi_hat, r, q


def default_beta_bounds(data: IVData) -> tuple[float, float]:
    y_std = float(np.std(data.y))
    d_std = float(np.std(data.d))
    scale = y_std / max(d_std, 1e-12)
    width = max(10.0 * scale, 10.0)
    return (-width, width)
