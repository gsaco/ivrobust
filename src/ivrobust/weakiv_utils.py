from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._typing import FloatArray
from .covariance import CovType, cov_reduced_form
from .data import IVData


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
    X = np.asarray(x, dtype=np.float64)
    stacked = np.hstack([np.asarray(a, dtype=np.float64) for a in args])
    coef, *_ = np.linalg.lstsq(X, stacked, rcond=None)
    fitted = X @ coef
    resid = stacked - fitted

    outs: list[FloatArray] = []
    col = 0
    for arr in args:
        arr2 = np.asarray(arr, dtype=np.float64)
        ncols = arr2.shape[1] if arr2.ndim == 2 else 1
        outs.append(resid[:, col : col + ncols].reshape(arr2.shape))
        col += ncols
    return tuple(outs)


def proj(z: FloatArray, *args: FloatArray) -> tuple[FloatArray, ...]:
    """
    Project each array in args onto the column space of z.
    """
    Z = np.asarray(z, dtype=np.float64)
    if Z.size == 0:
        return tuple(np.zeros_like(a) for a in args)
    stacked = np.hstack([np.asarray(a, dtype=np.float64) for a in args])
    coef, *_ = np.linalg.lstsq(Z, stacked, rcond=None)
    fitted = Z @ coef

    outs: list[FloatArray] = []
    col = 0
    for arr in args:
        arr2 = np.asarray(arr, dtype=np.float64)
        ncols = arr2.shape[1] if arr2.ndim == 2 else 1
        outs.append(fitted[:, col : col + ncols].reshape(arr2.shape))
        col += ncols
    return tuple(outs)


def reduced_form(
    data: IVData,
    *,
    cov_type: CovType,
    clusters: IntArray | None = None,
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

    pi_y = coef[:, [0]]
    pi_d = coef[:, [1]]
    resid_y = resid[:, [0]]
    resid_d = resid[:, [1]]

    clusters_use = clusters if clusters is not None else data.clusters
    cov_res = cov_reduced_form(
        X=Z,
        resid_y=resid_y,
        resid_d=resid_d,
        cov_type=cov_type,
        clusters=clusters_use,
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
