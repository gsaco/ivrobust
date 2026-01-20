from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import f

from .covariance import CovType
from .data import IVData
from .weakiv_utils import reduced_form


@dataclass(frozen=True)
class FirstStageDiagnostics:
    """
    First-stage diagnostics for a single endogenous regressor.
    """

    f_statistic: float
    pvalue: float
    df_num: int
    df_denom: int
    partial_r2: float
    k_instr: int
    nobs: int


@dataclass(frozen=True)
class EffectiveFResult:
    """
    Effective F statistic for weak-instrument diagnostics (single endogenous regressor).
    """

    statistic: float
    df_num: int
    df_denom: int
    cov_type: str
    nobs: int
    k_instr: int
    warnings: tuple[str, ...] = ()


def first_stage_diagnostics(data: IVData) -> FirstStageDiagnostics:
    """
    Compute classical first-stage F-statistic and partial R^2.

    Parameters
    ----------
    data
        IVData.

    Returns
    -------
    FirstStageDiagnostics

    Notes
    -----
    - This is the classical homoskedastic first-stage F statistic.
    - For weak-instrument testing robust to heteroskedasticity/clustered data,
      see Montiel Olea and Pflueger (2013) effective F (planned; see roadmap).
      DOI: 10.1080/00401706.2013.806694.
    """
    if data.p_endog != 1:
        raise NotImplementedError("Diagnostics currently support p_endog=1.")

    d = data.d  # n x 1
    Xr = data.x  # restricted: exog only
    Xf = np.hstack([data.x, data.z])  # full: exog + instruments

    n = data.nobs
    k = data.k_instr
    p_full = Xf.shape[1]

    # OLS fits
    br, *_ = np.linalg.lstsq(Xr, d, rcond=None)
    bf, *_ = np.linalg.lstsq(Xf, d, rcond=None)

    resid_r = d - Xr @ br
    resid_f = d - Xf @ bf

    rss_r = float((resid_r.T @ resid_r).item())
    rss_f = float((resid_f.T @ resid_f).item())

    df_num = k
    df_denom = n - p_full
    if df_denom <= 0:
        raise ValueError("Need n > number of first-stage regressors.")

    f_stat = ((rss_r - rss_f) / df_num) / (rss_f / df_denom)
    pval = float(f.sf(f_stat, df_num, df_denom))

    partial_r2 = (rss_r - rss_f) / max(rss_r, 1e-30)

    return FirstStageDiagnostics(
        f_statistic=float(f_stat),
        pvalue=pval,
        df_num=df_num,
        df_denom=df_denom,
        partial_r2=float(partial_r2),
        k_instr=k,
        nobs=n,
    )


def effective_f(
    data: IVData,
    *,
    cov_type: CovType = "HC1",
    clusters: np.ndarray | None = None,
) -> EffectiveFResult:
    """
    Compute the effective F statistic (Montiel Olea & Pflueger) for p_endog=1.
    """
    if data.p_endog != 1:
        raise NotImplementedError("effective_f currently supports p_endog=1.")

    rf = reduced_form(data, cov_type=cov_type, clusters=clusters)
    k = rf.k_instr
    V_dd = rf.cov[k:, k:]
    V_dd_inv = np.linalg.pinv(V_dd)
    stat = float((rf.pi_d.T @ V_dd_inv @ rf.pi_d).ravel()[0] / k)

    df_denom = data.nobs - data.k_instr - data.p_exog
    if df_denom <= 0:
        df_denom = data.nobs - data.k_instr

    return EffectiveFResult(
        statistic=stat,
        df_num=k,
        df_denom=df_denom,
        cov_type=str(cov_type),
        nobs=data.nobs,
        k_instr=k,
        warnings=rf.warnings,
    )
