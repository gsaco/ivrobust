from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.stats import chi2, f

from ..covariance import CovSpec, CovType
from ..data import IVData
from ..linalg.ops import sym_solve
from ..utils.warnings import WarningCategory, warn
from ..weakiv_utils import reduced_form


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


@dataclass(frozen=True)
class WeakIdDiagnostics:
    effective_f: float
    first_stage_f: float
    partial_r2: float
    cragg_donald_f: float
    kp_rk_stat: float
    kp_rk_pvalue: float
    nobs: int
    k_instr: int
    p_endog: int
    notes: dict[str, Any] = field(default_factory=dict)


def first_stage_diagnostics(data: IVData) -> FirstStageDiagnostics:
    """
    Compute classical first-stage F-statistic and partial R^2.
    """
    if data.p_endog != 1:
        raise NotImplementedError("Diagnostics currently support p_endog=1.")

    d = data.d
    Xr = data.x
    Xf = np.hstack([data.x, data.z])

    n = data.nobs
    k = data.k_instr
    p_full = Xf.shape[1]

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

    partial = (rss_r - rss_f) / max(rss_r, 1e-30)

    return FirstStageDiagnostics(
        f_statistic=float(f_stat),
        pvalue=pval,
        df_num=df_num,
        df_denom=df_denom,
        partial_r2=float(partial),
        k_instr=k,
        nobs=n,
    )


def partial_r2(data: IVData) -> float:
    return first_stage_diagnostics(data).partial_r2


def effective_f(
    data: IVData,
    *,
    cov: CovSpec | str | None = None,
    cov_type: CovType = "HC1",
    clusters: np.ndarray | None = None,
) -> EffectiveFResult:
    """
    Compute the effective F statistic (Montiel Olea & Pflueger) for p_endog=1.
    """
    if data.p_endog != 1:
        raise NotImplementedError("effective_f currently supports p_endog=1.")

    rf = reduced_form(data, cov_type=cov_type, cov=cov, clusters=clusters)
    k = rf.k_instr
    V_dd = rf.cov[k:, k:]
    stat = float((rf.pi_d.T @ sym_solve(V_dd, rf.pi_d)).ravel()[0] / k)

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


def cragg_donald_f(data: IVData) -> float:
    """
    Cragg-Donald F statistic (scalar endogenous regressor fallback).
    """
    if data.p_endog != 1:
        raise NotImplementedError("cragg_donald_f currently supports p_endog=1.")
    return first_stage_diagnostics(data).f_statistic


def kp_rk_stat(
    data: IVData,
    *,
    cov: CovSpec | str | None = None,
    cov_type: CovType = "HC1",
    clusters: np.ndarray | None = None,
) -> tuple[float, float, int]:
    rf = reduced_form(data, cov_type=cov_type, cov=cov, clusters=clusters)
    k = rf.k_instr
    V_dd = rf.cov[k:, k:]
    stat = float((rf.pi_d.T @ sym_solve(V_dd, rf.pi_d)).ravel()[0])
    pval = float(chi2.sf(stat, df=k))
    return stat, pval, k


def stock_yogo_critical_values(
    k_endog: int, k_instr: int, *, size_distortion: float = 0.10
) -> float:
    """
    Stock-Yogo critical values for maximal size distortion (partial table).
    """
    if k_endog != 1:
        raise NotImplementedError("Stock-Yogo table implemented for k_endog=1 only.")

    if size_distortion != 0.10:
        raise NotImplementedError("Only size_distortion=0.10 is available.")

    table = {
        1: 16.38,
        2: 8.96,
        3: 7.25,
        4: 6.16,
        5: 5.47,
        6: 4.99,
        7: 4.64,
        8: 4.39,
        9: 4.19,
        10: 4.03,
    }
    if k_instr not in table:
        raise NotImplementedError("Stock-Yogo table supports k_instr in 1..10.")
    return table[k_instr]


def weak_id_diagnostics(
    data: IVData,
    *,
    cov: CovSpec | str | None = None,
    cov_type: CovType = "HC1",
    clusters: np.ndarray | None = None,
) -> WeakIdDiagnostics:
    diag = first_stage_diagnostics(data)
    eff = effective_f(data, cov=cov, cov_type=cov_type, clusters=clusters)
    rk_stat, rk_pval, _ = kp_rk_stat(
        data, cov=cov, cov_type=cov_type, clusters=clusters
    )

    if eff.statistic < 10.0:
        warn(
            WarningCategory.WEAK_ID,
            f"effective F below 10 (F_eff={eff.statistic:.2f})",
        )
    if data.k_instr / max(data.nobs, 1) > 0.2:
        warn(
            WarningCategory.MANY_INSTRUMENTS,
            "many instruments relative to sample size (k/n > 0.2)",
        )

    notes: dict[str, Any] = {"stock_yogo_10pct": None}
    try:
        notes["stock_yogo_10pct"] = stock_yogo_critical_values(
            data.p_endog, data.k_instr, size_distortion=0.10
        )
    except Exception:
        notes["stock_yogo_10pct"] = None

    return WeakIdDiagnostics(
        effective_f=eff.statistic,
        first_stage_f=diag.f_statistic,
        partial_r2=diag.partial_r2,
        cragg_donald_f=cragg_donald_f(data),
        kp_rk_stat=rk_stat,
        kp_rk_pvalue=rk_pval,
        nobs=data.nobs,
        k_instr=data.k_instr,
        p_endog=data.p_endog,
        notes=notes,
    )
