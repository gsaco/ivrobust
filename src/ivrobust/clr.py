from __future__ import annotations

from typing import Sequence

import numpy as np
import scipy.integrate
import scipy.special
from scipy.stats import chi2

from .covariance import CovType, _pinv_sym
from .data import IVData
from .intervals import invert_pvalue_grid
from .results import ConfidenceSetResult, TestResult
from .weakiv_utils import default_beta_bounds, md_optimal_pi, proj, reduced_form


def _clr_lambda(beta: float, *, rf, p_exog: int) -> float:
    resid = rf.y - beta * rf.d
    resid_proj = proj(rf.z, resid)[0]
    resid_orth = resid - resid_proj

    d_proj = proj(rf.z, rf.d)[0]
    sigma_hat = float((resid_orth.T @ resid_orth).ravel()[0])
    if sigma_hat <= 0 or not np.isfinite(sigma_hat):
        return 0.0

    Sigma = float((resid_orth.T @ rf.d).ravel()[0]) / sigma_hat
    d_tilde = rf.d - resid * Sigma
    d_tilde_proj = d_proj - resid_proj * Sigma
    d_tilde_orth = d_tilde - d_tilde_proj

    denom = float((d_tilde_orth.T @ d_tilde_orth).ravel()[0])
    if denom <= 0 or not np.isfinite(denom):
        return 0.0

    dof = rf.nobs - rf.k_instr - p_exog
    if dof <= 0:
        return 0.0
    numer = float((d_tilde_proj.T @ d_tilde_proj).ravel()[0])
    return max(0.0, dof * numer / denom)


def _clr_pvalue(
    *,
    stat: float,
    k: int,
    lambda1: float,
    tol: float = 1e-6,
) -> float:
    if stat <= 0:
        return 1.0
    if k <= 1 or lambda1 <= 0:
        return float(chi2.sf(stat, df=k))

    p = 1
    q = k
    alpha = (q - p) / 2.0
    beta = p / 2.0
    a = lambda1 / (stat + lambda1)
    if a <= 0:
        return float(chi2.sf(stat, df=k))

    k_half = q / 2.0
    z_over_2 = stat / 2.0
    const = np.power(a, -alpha - beta + 1) / scipy.special.beta(alpha, beta)

    def integrand(y: float) -> float:
        return const * scipy.special.gammainc(k_half, z_over_2 / y)

    res = scipy.integrate.quad(
        integrand,
        1 - a,
        1,
        weight="alg",
        wvar=(beta - 1, alpha - 1),
        epsabs=tol,
    )
    return float(1 - res[0])


def _md_q_min(
    *,
    V_inv: np.ndarray,
    k: int,
    pi_y: np.ndarray,
    pi_d: np.ndarray,
    bounds: tuple[float, float],
) -> tuple[float, float]:
    import scipy.optimize

    def obj(b: float) -> float:
        _, _, q = md_optimal_pi(b, V_inv=V_inv, k=k, pi_y=pi_y, pi_d=pi_d)
        return q

    res = scipy.optimize.minimize_scalar(
        obj, bounds=bounds, method="bounded", options={"xatol": 1e-6}
    )
    if not res.success:
        return float(np.mean(bounds)), float(obj(float(np.mean(bounds))))
    return float(res.x), float(res.fun)


def clr_test(
    data: IVData,
    beta0: float | Sequence[float],
    *,
    cov_type: CovType = "HC1",
    clusters: np.ndarray | None = None,
    tol: float = 1e-6,
) -> TestResult:
    """
    Conditional likelihood ratio (CLR) test for H0: beta = beta0 (scalar).
    """
    if data.p_endog != 1:
        raise NotImplementedError(
            "clr_test currently supports a single endogenous regressor (p_endog=1)."
        )

    b0 = float(np.asarray(beta0, dtype=np.float64).ravel()[0])
    rf = reduced_form(data, cov_type=cov_type, clusters=clusters)

    k = rf.k_instr
    V_inv = _pinv_sym(rf.cov)
    pi_hat, r, q_beta = md_optimal_pi(
        b0, V_inv=V_inv, k=k, pi_y=rf.pi_y, pi_d=rf.pi_d
    )

    bounds = default_beta_bounds(data)
    _, q_min = _md_q_min(
        V_inv=V_inv, k=k, pi_y=rf.pi_y, pi_d=rf.pi_d, bounds=bounds
    )
    stat = max(0.0, q_beta - q_min)

    lambda1 = _clr_lambda(b0, rf=rf, p_exog=data.p_exog)
    pval = _clr_pvalue(stat=stat, k=k, lambda1=lambda1, tol=tol)

    warnings = list(rf.warnings)
    if data.p_exog + k >= data.nobs:
        warnings.append("degrees of freedom nonpositive; CLR may be unreliable")

    return TestResult(
        statistic=float(stat),
        pvalue=float(pval),
        df=k,
        method="CLR",
        cov_type=str(cov_type),
        warnings=tuple(warnings),
        details={"beta0": b0, "lambda1": float(lambda1)},
    )


def clr_confidence_set(
    data: IVData,
    *,
    alpha: float = 0.05,
    cov_type: CovType = "HC1",
    clusters: np.ndarray | None = None,
    grid: np.ndarray | None = None,
    beta_bounds: tuple[float, float] | None = None,
    n_grid: int = 2001,
    refine: bool = True,
    refine_tol: float = 1e-6,
    max_refine_iter: int = 80,
    tol: float = 1e-6,
) -> ConfidenceSetResult:
    """
    Invert the CLR test to obtain a (possibly disjoint) confidence set for beta.
    """
    if not (0 < alpha < 1):
        raise ValueError("alpha must be in (0, 1).")
    if grid is None and n_grid < 301:
        raise ValueError("n_grid must be at least 301 for stable inversion.")
    if data.p_endog != 1:
        raise NotImplementedError(
            "clr_confidence_set currently supports a single endogenous regressor (p_endog=1)."
        )

    if grid is None:
        if beta_bounds is None:
            beta_bounds = default_beta_bounds(data)

        lo, hi = map(float, beta_bounds)
        if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
            raise ValueError("beta_bounds must be finite with lo < hi.")
        grid = np.linspace(lo, hi, n_grid, dtype=np.float64)
    else:
        grid = np.asarray(grid, dtype=np.float64).reshape(-1)
        if grid.size < 3:
            raise ValueError("grid must contain at least 3 points.")
        lo, hi = float(grid[0]), float(grid[-1])

    rf = reduced_form(data, cov_type=cov_type, clusters=clusters)
    V_inv = _pinv_sym(rf.cov)
    k = rf.k_instr
    _, q_min = _md_q_min(
        V_inv=V_inv, k=k, pi_y=rf.pi_y, pi_d=rf.pi_d, bounds=(lo, hi)
    )

    pvals = np.empty_like(grid)
    for i, b0 in enumerate(grid):
        _, _, q_beta = md_optimal_pi(
            b0, V_inv=V_inv, k=k, pi_y=rf.pi_y, pi_d=rf.pi_d
        )
        stat = max(0.0, q_beta - q_min)
        lambda1 = _clr_lambda(b0, rf=rf, p_exog=data.p_exog)
        pvals[i] = _clr_pvalue(stat=stat, k=k, lambda1=lambda1, tol=tol)

    cs = invert_pvalue_grid(
        grid=grid,
        pvalues=pvals,
        alpha=alpha,
        refine=refine,
        refine_tol=refine_tol,
        max_refine_iter=max_refine_iter,
        pvalue_func=lambda b: clr_test(
            data, beta0=b, cov_type=cov_type, clusters=clusters, tol=tol
        ).pvalue,
    )

    return ConfidenceSetResult(
        confidence_set=cs,
        alpha=alpha,
        method="CLR",
        grid_info={
            "grid": grid,
            "pvalues": pvals,
            "beta_bounds": (lo, hi),
            "n_grid": int(grid.size),
            "cov_type": str(cov_type),
            "df": k,
        },
    )
