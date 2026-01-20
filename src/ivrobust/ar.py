from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.stats import chi2

from ._typing import FloatArray
from .covariance import CovType
from .data import IVData
from .intervals import IntervalSet, invert_pvalue_grid
from .results import ConfidenceSetResult, TestResult
from .weakiv_utils import reduced_form

ARTestResult = TestResult
ARConfidenceSetResult = ConfidenceSetResult


def ar_test(
    data: IVData,
    *,
    beta0: float | Sequence[float],
    cov_type: CovType = "HC1",
    clusters: np.ndarray | None = None,
) -> ARTestResult:
    """
    Anderson-Rubin test of H0: beta = beta0 (single endogenous regressor).

    Parameters
    ----------
    data
        IVData with y, d, x, z.
    beta0
        Null value for the coefficient on d. If sequence is given, the first
        element is used.
    cov_type
        Covariance type for the Wald statistic: "unadjusted", "HC0", "HC1",
        "HC2", "HC3", or "cluster".

    Returns
    -------
    TestResult
        Contains chi-square Wald statistic and p-value.

    Notes
    -----
    The implementation performs the AR regression:

        y - beta0 * d  ~  x + z

    and tests that the coefficients on the excluded instruments z are jointly
    zero.
    """
    if data.p_endog != 1:
        raise NotImplementedError(
            "ar_test currently supports a single endogenous regressor (p_endog=1)."
        )
    b0 = float(np.asarray(beta0, dtype=np.float64).ravel()[0])
    rf = reduced_form(data, cov_type=cov_type, clusters=clusters)

    k = rf.k_instr
    pi_y = rf.pi_y
    pi_d = rf.pi_d
    g = pi_y - b0 * pi_d

    V = rf.cov
    V_yy = V[:k, :k]
    V_yd = V[:k, k:]
    V_dd = V[k:, k:]
    V_g = V_yy - b0 * (V_yd + V_yd.T) + (b0**2) * V_dd
    Vg_inv = np.linalg.pinv(V_g)
    stat = float((g.T @ Vg_inv @ g).ravel()[0])
    pval = float(chi2.sf(stat, df=k))

    return TestResult(
        statistic=stat,
        df=k,
        pvalue=pval,
        method="AR",
        cov_type=str(cov_type),
        warnings=rf.warnings,
        details={
            "beta0": b0,
            "nobs": data.nobs,
            "k_instr": k,
        },
    )


def ar_confidence_set(
    data: IVData,
    *,
    alpha: float = 0.05,
    cov_type: CovType = "HC1",
    clusters: np.ndarray | None = None,
    grid: FloatArray | None = None,
    beta_bounds: tuple[float, float] | None = None,
    n_grid: int = 2001,
    refine: bool = True,
    refine_tol: float = 1e-6,
    max_refine_iter: int = 80,
) -> ConfidenceSetResult:
    """
    Invert the AR test to obtain a (possibly disjoint) confidence set for beta.

    Parameters
    ----------
    data
        IVData.
    alpha
        Size of the test; confidence level is 1 - alpha.
    cov_type
        "unadjusted", "HC0", "HC1", "HC2", "HC3", or "cluster".
    beta_bounds
        Search interval (low, high). If None, a data-dependent default is used.
    n_grid
        Number of grid points used for inversion.
    refine
        If True, refine boundaries via bisection for improved interval endpoints.
    refine_tol
        Absolute tolerance for boundary refinement.
    max_refine_iter
        Maximum bisection iterations per boundary.

    Returns
    -------
    ARConfidenceSetResult
        Contains IntervalSet with (possibly unbounded) intervals.
    """
    if data.p_endog != 1:
        raise NotImplementedError(
            "ar_confidence_set currently supports a single endogenous regressor (p_endog=1)."
        )
    if not (0 < alpha < 1):
        raise ValueError("alpha must be in (0, 1).")
    if grid is None and n_grid < 301:
        raise ValueError("n_grid must be at least 301 for stable inversion.")

    if grid is None:
        # Default bounds: scale by outcome/endogenous magnitude and ensure a wide range.
        if beta_bounds is None:
            y_std = float(np.std(data.y))
            d_std = float(np.std(data.d))
            scale = y_std / max(d_std, 1e-12)
            width = max(10.0 * scale, 10.0)
            beta_bounds = (-width, width)

        lo, hi = map(float, beta_bounds)
        if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
            raise ValueError("beta_bounds must be finite with lo < hi.")
        grid = np.linspace(lo, hi, n_grid, dtype=np.float64)
    else:
        grid = np.asarray(grid, dtype=np.float64).reshape(-1)
        if grid.size < 3:
            raise ValueError("grid must contain at least 3 points.")
        lo, hi = float(grid[0]), float(grid[-1])

    pvals = np.empty_like(grid)
    for i, b0 in enumerate(grid):
        pvals[i] = ar_test(
            data, beta0=b0, cov_type=cov_type, clusters=clusters
        ).pvalue

    cs = invert_pvalue_grid(
        grid=grid,
        pvalues=pvals,
        alpha=alpha,
        refine=refine,
        refine_tol=refine_tol,
        max_refine_iter=max_refine_iter,
        pvalue_func=lambda b: ar_test(
            data, beta0=b, cov_type=cov_type, clusters=clusters
        ).pvalue,
    )

    return ConfidenceSetResult(
        confidence_set=cs,
        alpha=alpha,
        method="AR",
        grid_info={
            "grid": grid,
            "pvalues": pvals,
            "beta_bounds": (lo, hi),
            "n_grid": int(grid.size),
            "cov_type": str(cov_type),
            "df": data.k_instr,
        },
    )
