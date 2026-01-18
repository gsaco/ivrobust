from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.stats import chi2

from ._typing import FloatArray
from .covariance import CovType, cov_ols
from .data import IVData


@dataclass(frozen=True)
class ARTestResult:
    """
    Anderson-Rubin (AR) test result.

    The AR test is valid under weak instruments in linear IV models when testing
    the structural coefficient on a (single) endogenous regressor by testing
    whether excluded instruments predict the null-implied residual.

    References
    ----------
    - Anderson, T. W., and Rubin, H. (1949). Estimation of the Parameters of a
      Single Equation in a Complete System of Stochastic Equations. Annals of
      Mathematical Statistics. DOI: 10.1214/aoms/1177730090.
    - Andrews, I., Stock, J. H., and Sun, L. (2019). Weak Instruments in
      Instrumental Variables Regression: Theory and Practice. Annual Review of
      Economics. DOI: 10.1146/annurev-economics-080218-025643.
    """

    statistic: float
    df: int
    pvalue: float
    beta0: float
    cov_type: str
    nobs: int
    k_instr: int


@dataclass(frozen=True)
class IntervalSet:
    """
    A (possibly disjoint) set of intervals on the real line.

    Intervals are represented as (lower, upper), where bounds may be -inf/inf.
    """

    intervals: list[tuple[float, float]]

    def contains(self, x: float) -> bool:
        for lo, hi in self.intervals:
            if lo <= x <= hi:
                return True
        return False


@dataclass(frozen=True)
class ARConfidenceSetResult:
    confidence_set: IntervalSet
    alpha: float
    critical_value: float
    df: int
    cov_type: str
    grid: FloatArray


def _stack_xz(data: IVData) -> FloatArray:
    return np.hstack([data.x, data.z]).astype(np.float64, copy=False)


def ar_test(
    data: IVData,
    *,
    beta0: float | Sequence[float],
    cov_type: CovType = "HC1",
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
        Covariance type for the Wald statistic: "HC0", "HC1", or "cluster".

    Returns
    -------
    ARTestResult
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

    y = data.y.reshape(-1, 1)
    d = data.d.reshape(-1, 1)
    y_tilde = y - b0 * d

    W = _stack_xz(data)
    coef, *_ = np.linalg.lstsq(W, y_tilde, rcond=None)
    resid = y_tilde - W @ coef

    cov_res = cov_ols(
        X=W,
        resid=resid,
        cov_type=cov_type,
        clusters=data.clusters if cov_type == "cluster" else None,
    )

    p_x = data.p_exog
    k = data.k_instr

    b_z = coef[p_x : p_x + k, :].reshape(-1, 1)
    V_z = cov_res.cov[p_x : p_x + k, p_x : p_x + k]

    Vz_inv = np.linalg.pinv(V_z)
    stat = float((b_z.T @ Vz_inv @ b_z).ravel()[0])
    pval = float(chi2.sf(stat, df=k))

    return ARTestResult(
        statistic=stat,
        df=k,
        pvalue=pval,
        beta0=b0,
        cov_type=str(cov_type),
        nobs=data.nobs,
        k_instr=k,
    )


def ar_confidence_set(
    data: IVData,
    *,
    alpha: float = 0.05,
    cov_type: CovType = "HC1",
    beta_bounds: tuple[float, float] | None = None,
    n_grid: int = 2001,
    refine: bool = True,
    refine_tol: float = 1e-6,
    max_refine_iter: int = 80,
) -> ARConfidenceSetResult:
    """
    Invert the AR test to obtain a (possibly disjoint) confidence set for beta.

    Parameters
    ----------
    data
        IVData.
    alpha
        Size of the test; confidence level is 1 - alpha.
    cov_type
        "HC0", "HC1", or "cluster".
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
    if not (0 < alpha < 1):
        raise ValueError("alpha must be in (0, 1).")
    if n_grid < 301:
        raise ValueError("n_grid must be at least 301 for stable inversion.")
    if data.p_endog != 1:
        raise NotImplementedError(
            "ar_confidence_set currently supports a single endogenous regressor (p_endog=1)."
        )

    df = data.k_instr
    critical = float(chi2.ppf(1.0 - alpha, df=df))

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

    stats = np.empty_like(grid)
    for i, b0 in enumerate(grid):
        stats[i] = ar_test(data, beta0=b0, cov_type=cov_type).statistic

    inside = stats <= critical

    def f(b: float) -> float:
        return ar_test(data, beta0=b, cov_type=cov_type).statistic - critical

    def bisect(a: float, b: float) -> float:
        fa = f(a)
        fb = f(b)
        if fa == 0.0:
            return a
        if fb == 0.0:
            return b
        if fa * fb > 0:
            # Not bracketed; return midpoint as conservative fallback.
            return 0.5 * (a + b)

        left, right = a, b
        for _ in range(max_refine_iter):
            mid = 0.5 * (left + right)
            fm = f(mid)
            if abs(fm) <= refine_tol or abs(right - left) <= refine_tol:
                return mid
            if fa * fm <= 0:
                right = mid
                fb = fm
            else:
                left = mid
                fa = fm
        return 0.5 * (left + right)

    intervals: list[tuple[float, float]] = []

    # Identify contiguous True segments.
    idx = np.flatnonzero(inside)
    if idx.size == 0:
        cs = IntervalSet(intervals=[])
        return ARConfidenceSetResult(
            confidence_set=cs,
            alpha=alpha,
            critical_value=critical,
            df=df,
            cov_type=str(cov_type),
            grid=grid.reshape(-1, 1),
        )

    # Walk segments in order.
    start = idx[0]
    prev = idx[0]
    for j in idx[1:]:
        if j == prev + 1:
            prev = j
            continue
        # segment [start, prev]
        intervals.append((grid[start], grid[prev]))
        start = j
        prev = j
    intervals.append((grid[start], grid[prev]))

    # Convert grid-segment endpoints to refined endpoints and handle unboundedness.
    refined: list[tuple[float, float]] = []
    for seg_lo, seg_hi in intervals:
        left = float(seg_lo)
        right = float(seg_hi)

        # Extend to -inf/+inf if the segment touches the search bounds.
        unbounded_left = np.isclose(left, lo)
        unbounded_right = np.isclose(right, hi)

        if refine:
            if not unbounded_left:
                # Boundary is between previous grid point (outside) and seg_lo (inside).
                left = bisect(left - (grid[1] - grid[0]), left)
            if not unbounded_right:
                right = bisect(right, right + (grid[1] - grid[0]))

        refined.append(
            (-np.inf if unbounded_left else left, np.inf if unbounded_right else right)
        )

    cs = IntervalSet(intervals=refined)
    return ARConfidenceSetResult(
        confidence_set=cs,
        alpha=alpha,
        critical_value=critical,
        df=df,
        cov_type=str(cov_type),
        grid=grid.reshape(-1, 1),
    )
