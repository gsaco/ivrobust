from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.stats import chi2

from .covariance import CovType, _pinv_sym
from .data import IVData
from .intervals import invert_pvalue_grid
from .results import ConfidenceSetResult, TestResult
from .weakiv_utils import md_optimal_pi, proj, reduced_form


def lm_test(
    data: IVData,
    beta0: float | Sequence[float],
    *,
    cov_type: CovType = "HC1",
    clusters: np.ndarray | None = None,
) -> TestResult:
    """
    Lagrange multiplier (LM/K) test for H0: beta = beta0 (scalar).
    """
    if data.p_endog != 1:
        raise NotImplementedError(
            "lm_test currently supports a single endogenous regressor (p_endog=1)."
        )

    b0 = float(np.asarray(beta0, dtype=np.float64).ravel()[0])
    rf = reduced_form(data, cov_type=cov_type, clusters=clusters)
    k = rf.k_instr

    warnings: list[str] = list(rf.warnings)
    if cov_type == "unadjusted":
        residuals = rf.y - b0 * rf.d
        residuals_proj = proj(rf.z, residuals)[0]
        residuals_orth = residuals - residuals_proj

        sigma_hat = float((residuals_orth.T @ residuals_orth).ravel()[0])
        if sigma_hat <= 0 or not np.isfinite(sigma_hat):
            warnings.append("LM sigma_hat not positive; statistic set to 0")
            stat = 0.0
        else:
            Sigma = float((residuals_orth.T @ rf.d).ravel()[0]) / sigma_hat
            d_proj = proj(rf.z, rf.d)[0]
            x_tilde_proj = d_proj - residuals_proj * Sigma
            xtx = float((x_tilde_proj.T @ x_tilde_proj).ravel()[0])
            if xtx <= 0 or not np.isfinite(xtx):
                warnings.append("LM projection not positive; statistic set to 0")
                stat = 0.0
            else:
                proj_resid = x_tilde_proj * (x_tilde_proj.T @ residuals) / xtx
                dof = data.nobs - data.k_instr - data.p_exog
                if dof <= 0:
                    warnings.append("LM degrees of freedom nonpositive")
                    dof = data.nobs - data.k_instr
                stat = float(dof * (proj_resid.T @ proj_resid).ravel()[0] / sigma_hat)
    else:
        V_inv = _pinv_sym(rf.cov)
        pi_hat, r, _ = md_optimal_pi(
            b0, V_inv=V_inv, k=k, pi_y=rf.pi_y, pi_d=rf.pi_d
        )
        dvec = np.vstack([pi_hat, np.zeros_like(pi_hat)])
        score = float((dvec.T @ V_inv @ r).ravel()[0])
        info = float((dvec.T @ V_inv @ dvec).ravel()[0])

        if info <= 0 or not np.isfinite(info):
            warnings.append("LM information term not positive; statistic set to 0")
            stat = 0.0
        else:
            stat = (score**2) / info

    pval = float(chi2.sf(stat, df=1))

    return TestResult(
        statistic=float(stat),
        pvalue=pval,
        df=1,
        method="LM",
        cov_type=str(cov_type),
        warnings=tuple(warnings),
        details={"beta0": b0, "nobs": data.nobs, "k_instr": k},
    )


def lm_confidence_set(
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
) -> ConfidenceSetResult:
    """
    Invert the LM test to obtain a (possibly disjoint) confidence set for beta.
    """
    if not (0 < alpha < 1):
        raise ValueError("alpha must be in (0, 1).")
    if grid is None and n_grid < 301:
        raise ValueError("n_grid must be at least 301 for stable inversion.")
    if data.p_endog != 1:
        raise NotImplementedError(
            "lm_confidence_set currently supports a single endogenous regressor (p_endog=1)."
        )

    if grid is None:
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
        pvals[i] = lm_test(
            data, beta0=b0, cov_type=cov_type, clusters=clusters
        ).pvalue

    cs = invert_pvalue_grid(
        grid=grid,
        pvalues=pvals,
        alpha=alpha,
        refine=refine,
        refine_tol=refine_tol,
        max_refine_iter=max_refine_iter,
        pvalue_func=lambda b: lm_test(
            data, beta0=b, cov_type=cov_type, clusters=clusters
        ).pvalue,
    )

    return ConfidenceSetResult(
        confidence_set=cs,
        alpha=alpha,
        method="LM",
        grid_info={
            "grid": grid,
            "pvalues": pvals,
            "beta_bounds": (lo, hi),
            "n_grid": int(grid.size),
            "cov_type": str(cov_type),
            "df": 1,
        },
    )
