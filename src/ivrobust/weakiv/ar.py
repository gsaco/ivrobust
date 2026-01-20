from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.stats import chi2

from .._typing import FloatArray
from ..covariance import CovSpec, CovType
from ..data import IVData
from ..linalg.ops import sym_solve
from ..weakiv_utils import default_beta_bounds, reduced_form
from .inversion import GridSpec, InversionSpec, invert_test
from .results import ARTestResult, ConfidenceSetResult


def ar_test(
    data: IVData,
    beta0: float | Sequence[float],
    *,
    cov: CovSpec | str | None = None,
    cov_type: CovType = "HC1",
    clusters: np.ndarray | None = None,
    alpha: float | None = None,
) -> ARTestResult:
    """
    Anderson-Rubin test of H0: beta = beta0 (single endogenous regressor).
    """
    if data.p_endog != 1:
        raise NotImplementedError(
            "ar_test currently supports a single endogenous regressor (p_endog=1)."
        )
    b0 = float(np.asarray(beta0, dtype=np.float64).ravel()[0])
    rf = reduced_form(
        data, cov_type=cov_type, cov=cov, clusters=clusters, kernel="bartlett"
    )

    k = rf.k_instr
    pi_y = rf.pi_y
    pi_d = rf.pi_d
    g = pi_y - b0 * pi_d

    V = rf.cov
    V_yy = V[:k, :k]
    V_yd = V[:k, k:]
    V_dd = V[k:, k:]
    V_g = V_yy - b0 * (V_yd + V_yd.T) + (b0**2) * V_dd
    x = sym_solve(V_g, g)
    stat = float((g.T @ x).ravel()[0])
    pval = float(chi2.sf(stat, df=k))

    return ARTestResult(
        statistic=stat,
        df=k,
        pvalue=pval,
        method="AR",
        cov_type=str(cov_type) if cov is None else str(getattr(cov, "cov_type", cov)),
        alpha=alpha,
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
    cov: CovSpec | str | None = None,
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
    """
    if data.p_endog != 1:
        raise NotImplementedError(
            "ar_confidence_set currently supports a single endogenous regressor (p_endog=1)."
        )
    if grid is None and beta_bounds is None:
        beta_bounds = default_beta_bounds(data)
    grid_spec = GridSpec(grid=grid, beta_bounds=beta_bounds, n_grid=n_grid)
    inversion_spec = InversionSpec(
        refine=refine, refine_tol=refine_tol, max_refine_iter=max_refine_iter
    )

    cs, grid_info = invert_test(
        test_fn=lambda b: ar_test(
            data,
            beta0=b,
            cov=cov,
            cov_type=cov_type,
            clusters=clusters,
            alpha=alpha,
        ).pvalue,
        alpha=alpha,
        grid_spec=grid_spec,
        inversion_spec=inversion_spec,
    )

    grid_info.update({"cov_type": str(cov_type), "df": data.k_instr})

    return ConfidenceSetResult(
        confidence_set=cs,
        alpha=alpha,
        method="AR",
        grid_info=grid_info,
    )


__all__ = ["ARConfidenceSetResult", "ARTestResult", "ar_confidence_set", "ar_test"]

ARConfidenceSetResult = ConfidenceSetResult
