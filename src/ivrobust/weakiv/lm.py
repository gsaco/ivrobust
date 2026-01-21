from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.stats import chi2

from ..covariance import CovSpec, CovType, _pinv_sym
from ..data import IVData
from ..linalg.ops import sym_solve
from ..weakiv_utils import default_beta_bounds, md_optimal_pi, proj, reduced_form
from .inversion import GridSpec, InversionSpec, invert_test
from .results import ConfidenceSetResult, LMTestResult


def kp_lm_test(
    data: IVData,
    beta0: float | Sequence[float],
    *,
    cov: CovSpec | str | None = None,
    cov_type: CovType = "HC1",
    clusters: np.ndarray | None = None,
    hac_lags: int | None = None,
    kernel: str = "bartlett",
    alpha: float | None = None,
) -> LMTestResult:
    """
    Kleibergen-Paap LM test for H0: beta = beta0 (scalar).
    """
    if data.p_endog != 1:
        raise NotImplementedError(
            "kp_lm_test currently supports a single endogenous regressor (p_endog=1)."
        )

    b0 = float(np.asarray(beta0, dtype=np.float64).ravel()[0])
    rf = reduced_form(
        data,
        cov_type=cov_type,
        cov=cov,
        clusters=clusters,
        hac_lags=hac_lags,
        kernel=kernel,
    )
    k = rf.k_instr

    warnings: list[str] = list(rf.warnings)
    if cov_type == "unadjusted" and cov is None:
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
        score = float((dvec.T @ sym_solve(rf.cov, r)).ravel()[0])
        info = float((dvec.T @ sym_solve(rf.cov, dvec)).ravel()[0])

        if info <= 0 or not np.isfinite(info):
            warnings.append("LM information term not positive; statistic set to 0")
            stat = 0.0
        else:
            stat = (score**2) / info

    pval = float(chi2.sf(stat, df=1))

    return LMTestResult(
        statistic=float(stat),
        pvalue=pval,
        df=1,
        method="KP-LM",
        cov_type=str(cov_type) if cov is None else str(getattr(cov, "cov_type", cov)),
        alpha=alpha,
        cov_config={"hac_lags": hac_lags, "kernel": kernel}
        if str(cov_type).upper() == "HAC"
        else {},
        warnings=tuple(warnings),
        details={"beta0": b0, "nobs": data.nobs, "k_instr": k},
    )


def lm_test(
    data: IVData,
    beta0: float | Sequence[float],
    *,
    cov: CovSpec | str | None = None,
    cov_type: CovType = "HC1",
    clusters: np.ndarray | None = None,
    hac_lags: int | None = None,
    kernel: str = "bartlett",
    alpha: float | None = None,
) -> LMTestResult:
    return kp_lm_test(
        data,
        beta0,
        cov=cov,
        cov_type=cov_type,
        clusters=clusters,
        hac_lags=hac_lags,
        kernel=kernel,
        alpha=alpha,
    )


def kp_rank_test(
    data: IVData,
    *,
    cov: CovSpec | str | None = None,
    cov_type: CovType = "HC1",
    clusters: np.ndarray | None = None,
    hac_lags: int | None = None,
    kernel: str = "bartlett",
) -> LMTestResult:
    """
    Kleibergen-Paap rk underidentification test (scalar endogenous regressor).
    """
    if data.p_endog != 1:
        raise NotImplementedError(
            "kp_rank_test currently supports a single endogenous regressor (p_endog=1)."
        )
    rf = reduced_form(
        data,
        cov_type=cov_type,
        cov=cov,
        clusters=clusters,
        hac_lags=hac_lags,
        kernel=kernel,
    )
    k = rf.k_instr
    V_dd = rf.cov[k:, k:]
    stat = float((rf.pi_d.T @ sym_solve(V_dd, rf.pi_d)).ravel()[0])
    pval = float(chi2.sf(stat, df=k))
    return LMTestResult(
        statistic=stat,
        pvalue=pval,
        df=k,
        method="KP-rk",
        cov_type=str(cov_type) if cov is None else str(getattr(cov, "cov_type", cov)),
        cov_config={"hac_lags": hac_lags, "kernel": kernel}
        if str(cov_type).upper() == "HAC"
        else {},
        warnings=rf.warnings,
        details={"nobs": data.nobs, "k_instr": k},
    )


def lm_confidence_set(
    data: IVData,
    *,
    alpha: float = 0.05,
    cov: CovSpec | str | None = None,
    cov_type: CovType = "HC1",
    clusters: np.ndarray | None = None,
    hac_lags: int | None = None,
    kernel: str = "bartlett",
    grid: np.ndarray | None = None,
    beta_bounds: tuple[float, float] | None = None,
    n_grid: int = 2001,
    refine: bool = True,
    refine_tol: float = 1e-6,
    max_refine_iter: int = 80,
) -> ConfidenceSetResult:
    if data.p_endog != 1:
        raise NotImplementedError(
            "lm_confidence_set currently supports a single endogenous regressor (p_endog=1)."
        )

    if grid is None and beta_bounds is None:
        beta_bounds = default_beta_bounds(data)
    grid_spec = GridSpec(grid=grid, beta_bounds=beta_bounds, n_grid=n_grid)
    inversion_spec = InversionSpec(
        refine=refine, refine_tol=refine_tol, max_refine_iter=max_refine_iter
    )
    cs, grid_info = invert_test(
        test_fn=lambda b: lm_test(
            data,
            beta0=b,
            cov=cov,
            cov_type=cov_type,
            clusters=clusters,
            hac_lags=hac_lags,
            kernel=kernel,
            alpha=alpha,
        ).pvalue,
        alpha=alpha,
        grid_spec=grid_spec,
        inversion_spec=inversion_spec,
    )
    grid_info.update(
        {"cov_type": str(cov_type), "df": 1, "hac_lags": hac_lags, "kernel": kernel}
    )
    return ConfidenceSetResult(
        confidence_set=cs, alpha=alpha, method="LM", grid_info=grid_info
    )


__all__ = ["kp_lm_test", "kp_rank_test", "lm_confidence_set", "lm_test"]
