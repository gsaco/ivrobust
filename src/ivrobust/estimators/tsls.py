"""Two-stage least squares estimator."""

from __future__ import annotations

from typing import Optional

import numpy as np

from ivrobust.cov import moment_covariance
from ivrobust.data.design import IVDesign
from ivrobust.data.ivdata import IVData
from ivrobust.diagnostics import first_stage_diagnostics
from ivrobust.linalg import safe_solve
from ivrobust.utils.specs import AssumptionSpec
from ivrobust.utils.warnings import WarningRecord

from .results import EstimatorResult


def tsls(
    data: IVData,
    *,
    cov: str = "HC1",
    clusters: Optional[np.ndarray] = None,
    df_adjust: bool = True,
) -> EstimatorResult:
    """Compute the TSLS estimator and strong-ID covariance."""

    warnings = WarningRecord()
    design = IVDesign.from_data(data)

    z = design.z_resid
    x = design.x_endog_resid
    y = design.y_resid
    n = data.n

    ztz = design.cross_products["ZtZ"]
    ztx = design.cross_products["ZtX"]

    ztz_inv, diag_ztz = safe_solve(
        ztz,
        np.eye(ztz.shape[0]),
        context="Z'Z inversion",
        warnings=warnings,
    )
    x_hat = z @ (ztz_inv @ ztx)
    beta, diag_beta = safe_solve(
        x_hat.T @ x,
        x_hat.T @ y,
        context="TSLS normal equations",
        warnings=warnings,
    )
    beta = beta.reshape(-1)

    fitted = x @ beta
    residuals = y - fitted

    omega, cov_spec = moment_covariance(
        z,
        residuals,
        kind=cov,
        clusters=clusters if clusters is not None else data.clusters,
        df_adjust=df_adjust,
        warnings=warnings,
    )

    g = (z.T @ x) / n
    w = n * ztz_inv
    a = g.T @ w @ g
    a_inv, diag_a = safe_solve(
        a,
        np.eye(a.shape[0]),
        context="TSLS covariance A inversion",
        warnings=warnings,
    )
    middle = g.T @ w @ omega @ w @ g
    cov_beta = (a_inv @ middle @ a_inv) / n

    std_errors = np.sqrt(np.diag(cov_beta))

    numerics = design.numerics
    numerics.merge(diag_ztz)
    numerics.merge(diag_beta)
    numerics.merge(diag_a)

    assumptions = AssumptionSpec(
        id_regime="strong",
        error="heteroskedastic",
        asymptotics="fixed_k",
        notes=("TSLS standard errors are not weak-IV robust.",),
        citations=(),
    )

    diagnostics = first_stage_diagnostics(data, design=design)

    return EstimatorResult(
        params=beta,
        cov=cov_beta,
        std_errors=std_errors,
        residuals=residuals,
        fitted=fitted,
        method="TSLS",
        cov_spec=cov_spec,
        assumptions=assumptions,
        diagnostics=diagnostics,
        numerics=numerics,
        warnings=tuple(warnings.messages),
    )
