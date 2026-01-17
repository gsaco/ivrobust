"""Limited information maximum likelihood estimator."""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import linalg  # type: ignore[import-untyped]

from ivrobust.data.design import IVDesign
from ivrobust.data.ivdata import IVData
from ivrobust.diagnostics import first_stage_diagnostics
from ivrobust.linalg import safe_solve
from ivrobust.utils.specs import AssumptionSpec
from ivrobust.utils.warnings import NumericalWarning, WarningRecord, warn_and_record

from .results import EstimatorResult


def _min_generalized_eigenvalue(a: np.ndarray, b: np.ndarray) -> float:
    """Smallest generalized eigenvalue of (a, b)."""

    try:
        eigvals = linalg.eigvalsh(a=a, b=b, check_finite=False)
    except linalg.LinAlgError:
        eigvals = np.real(linalg.eigvals(a=a, b=b, check_finite=False))
        eigvals = eigvals[np.isfinite(eigvals)]
    if eigvals.size == 0:
        return float("inf")
    return float(np.min(eigvals))


def liml(
    data: IVData,
    *,
    clusters: Optional[np.ndarray] = None,
) -> EstimatorResult:
    """Compute the LIML estimator (point estimate only)."""

    warnings = WarningRecord()
    design = IVDesign.from_data(data)

    z = design.z_resid
    x = design.x_endog_resid
    y = design.y_resid

    ztz = design.cross_products["ZtZ"]
    ztz_inv, diag_ztz = safe_solve(
        ztz,
        np.eye(ztz.shape[0]),
        context="Z'Z inversion",
        warnings=warnings,
    )

    x_proj = z @ (ztz_inv @ (z.T @ x))
    y_proj = z @ (ztz_inv @ (z.T @ y))

    xy = np.column_stack([x, y])
    xy_proj = np.column_stack([x_proj, y_proj])
    xy_orth = xy - xy_proj

    a = xy_proj.T @ xy_proj
    b = xy_orth.T @ xy_orth

    ar_min = _min_generalized_eigenvalue(a, b)
    if not np.isfinite(ar_min):
        warn_and_record(
            "Failed to compute LIML eigenvalue; returning NaN parameters.",
            category=NumericalWarning,
            record=warnings,
        )
        beta = np.full(x.shape[1], np.nan)
        fitted = np.full_like(y, np.nan)
        residuals = np.full_like(y, np.nan)
    else:
        kappa = 1.0 + ar_min
        x_kappa = kappa * x_proj + (1.0 - kappa) * x
        beta, diag_beta = safe_solve(
            x_kappa.T @ x,
            x_kappa.T @ y,
            context="LIML normal equations",
            warnings=warnings,
        )
        beta = beta.reshape(-1)
        fitted = x @ beta
        residuals = y - fitted
        design.numerics.merge(diag_beta)

    design.numerics.merge(diag_ztz)

    assumptions = AssumptionSpec(
        id_regime="strong",
        error="heteroskedastic",
        asymptotics="fixed_k",
        notes=(
            "LIML covariance is not implemented in v0.1; only point estimates are "
            "returned.",
        ),
        citations=(),
    )

    diagnostics = first_stage_diagnostics(data, design=design)

    return EstimatorResult(
        params=beta,
        cov=None,
        std_errors=None,
        residuals=residuals,
        fitted=fitted,
        method="LIML",
        cov_spec=None,
        assumptions=assumptions,
        diagnostics=diagnostics,
        numerics=design.numerics,
        warnings=tuple(warnings.messages),
    )
