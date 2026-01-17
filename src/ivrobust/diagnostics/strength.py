"""Instrument strength diagnostics."""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import linalg  # type: ignore[import-untyped]

from ivrobust.data.design import IVDesign
from ivrobust.data.ivdata import IVData
from ivrobust.utils.specs import StrengthDiagnostics
from ivrobust.utils.warnings import DataWarning, warn_and_record


def first_stage_diagnostics(
    data: IVData, design: Optional[IVDesign] = None
) -> StrengthDiagnostics:
    """Compute first-stage F statistics and partial R2 values."""

    if design is None:
        design = IVDesign.from_data(data)
    z = design.z_resid
    x = design.x_endog_resid
    n, k = z.shape
    q = data.q

    rank_z = np.linalg.matrix_rank(z) if z.size else 0
    if rank_z < k:
        warn_and_record(
            "Z is rank deficient after partialling out X_exog.",
            category=DataWarning,
            record=None,
        )

    f_stats = np.full(x.shape[1], np.nan)
    partial_r2 = np.full(x.shape[1], np.nan)

    df_num = k
    df_den = n - k - q
    if df_den <= 0:
        warn_and_record(
            "Insufficient degrees of freedom for first-stage F statistics.",
            category=DataWarning,
            record=None,
        )

    for idx in range(x.shape[1]):
        x_col = x[:, idx]
        coef, _, rank, _ = linalg.lstsq(z, x_col, cond=None, check_finite=False)
        x_hat = z @ coef
        resid = x_col - x_hat
        ssr = float(np.dot(x_hat, x_hat))
        sse = float(np.dot(resid, resid))
        if ssr + sse > 0:
            r2 = ssr / (ssr + sse)
            partial_r2[idx] = r2
            if df_den > 0:
                f_stats[idx] = (r2 / (1 - r2)) * (df_den / df_num)
        if rank < k:
            warn_and_record(
                "Z rank deficiency detected in first-stage regression.",
                category=DataWarning,
                record=None,
            )

    diagnostics = StrengthDiagnostics(
        first_stage_F=f_stats,
        partial_R2=partial_r2,
        effective_F=None,
        k=k,
        n=n,
        k_over_n=k / n if n > 0 else None,
        rank_z=rank_z,
        warnings=[],
    )

    if diagnostics.k_over_n is not None and diagnostics.k_over_n > 0.1:
        diagnostics.warnings.append(
            "k/n is large; fixed-k weak-IV approximations may be unreliable."
        )

    return diagnostics


def effective_f_statistic(*args: object, **kwargs: object) -> float:
    """Placeholder for effective F (Montiel Olea-Pflueger).

    This is not implemented in v0.1 because it requires careful validation.
    """

    raise NotImplementedError(
        "Effective F is not implemented yet; see documentation for scope."
    )
