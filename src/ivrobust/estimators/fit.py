from __future__ import annotations

from typing import Any, Literal

from ..covariance import CovSpec, CovType
from ..data import IVData
from ..diagnostics.strength import (
    effective_f,
    first_stage_diagnostics,
    weak_id_diagnostics,
)
from .liml import fuller, liml
from .results import IVResults
from .tsls import tsls


def fit(
    data: IVData,
    *,
    estimator: Literal["2sls", "liml", "fuller"] = "2sls",
    cov: CovSpec | str | None = None,
    cov_type: CovType = "HC1",
    **kwargs: Any,
) -> IVResults:
    est = estimator.lower()
    if est in {"2sls", "tsls"}:
        res = tsls(data, cov=cov, cov_type=cov_type, clusters=kwargs.get("clusters"))
    elif est == "liml":
        res = liml(data, cov=cov, cov_type=cov_type, clusters=kwargs.get("clusters"))
    elif est == "fuller":
        res = fuller(
            data,
            alpha=float(kwargs.get("alpha", 1.0)),
            cov=cov,
            cov_type=cov_type,
            clusters=kwargs.get("clusters"),
        )
    else:
        raise ValueError(f"Unknown estimator: {estimator}")

    diagnostics = {
        "first_stage": first_stage_diagnostics(data),
        "effective_f": effective_f(data, cov=cov, cov_type=cov_type),
        "weak_id": weak_id_diagnostics(data, cov=cov, cov_type=cov_type),
    }

    return IVResults(
        params=res.params,
        stderr=res.stderr,
        vcov=res.vcov,
        cov_type=res.cov_type,
        cov_config=res.cov_config,
        nobs=res.nobs,
        df_resid=res.df_resid,
        k_endog=res.k_endog,
        k_instr=res.k_instr,
        k_exog=res.k_exog,
        method=res.method,
        data=res.data,
        diagnostics=diagnostics,
    )


__all__ = ["fit"]
