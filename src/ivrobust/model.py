from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ._typing import FloatArray, IntArray
from .data import IVData
from .diagnostics import effective_f, first_stage_diagnostics
from .estimators import fuller, liml, tsls
from .weakiv import weakiv_inference


@dataclass(frozen=True)
class IVResults:
    params: FloatArray
    stderr: FloatArray
    vcov: FloatArray
    cov_type: str
    nobs: int
    df_resid: int
    method: str
    data: IVData

    @property
    def beta(self) -> float:
        return float(self.params[-1, 0])

    @property
    def diagnostics(self) -> dict[str, Any]:
        return {
            "first_stage": first_stage_diagnostics(self.data),
            "effective_f": effective_f(self.data, cov_type=self.cov_type),
        }

    def weakiv(
        self,
        *,
        methods: tuple[str, ...] = ("AR", "LM", "CLR"),
        alpha: float = 0.05,
        cov_type: str | None = None,
        grid: tuple[float, float, int] | None = None,
    ):
        return weakiv_inference(
            self.data,
            beta0=self.beta,
            alpha=alpha,
            methods=methods,
            cov_type=self.cov_type if cov_type is None else cov_type,
            grid=grid,
        )


@dataclass(frozen=True)
class IVModel:
    data: IVData

    @classmethod
    def from_arrays(
        cls,
        y: FloatArray,
        x_endog: FloatArray,
        z: FloatArray,
        x_exog: FloatArray | None = None,
        *,
        add_const: bool = True,
        clusters: IntArray | None = None,
    ) -> "IVModel":
        y2 = np.asarray(y, dtype=np.float64)
        d2 = np.asarray(x_endog, dtype=np.float64)
        z2 = np.asarray(z, dtype=np.float64)

        if y2.ndim == 1:
            y2 = y2.reshape(-1, 1)
        if d2.ndim == 1:
            d2 = d2.reshape(-1, 1)
        if z2.ndim == 1:
            z2 = z2.reshape(-1, 1)

        n = y2.shape[0]
        if x_exog is None:
            if add_const:
                x = np.ones((n, 1), dtype=np.float64)
            else:
                raise ValueError("x_exog is None and add_const=False.")
        else:
            x_exog2 = np.asarray(x_exog, dtype=np.float64)
            if x_exog2.ndim == 1:
                x_exog2 = x_exog2.reshape(-1, 1)
            if add_const:
                x = np.hstack([np.ones((n, 1), dtype=np.float64), x_exog2])
            else:
                x = x_exog2

        data = IVData(y=y2, d=d2, x=x, z=z2, clusters=clusters)
        return cls(data=data)

    def fit(self, *, estimator: str = "2sls", cov_type: str = "HC1", alpha: float = 1.0) -> IVResults:
        est = estimator.lower()
        if est in {"2sls", "tsls"}:
            res = tsls(self.data, cov_type=cov_type)
        elif est == "liml":
            res = liml(self.data, cov_type=cov_type)
        elif est == "fuller":
            res = fuller(self.data, cov_type=cov_type, alpha=alpha)
        else:
            raise ValueError(f"Unknown estimator: {estimator}")

        return IVResults(
            params=res.params,
            stderr=res.stderr,
            vcov=res.vcov,
            cov_type=res.cov_type,
            nobs=res.nobs,
            df_resid=res.df_resid,
            method=res.method,
            data=self.data,
        )
