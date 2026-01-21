from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from ._typing import FloatArray, IntArray
from .covariance import CovType
from .data import IVData
from .diagnostics.strength import effective_f, first_stage_diagnostics
from .estimators.fit import fit
from .estimators.results import IVResults
from .weakiv import weakiv_inference


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
    ) -> IVModel:
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

    def fit(
        self,
        *,
        estimator: Literal["2sls", "liml", "fuller"] = "2sls",
        cov_type: CovType = "HC1",
        alpha: float = 1.0,
        hac_lags: int | None = None,
        kernel: str = "bartlett",
    ) -> IVResults:
        return fit(
            self.data,
            estimator=estimator,
            cov_type=cov_type,
            alpha=alpha,
            hac_lags=hac_lags,
            kernel=kernel,
        )

    def diagnostics(self) -> dict[str, Any]:
        return {
            "first_stage": first_stage_diagnostics(self.data),
            "effective_f": effective_f(self.data, cov_type="HC1"),
        }

    def weakiv(
        self,
        *,
        methods: tuple[str, ...] = ("AR", "LM", "CLR"),
        alpha: float = 0.05,
        cov_type: CovType | None = None,
        grid: tuple[float, float, int] | None = None,
        hac_lags: int | None = None,
        kernel: str = "bartlett",
    ) -> Any:
        return weakiv_inference(
            self.data,
            beta0=None,
            alpha=alpha,
            methods=methods,
            cov_type="HC1" if cov_type is None else cov_type,
            grid=grid,
            hac_lags=hac_lags,
            kernel=kernel,
        )
