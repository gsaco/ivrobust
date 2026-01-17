"""Cached IV design matrices and residualized variables."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ivrobust.data.ivdata import IVData
from ivrobust.linalg import matrix_diagnostics, residualize
from ivrobust.utils.specs import NumericalDiagnostics


@dataclass(frozen=True)
class IVDesign:
    """Cached residualized variables and cross-products."""

    data: IVData
    y: np.ndarray
    x_endog: np.ndarray
    x_exog: np.ndarray
    z: np.ndarray
    y_resid: np.ndarray
    x_endog_resid: np.ndarray
    z_resid: np.ndarray
    cross_products: dict[str, np.ndarray]
    numerics: NumericalDiagnostics

    @classmethod
    def from_data(cls, data: IVData) -> IVDesign:
        y = data.y
        x_endog = data.x_endog
        x_exog = data.x_exog if data.x_exog is not None else np.zeros((data.n, 0))
        z = data.z

        if data.weights is not None:
            w_sqrt = np.sqrt(data.weights)
            y = y * w_sqrt
            x_endog = x_endog * w_sqrt[:, None]
            x_exog = x_exog * w_sqrt[:, None]
            z = z * w_sqrt[:, None]

        y_resid, diag_y = residualize(y, x_exog, name="X_exog")
        x_endog_resid, diag_x = residualize(x_endog, x_exog, name="X_exog")
        z_resid, diag_z = residualize(z, x_exog, name="X_exog")

        numerics = NumericalDiagnostics()
        numerics.merge(diag_y)
        numerics.merge(diag_x)
        numerics.merge(diag_z)
        numerics.merge(matrix_diagnostics(z_resid, "Z_resid"))
        numerics.merge(matrix_diagnostics(x_endog_resid, "X_endog_resid"))

        cross_products = {
            "ZtZ": z_resid.T @ z_resid,
            "ZtX": z_resid.T @ x_endog_resid,
            "XtZ": x_endog_resid.T @ z_resid,
            "XtX": x_endog_resid.T @ x_endog_resid,
        }

        return cls(
            data=data,
            y=y,
            x_endog=x_endog,
            x_exog=x_exog,
            z=z,
            y_resid=y_resid,
            x_endog_resid=x_endog_resid,
            z_resid=z_resid,
            cross_products=cross_products,
            numerics=numerics,
        )
