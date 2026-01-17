"""Validated IV dataset container."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike

from ivrobust.utils.arrays import as_1d_array, as_2d_array, check_finite
from ivrobust.utils.warnings import DataWarning, warn_and_record


@dataclass(frozen=True)
class IVData:
    """Validated IV dataset container.

    This class expects that missing data has already been handled. Use
    `IVData.from_arrays` for listwise deletion and optional intercept handling.
    """

    y: np.ndarray
    x_endog: np.ndarray
    x_exog: Optional[np.ndarray]
    z: np.ndarray
    weights: Optional[np.ndarray] = None
    clusters: Optional[np.ndarray] = None
    time: Optional[np.ndarray] = None
    metadata: dict[str, object] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        warnings_list = list(self.warnings)

        y = as_1d_array(self.y, "y")
        check_finite(y, "y")
        n = y.shape[0]

        x_endog = as_2d_array(self.x_endog, "X_endog", n_rows=n)
        check_finite(x_endog, "X_endog")

        if self.x_exog is None:
            x_exog = np.zeros((n, 0), dtype=float)
        else:
            x_exog = as_2d_array(self.x_exog, "X_exog", n_rows=n)
            check_finite(x_exog, "X_exog")

        z = as_2d_array(self.z, "Z", n_rows=n)
        check_finite(z, "Z")

        if z.shape[1] == 0:
            raise ValueError("Z must have at least one column.")
        if x_endog.shape[1] == 0:
            raise ValueError("X_endog must have at least one column.")

        weights = None
        if self.weights is not None:
            weights = as_1d_array(self.weights, "weights")
            if weights.shape[0] != n:
                raise ValueError("weights length must match y.")
            if np.any(weights <= 0):
                raise ValueError("weights must be positive.")
            check_finite(weights, "weights")

        clusters = None
        if self.clusters is not None:
            clusters_arr = np.asarray(self.clusters)
            if clusters_arr.ndim == 2 and clusters_arr.shape[1] > 1:
                raise NotImplementedError("Multi-way clustering is not supported.")
            if clusters_arr.ndim == 2:
                clusters_arr = clusters_arr[:, 0]
            if clusters_arr.shape[0] != n:
                raise ValueError("clusters length must match y.")
            clusters = clusters_arr

        time = None
        if self.time is not None:
            time_arr = as_1d_array(self.time, "time")
            if time_arr.shape[0] != n:
                raise ValueError("time length must match y.")
            time = time_arr

        if x_exog.shape[1] > 0:
            rank = np.linalg.matrix_rank(x_exog)
            if rank < x_exog.shape[1]:
                message = "X_exog is rank deficient; residualization may be unstable."
                warn_and_record(message, category=DataWarning, record=None)
                warnings_list.append(message)

        object.__setattr__(self, "y", np.asarray(y, dtype=float))
        object.__setattr__(self, "x_endog", np.asarray(x_endog, dtype=float, order="F"))
        object.__setattr__(self, "x_exog", np.asarray(x_exog, dtype=float, order="F"))
        object.__setattr__(self, "z", np.asarray(z, dtype=float, order="F"))
        object.__setattr__(
            self,
            "weights",
            None if weights is None else np.asarray(weights, dtype=float),
        )
        object.__setattr__(self, "clusters", clusters)
        object.__setattr__(self, "time", time)
        object.__setattr__(self, "warnings", tuple(warnings_list))

    @property
    def n(self) -> int:
        return int(self.y.shape[0])

    @property
    def p(self) -> int:
        return int(self.x_endog.shape[1])

    @property
    def q(self) -> int:
        return int(self.x_exog.shape[1]) if self.x_exog is not None else 0

    @property
    def k(self) -> int:
        return int(self.z.shape[1])

    @classmethod
    def from_arrays(
        cls,
        y: ArrayLike,
        X_endog: ArrayLike,
        X_exog: Optional[ArrayLike],
        Z: ArrayLike,
        *,
        weights: Optional[ArrayLike] = None,
        clusters: Optional[ArrayLike] = None,
        time: Optional[ArrayLike] = None,
        metadata: Optional[dict[str, object]] = None,
        drop_missing: bool = False,
        add_intercept: bool = False,
    ) -> IVData:
        """Construct IVData with optional listwise deletion and intercept."""

        y_arr = as_1d_array(y, "y")
        x_endog_arr = as_2d_array(X_endog, "X_endog", n_rows=y_arr.shape[0])
        x_exog_arr = None
        if X_exog is not None:
            x_exog_arr = as_2d_array(X_exog, "X_exog", n_rows=y_arr.shape[0])
        z_arr = as_2d_array(Z, "Z", n_rows=y_arr.shape[0])

        if add_intercept:
            intercept = np.ones((y_arr.shape[0], 1), dtype=float)
            if x_exog_arr is None or x_exog_arr.shape[1] == 0:
                x_exog_arr = intercept
            else:
                has_const = np.any(
                    np.all(np.isclose(x_exog_arr, 1.0, atol=1e-12), axis=0)
                )
                if has_const:
                    warn_and_record(
                        "X_exog already contains a constant; intercept not added.",
                        category=DataWarning,
                        record=None,
                    )
                else:
                    x_exog_arr = np.hstack([intercept, x_exog_arr])

        meta = {} if metadata is None else dict(metadata)

        if drop_missing:
            mask = np.isfinite(y_arr)
            mask &= np.all(np.isfinite(x_endog_arr), axis=1)
            mask &= np.all(np.isfinite(z_arr), axis=1)
            if x_exog_arr is not None:
                mask &= np.all(np.isfinite(x_exog_arr), axis=1)
            if weights is not None:
                weights_arr = as_1d_array(weights, "weights")
                mask &= np.isfinite(weights_arr)
            if time is not None:
                time_arr = as_1d_array(time, "time")
                mask &= np.isfinite(time_arr)
            dropped = int(np.sum(~mask))
            if dropped > 0:
                warn_and_record(
                    f"Dropped {dropped} rows due to missing values.",
                    category=DataWarning,
                    record=None,
                )
                y_arr = y_arr[mask]
                x_endog_arr = x_endog_arr[mask]
                z_arr = z_arr[mask]
                if x_exog_arr is not None:
                    x_exog_arr = x_exog_arr[mask]
                if weights is not None:
                    weights = np.asarray(weights)[mask]
                if clusters is not None:
                    clusters = np.asarray(clusters)[mask]
                if time is not None:
                    time = np.asarray(time)[mask]
                meta["dropped_rows"] = dropped
                meta["missing_policy"] = "listwise"
        else:
            if not np.all(np.isfinite(y_arr)):
                raise ValueError("Missing values in y; use drop_missing=True.")
            if not np.all(np.isfinite(x_endog_arr)):
                raise ValueError("Missing values in X_endog; use drop_missing=True.")
            if not np.all(np.isfinite(z_arr)):
                raise ValueError("Missing values in Z; use drop_missing=True.")
            if x_exog_arr is not None and not np.all(np.isfinite(x_exog_arr)):
                raise ValueError("Missing values in X_exog; use drop_missing=True.")

        return cls(
            y=y_arr,
            x_endog=x_endog_arr,
            x_exog=x_exog_arr,
            z=z_arr,
            weights=None if weights is None else np.asarray(weights, dtype=float),
            clusters=None if clusters is None else np.asarray(clusters),
            time=None if time is None else np.asarray(time, dtype=float),
            metadata=meta,
        )
