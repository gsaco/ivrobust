"""Covariance estimators for moment vectors."""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray

from ivrobust.data.clusters import ClusterSpec
from ivrobust.utils.specs import CovSpec
from ivrobust.utils.warnings import (
    ClusterWarning,
    CovarianceWarning,
    WarningRecord,
    warn_and_record,
)


def moment_covariance(
    z: NDArray[np.float64],
    residuals: NDArray[np.float64],
    *,
    kind: str = "HC1",
    clusters: Optional[np.ndarray] = None,
    df_adjust: bool = True,
    warnings: Optional[WarningRecord] = None,
) -> tuple[NDArray[np.float64], CovSpec]:
    """Estimate covariance of sqrt(n) * g, where g = Z' e / sqrt(n)."""

    if warnings is None:
        warnings = WarningRecord()
    n, k = z.shape
    if residuals.ndim != 1:
        residuals = residuals.ravel()
    ze = z * residuals[:, None]

    if clusters is not None:
        spec = ClusterSpec.from_array(clusters)
        if spec.n_clusters < 30:
            message = (
                f"Only {spec.n_clusters} clusters; "
                "cluster-robust inference may be unreliable."
            )
            warn_and_record(
                message,
                category=ClusterWarning,
                record=warnings,
            )
        omega = np.zeros((k, k), dtype=float)
        for g in range(spec.n_clusters):
            idx = spec.codes == g
            s_g = ze[idx].sum(axis=0)[:, None]
            omega += s_g @ s_g.T
        omega = omega / n
        if df_adjust:
            if spec.n_clusters > 1 and n > k:
                omega *= (spec.n_clusters / (spec.n_clusters - 1)) * ((n - 1) / (n - k))
            else:
                warn_and_record(
                    "Cluster df adjustment skipped due to small sample size.",
                    category=CovarianceWarning,
                    record=warnings,
                )
        omega = 0.5 * (omega + omega.T)
        cov_spec = CovSpec(kind="cluster", df_adjust=df_adjust, cluster_ids=spec.ids)
        return omega, cov_spec

    kind = kind.upper()
    if kind in {"HOMOSKEDASTIC", "CLASSICAL"}:
        if df_adjust and n > k:
            sigma2 = float(residuals @ residuals) / (n - k)
        else:
            sigma2 = float(residuals @ residuals) / n
        omega = sigma2 * (z.T @ z) / n
    else:
        omega = (ze.T @ ze) / n
        if kind == "HC0":
            pass
        elif kind == "HC1":
            if df_adjust and n > k:
                omega *= n / (n - k)
            elif df_adjust:
                warn_and_record(
                    "HC1 df adjustment skipped due to n <= k.",
                    category=CovarianceWarning,
                    record=warnings,
                )
        else:
            raise ValueError(f"Unknown covariance kind: {kind}.")
    omega = 0.5 * (omega + omega.T)
    cov_spec = CovSpec(kind=kind, df_adjust=df_adjust, cluster_ids=None)
    if np.linalg.matrix_rank(omega) < k:
        warn_and_record(
            "Moment covariance is rank deficient; inference may be unstable.",
            category=CovarianceWarning,
            record=warnings,
        )
    return omega, cov_spec
