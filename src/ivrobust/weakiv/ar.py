"""Anderson-Rubin tests and confidence sets."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import optimize, stats

from ivrobust import __version__
from ivrobust.cov import moment_covariance
from ivrobust.data.design import IVDesign
from ivrobust.data.ivdata import IVData
from ivrobust.diagnostics import first_stage_diagnostics
from ivrobust.linalg import quadratic_form_inv
from ivrobust.utils.specs import (
    AssumptionSpec,
    ConfidenceSet,
    NumericalDiagnostics,
    ReproSpec,
)
from ivrobust.utils.warnings import NumericalWarning, WarningRecord, warn_and_record

from .results import ARTestResult


def _ensure_beta(beta0: ArrayLike, p: int) -> NDArray[np.float64]:
    beta = np.asarray(beta0, dtype=float).reshape(-1)
    if beta.shape[0] != p:
        raise ValueError(f"beta0 must have length {p}, got {beta.shape[0]}.")
    return cast(NDArray[np.float64], beta)


def _ar_statistic(
    design: IVDesign,
    beta0: np.ndarray,
    *,
    cov: str,
    clusters: Optional[np.ndarray],
    df_adjust: bool,
    warnings: WarningRecord,
) -> tuple[float, NumericalDiagnostics]:
    n = design.data.n
    residuals = design.y_resid - design.x_endog_resid @ beta0
    g = (design.z_resid.T @ residuals) / np.sqrt(n)
    omega, _ = moment_covariance(
        design.z_resid,
        residuals,
        kind=cov,
        clusters=clusters,
        df_adjust=df_adjust,
        warnings=warnings,
    )
    stat, numerics = quadratic_form_inv(
        g,
        omega,
        context="AR statistic",
        warnings=warnings,
    )
    return stat, numerics


def ar_test(
    data: IVData,
    beta0: ArrayLike,
    *,
    cov: str = "HC1",
    clusters: Optional[np.ndarray] = None,
    df_adjust: bool = True,
) -> ARTestResult:
    """Compute the Anderson-Rubin test statistic and p-value."""

    warnings = WarningRecord()
    design = IVDesign.from_data(data)
    beta = _ensure_beta(beta0, data.p)

    stat, numerics = _ar_statistic(
        design,
        beta,
        cov=cov,
        clusters=clusters if clusters is not None else data.clusters,
        df_adjust=df_adjust,
        warnings=warnings,
    )

    df = data.k
    pvalue = float(1.0 - stats.chi2.cdf(stat, df=df))

    error = (
        "cluster"
        if (clusters is not None or data.clusters is not None)
        else "heteroskedastic"
    )
    assumptions = AssumptionSpec(
        id_regime="weak_fixed_k",
        error=error,
        asymptotics="fixed_k",
        notes=(
            "AR is weak-IV robust under instrument exogeneity.",
            "Cluster robustness assumes many clusters.",
        ),
        citations=("Anderson & Rubin (1949)", "Andrews, Stock & Sun (2019)"),
    )

    diagnostics = first_stage_diagnostics(data, design=design)

    cov_spec = moment_covariance(
        design.z_resid,
        design.y_resid - design.x_endog_resid @ beta,
        kind=cov,
        clusters=clusters if clusters is not None else data.clusters,
        df_adjust=df_adjust,
        warnings=warnings,
    )[1]

    reproducibility = ReproSpec(seed=None, version=__version__, config=("ar_test",))

    numerics.merge(design.numerics)

    return ARTestResult(
        statistic=float(stat),
        df=df,
        pvalue=pvalue,
        alpha=None,
        confidence_set=None,
        assumptions=assumptions,
        cov_spec=cov_spec,
        numerics=numerics,
        diagnostics=diagnostics,
        reproducibility=reproducibility,
        warnings=tuple(warnings.messages),
    )


def _build_grid(
    *,
    center: float,
    scale: float,
    grid_size: int,
    expand: float,
) -> NDArray[np.float64]:
    half_width = max(scale * expand, 1.0)
    grid = np.linspace(center - half_width, center + half_width, grid_size)
    return cast(NDArray[np.float64], grid)


def _find_segments(accept: Sequence[bool]) -> Sequence[tuple[int, int]]:
    segments = []
    idx = 0
    n = len(accept)
    while idx < n:
        if accept[idx]:
            start = idx
            while idx + 1 < n and accept[idx + 1]:
                idx += 1
            segments.append((start, idx))
        idx += 1
    return segments


def _root_find_boundary(
    func,
    left: float,
    right: float,
    warnings: WarningRecord,
) -> float:
    try:
        return float(optimize.brentq(func, left, right))
    except ValueError:
        warn_and_record(
            "Root finding failed; using grid boundary.",
            category=NumericalWarning,
            record=warnings,
        )
        return float(right)


def ar_confidence_set(
    data: IVData,
    *,
    alpha: float = 0.05,
    grid: Optional[ArrayLike] = None,
    cov: str = "HC1",
    clusters: Optional[np.ndarray] = None,
    df_adjust: bool = True,
    grid_size: int = 401,
    max_expansions: int = 4,
    expand_factor: float = 2.0,
) -> ARTestResult:
    """Invert the AR test to obtain a set-valued confidence set (p=1)."""

    if data.p != 1:
        raise NotImplementedError("AR confidence sets are only implemented for p=1.")

    warnings = WarningRecord()
    design = IVDesign.from_data(data)
    diagnostics = first_stage_diagnostics(data, design=design)

    if grid is None:
        from ivrobust.estimators import tsls

        tsls_res = tsls(
            data,
            cov=cov,
            clusters=clusters,
            df_adjust=df_adjust,
        )
        center = float(tsls_res.params[0])
        if tsls_res.std_errors is not None and np.isfinite(tsls_res.std_errors[0]):
            scale = float(tsls_res.std_errors[0])
        else:
            std_x = float(np.std(design.x_endog_resid))
            std_y = float(np.std(design.y_resid))
            if std_x > 0 and np.isfinite(std_x):
                scale = std_y / std_x
            else:
                scale = 1.0
            warn_and_record(
                "Using fallback scale for AR grid; TSLS standard error unavailable.",
                category=NumericalWarning,
                record=warnings,
            )
        grid_arr = _build_grid(
            center=center,
            scale=scale,
            grid_size=grid_size,
            expand=8.0,
        )
    else:
        grid_arr = np.asarray(grid, dtype=float)
        if grid_arr.ndim != 1:
            raise ValueError("grid must be a 1D array.")
        grid_arr = np.sort(grid_arr)

    df = data.k
    crit = float(stats.chi2.ppf(1.0 - alpha, df=df))

    numerics = NumericalDiagnostics()
    expansions = 0
    unbounded_left = False
    unbounded_right = False

    for expansion in range(max_expansions + 1):
        stats_grid = np.empty_like(grid_arr)
        for idx, beta0 in enumerate(grid_arr):
            stat, diag = _ar_statistic(
                design,
                np.array([beta0]),
                cov=cov,
                clusters=clusters if clusters is not None else data.clusters,
                df_adjust=df_adjust,
                warnings=warnings,
            )
            stats_grid[idx] = stat
            numerics.merge(diag)
        accept = stats_grid <= crit
        if accept[0]:
            if expansion == max_expansions:
                unbounded_left = True
            else:
                grid_arr = _build_grid(
                    center=float(grid_arr.mean()),
                    scale=float((grid_arr[-1] - grid_arr[0]) / 2),
                    grid_size=grid_size,
                    expand=expand_factor,
                )
                expansions += 1
                continue
        if accept[-1]:
            if expansion == max_expansions:
                unbounded_right = True
            else:
                grid_arr = _build_grid(
                    center=float(grid_arr.mean()),
                    scale=float((grid_arr[-1] - grid_arr[0]) / 2),
                    grid_size=grid_size,
                    expand=expand_factor,
                )
                expansions += 1
                continue
        break

    segments = _find_segments(accept)
    intervals = []
    for start, end in segments:
        if start == 0:
            left = float("-inf") if unbounded_left else float(grid_arr[start])
        else:
            left = _root_find_boundary(
                lambda b: _ar_statistic(
                    design,
                    np.array([b]),
                    cov=cov,
                    clusters=clusters if clusters is not None else data.clusters,
                    df_adjust=df_adjust,
                    warnings=warnings,
                )[0]
                - crit,
                float(grid_arr[start - 1]),
                float(grid_arr[start]),
                warnings,
            )
        if end == len(grid_arr) - 1:
            right = float("inf") if unbounded_right else float(grid_arr[end])
        else:
            right = _root_find_boundary(
                lambda b: _ar_statistic(
                    design,
                    np.array([b]),
                    cov=cov,
                    clusters=clusters if clusters is not None else data.clusters,
                    df_adjust=df_adjust,
                    warnings=warnings,
                )[0]
                - crit,
                float(grid_arr[end]),
                float(grid_arr[end + 1]),
                warnings,
            )
        intervals.append((left, right))

    if len(intervals) == 0:
        warn_and_record(
            "AR confidence set is empty on the evaluated grid.",
            category=NumericalWarning,
            record=warnings,
        )

    cs = ConfidenceSet(
        intervals=intervals,
        alpha=alpha,
        method="AR",
        grid=grid_arr,
        diagnostics={"expansions": float(expansions)},
        warnings=tuple(warnings.messages),
    )

    error = (
        "cluster"
        if (clusters is not None or data.clusters is not None)
        else "heteroskedastic"
    )
    assumptions = AssumptionSpec(
        id_regime="weak_fixed_k",
        error=error,
        asymptotics="fixed_k",
        notes=("AR confidence sets can be disjoint or unbounded under weak ID.",),
        citations=("Anderson & Rubin (1949)", "Mikusheva (2010)"),
    )

    cov_spec = moment_covariance(
        design.z_resid,
        design.y_resid - design.x_endog_resid @ np.array([0.0]),
        kind=cov,
        clusters=clusters if clusters is not None else data.clusters,
        df_adjust=df_adjust,
        warnings=warnings,
    )[1]

    reproducibility = ReproSpec(
        seed=None, version=__version__, config=("ar_confidence_set",)
    )

    numerics.merge(design.numerics)

    return ARTestResult(
        statistic=float("nan"),
        df=df,
        pvalue=float("nan"),
        alpha=alpha,
        confidence_set=cs,
        assumptions=assumptions,
        cov_spec=cov_spec,
        numerics=numerics,
        diagnostics=diagnostics,
        reproducibility=reproducibility,
        warnings=tuple(warnings.messages),
    )
