"""Linear algebra utilities for ivrobust."""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy import linalg  # type: ignore[import-untyped]

from ivrobust.utils.specs import NumericalDiagnostics
from ivrobust.utils.warnings import NumericalWarning, WarningRecord, warn_and_record


def matrix_diagnostics(mat: NDArray[np.float64], name: str) -> NumericalDiagnostics:
    """Compute rank and conditioning diagnostics for a matrix."""

    diagnostics = NumericalDiagnostics()
    if mat.size == 0:
        return diagnostics
    u, s, vt = linalg.svd(mat, full_matrices=False, check_finite=False)
    rank = int(np.sum(s > np.finfo(float).eps * max(mat.shape)))
    diagnostics.ranks[name] = rank
    if s.size > 0:
        diagnostics.min_singular_values[name] = float(s[-1])
        if s[-1] > 0:
            diagnostics.condition_numbers[name] = float(s[0] / s[-1])
        else:
            diagnostics.condition_numbers[name] = float("inf")
    return diagnostics


def residualize(
    y: NDArray[np.float64],
    x: NDArray[np.float64],
    name: str = "x_exog",
) -> tuple[NDArray[np.float64], NumericalDiagnostics]:
    """Residualize y with respect to x using least squares."""

    diagnostics = NumericalDiagnostics()
    if x.size == 0:
        return y, diagnostics
    coef, _, rank, s = linalg.lstsq(x, y, cond=None, check_finite=False)
    diagnostics.ranks[name] = int(rank)
    if s.size > 0:
        diagnostics.min_singular_values[name] = float(s[-1])
        if s[-1] > 0:
            diagnostics.condition_numbers[name] = float(s[0] / s[-1])
        else:
            diagnostics.condition_numbers[name] = float("inf")
    if rank < x.shape[1]:
        warn_and_record(
            f"{name} is rank deficient (rank {rank} < {x.shape[1]}).",
            category=NumericalWarning,
            record=None,
        )
        diagnostics.notes.append(f"{name} rank deficient: {rank} < {x.shape[1]}.")
    residual = y - x @ coef
    return residual, diagnostics


def safe_solve(
    a: NDArray[np.float64],
    b: NDArray[np.float64],
    *,
    context: str,
    rcond: Optional[float] = None,
    warnings: Optional[WarningRecord] = None,
) -> tuple[NDArray[np.float64], NumericalDiagnostics]:
    """Solve a linear system with fallbacks and diagnostics."""

    diagnostics = NumericalDiagnostics()
    a_sym = np.allclose(a, a.T, atol=1e-10)
    try:
        if a_sym:
            cho = linalg.cho_factor(a, lower=True, check_finite=False)
            x = linalg.cho_solve(cho, b, check_finite=False)
        else:
            x = linalg.solve(a, b, assume_a="gen", check_finite=False)
        return x, diagnostics
    except linalg.LinAlgError:
        try:
            pinv = linalg.pinv(a, rcond=rcond, check_finite=False)
        except TypeError:
            pinv = np.linalg.pinv(a, rcond=rcond)
        diagnostics.used_pinv = True
        diagnostics.pinv_rcond = rcond
        diagnostics.notes.append(f"Used pseudoinverse for {context}.")
        warn_and_record(
            f"Used pseudoinverse for {context}; results may be unstable.",
            category=NumericalWarning,
            record=warnings,
        )
        return pinv @ b, diagnostics


def quadratic_form_inv(
    vec: NDArray[np.float64],
    mat: NDArray[np.float64],
    *,
    context: str,
    rcond: Optional[float] = None,
    warnings: Optional[WarningRecord] = None,
) -> tuple[float, NumericalDiagnostics]:
    """Compute vec' mat^{-1} vec with diagnostics."""

    vec = np.atleast_2d(vec)
    if vec.shape[0] == 1:
        vec = vec.T
    sol, diagnostics = safe_solve(
        mat, vec, context=context, rcond=rcond, warnings=warnings
    )
    stat = float((vec.T @ sol).item())
    return stat, diagnostics
