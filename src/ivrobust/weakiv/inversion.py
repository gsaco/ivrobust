from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .._typing import FloatArray
from ..intervals import IntervalSet, invert_pvalue_grid


@dataclass(frozen=True)
class GridSpec:
    grid: FloatArray | None = None
    beta_bounds: tuple[float, float] | None = None
    n_grid: int = 2001


@dataclass(frozen=True)
class InversionSpec:
    refine: bool = True
    refine_tol: float = 1e-6
    max_refine_iter: int = 80
    hysteresis: float = 1e-12


def invert_test(
    *,
    test_fn: Callable[[float], float],
    alpha: float,
    grid_spec: GridSpec,
    inversion_spec: InversionSpec,
) -> tuple[IntervalSet, dict[str, object]]:
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1).")

    if grid_spec.grid is None:
        if grid_spec.beta_bounds is None:
            raise ValueError("beta_bounds must be provided when grid is None.")
        lo, hi = map(float, grid_spec.beta_bounds)
        if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
            raise ValueError("beta_bounds must be finite with lo < hi.")
        if grid_spec.n_grid < 301:
            raise ValueError("n_grid must be at least 301 for stable inversion.")
        grid = np.linspace(lo, hi, grid_spec.n_grid, dtype=np.float64)
    else:
        grid = np.asarray(grid_spec.grid, dtype=np.float64).reshape(-1)
        if grid.size < 3:
            raise ValueError("grid must contain at least 3 points.")
        lo, hi = float(grid[0]), float(grid[-1])

    start = time.perf_counter()
    pvals = np.empty_like(grid)
    for i, b0 in enumerate(grid):
        pvals[i] = float(test_fn(float(b0)))
    runtime = time.perf_counter() - start

    cs = invert_pvalue_grid(
        grid=grid,
        pvalues=pvals,
        alpha=alpha,
        refine=inversion_spec.refine,
        refine_tol=inversion_spec.refine_tol,
        max_refine_iter=inversion_spec.max_refine_iter,
        pvalue_func=(lambda b: float(test_fn(float(b)))),
    )

    grid_info = {
        "grid": grid,
        "pvalues": pvals,
        "beta_bounds": (lo, hi),
        "n_grid": int(grid.size),
        "evaluations": int(grid.size),
        "runtime": float(runtime),
        "alpha": float(alpha),
    }
    return cs, grid_info
