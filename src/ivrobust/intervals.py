from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from ._typing import FloatArray


@dataclass(frozen=True)
class IntervalSet:
    """
    A (possibly disjoint) set of intervals on the real line.

    Intervals are represented as (lower, upper), where bounds may be -inf/inf.
    """

    intervals: list[tuple[float, float]]

    def contains(self, x: float) -> bool:
        for lo, hi in self.intervals:
            if lo <= x <= hi:
                return True
        return False


def invert_pvalue_grid(
    *,
    grid: FloatArray,
    pvalues: FloatArray,
    alpha: float,
    refine: bool,
    refine_tol: float,
    max_refine_iter: int,
    pvalue_func: Callable[[float], float] | None,
) -> IntervalSet:
    """
    Invert a p-value curve on a grid into a union of intervals.
    """
    if not (0 < alpha < 1):
        raise ValueError("alpha must be in (0, 1).")

    grid1 = np.asarray(grid, dtype=np.float64).reshape(-1)
    pvals = np.asarray(pvalues, dtype=np.float64).reshape(-1)
    if grid1.size != pvals.size:
        raise ValueError("grid and pvalues must have the same length.")
    if grid1.size == 0:
        return IntervalSet(intervals=[])
    if np.any(~np.isfinite(grid1)):
        raise ValueError("grid must be finite.")
    if np.any(np.diff(grid1) <= 0):
        raise ValueError("grid must be strictly increasing.")

    inside = pvals >= alpha
    idx = np.flatnonzero(inside)
    if idx.size == 0:
        return IntervalSet(intervals=[])

    segments: list[tuple[int, int]] = []
    start = idx[0]
    prev = idx[0]
    for j in idx[1:]:
        if j == prev + 1:
            prev = j
            continue
        segments.append((start, prev))
        start = j
        prev = j
    segments.append((start, prev))

    if refine and pvalue_func is None:
        raise ValueError("pvalue_func must be provided when refine=True.")

    def bisect(a: float, b: float) -> float:
        if pvalue_func is None:
            return 0.5 * (a + b)
        fa = pvalue_func(a) - alpha
        fb = pvalue_func(b) - alpha
        if fa == 0.0:
            return a
        if fb == 0.0:
            return b
        if fa * fb > 0.0:
            return 0.5 * (a + b)

        left, right = a, b
        for _ in range(max_refine_iter):
            mid = 0.5 * (left + right)
            fm = pvalue_func(mid) - alpha
            if abs(fm) <= refine_tol or abs(right - left) <= refine_tol:
                return mid
            if fa * fm <= 0:
                right = mid
                fb = fm
            else:
                left = mid
                fa = fm
        return 0.5 * (left + right)

    refined: list[tuple[float, float]] = []
    for seg_start, seg_end in segments:
        left = float(grid1[seg_start])
        right = float(grid1[seg_end])

        unbounded_left = seg_start == 0
        unbounded_right = seg_end == grid1.size - 1

        if refine:
            if not unbounded_left:
                left = bisect(float(grid1[seg_start - 1]), left)
            if not unbounded_right:
                right = bisect(right, float(grid1[seg_end + 1]))

        refined.append(
            (-np.inf if unbounded_left else left, np.inf if unbounded_right else right)
        )

    return IntervalSet(intervals=refined)
