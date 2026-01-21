from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

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
        return any(lo <= x <= hi for lo, hi in self.intervals)

    @property
    def is_empty(self) -> bool:
        return len(self.intervals) == 0

    @property
    def is_unbounded(self) -> bool:
        return any(
            (not (lo > float("-inf")) or not (hi < float("inf")))
            for lo, hi in self.intervals
        )

    @property
    def is_real_line(self) -> bool:
        normalized = self.normalized()
        return (
            len(normalized.intervals) == 1
            and normalized.intervals[0][0] == float("-inf")
            and normalized.intervals[0][1] == float("inf")
        )

    @property
    def is_disjoint(self) -> bool:
        return len(self.normalized().intervals) > 1

    def to_dict(self) -> dict[str, object]:
        return {
            "intervals": [(float(lo), float(hi)) for lo, hi in self.intervals],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> IntervalSet:
        intervals = payload.get("intervals", [])
        if not isinstance(intervals, list):
            raise ValueError("intervals must be a list of (lower, upper) pairs.")
        parsed: list[tuple[float, float]] = []
        for pair in intervals:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError("intervals must be a list of (lower, upper) pairs.")
            parsed.append((float(pair[0]), float(pair[1])))
        return cls(intervals=parsed)

    def normalized(self) -> IntervalSet:
        if not self.intervals:
            return IntervalSet(intervals=[])

        intervals = sorted(
            ((float(lo), float(hi)) for lo, hi in self.intervals),
            key=lambda pair: (pair[0], pair[1]),
        )
        merged: list[tuple[float, float]] = [intervals[0]]
        for lo, hi in intervals[1:]:
            prev_lo, prev_hi = merged[-1]
            if lo <= prev_hi or np.isinf(prev_hi):
                merged[-1] = (prev_lo, max(prev_hi, hi))
            else:
                merged.append((lo, hi))
        return IntervalSet(intervals=merged)

    def union(self, other: IntervalSet) -> IntervalSet:
        return IntervalSet(intervals=self.intervals + other.intervals).normalized()


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
