from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..results import ConfidenceSetResult, TestResult

WeakIVTestResult = TestResult
ARTestResult = TestResult
LMTestResult = TestResult
CLRTestResult = TestResult
ARConfidenceSetResult = ConfidenceSetResult
LMConfidenceSetResult = ConfidenceSetResult
CLRConfidenceSetResult = ConfidenceSetResult


@dataclass(frozen=True)
class GridDiagnostics:
    grid: np.ndarray
    pvalues: np.ndarray
    alpha: float
    beta_bounds: tuple[float, float]
    n_grid: int
    evaluations: int
    runtime: float
    extra: dict[str, Any] | None = None


__all__ = [
    "ARConfidenceSetResult",
    "ARTestResult",
    "CLRConfidenceSetResult",
    "CLRTestResult",
    "ConfidenceSetResult",
    "GridDiagnostics",
    "LMConfidenceSetResult",
    "LMTestResult",
    "WeakIVTestResult",
]
