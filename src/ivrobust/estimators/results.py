"""Estimator result containers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ivrobust.utils.specs import (
    AssumptionSpec,
    CovSpec,
    NumericalDiagnostics,
    StrengthDiagnostics,
)


@dataclass(frozen=True)
class EstimatorResult:
    """Generic estimator result container."""

    params: np.ndarray
    cov: Optional[np.ndarray]
    std_errors: Optional[np.ndarray]
    residuals: np.ndarray
    fitted: np.ndarray
    method: str
    cov_spec: Optional[CovSpec]
    assumptions: AssumptionSpec
    diagnostics: StrengthDiagnostics
    numerics: NumericalDiagnostics
    warnings: tuple[str, ...] = ()
