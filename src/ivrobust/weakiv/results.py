"""Weak-IV inference result containers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ivrobust.utils.specs import (
    AssumptionSpec,
    ConfidenceSet,
    CovSpec,
    NumericalDiagnostics,
    ReproSpec,
    StrengthDiagnostics,
)


@dataclass(frozen=True)
class ARTestResult:
    """Anderson-Rubin test result."""

    statistic: float
    df: int
    pvalue: float
    alpha: Optional[float]
    confidence_set: Optional[ConfidenceSet]
    assumptions: AssumptionSpec
    cov_spec: CovSpec
    numerics: NumericalDiagnostics
    diagnostics: StrengthDiagnostics
    reproducibility: ReproSpec
    warnings: tuple[str, ...] = ()
