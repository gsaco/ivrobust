"""Utility helpers for ivrobust."""

from .arrays import as_1d_array, as_2d_array, check_finite
from .specs import (
    AssumptionSpec,
    ConfidenceSet,
    CovSpec,
    NumericalDiagnostics,
    ReproSpec,
    StrengthDiagnostics,
)
from .warnings import (
    ClusterWarning,
    CovarianceWarning,
    DataWarning,
    IVRobustWarning,
    NumericalWarning,
    WarningRecord,
    warn_and_record,
)

__all__ = [
    "AssumptionSpec",
    "ConfidenceSet",
    "CovSpec",
    "NumericalDiagnostics",
    "ReproSpec",
    "StrengthDiagnostics",
    "as_1d_array",
    "as_2d_array",
    "check_finite",
    "ClusterWarning",
    "CovarianceWarning",
    "DataWarning",
    "IVRobustWarning",
    "NumericalWarning",
    "WarningRecord",
    "warn_and_record",
]
