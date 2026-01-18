from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from .ar import ARConfidenceSetResult, ARTestResult, ar_confidence_set, ar_test
from .data import IVData, weak_iv_dgp
from .diagnostics import FirstStageDiagnostics, first_stage_diagnostics
from .estimators import TSLSResult, tsls
from .plot_style import savefig, set_style
from .plots import plot_ar_confidence_set

try:
    __version__ = version("ivrobust")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

__all__ = [
    "__version__",
    # Data model
    "IVData",
    "weak_iv_dgp",
    # Inference
    "ARTestResult",
    "ARConfidenceSetResult",
    "ar_test",
    "ar_confidence_set",
    # Estimation / diagnostics
    "TSLSResult",
    "tsls",
    "FirstStageDiagnostics",
    "first_stage_diagnostics",
    # Plotting
    "set_style",
    "savefig",
    "plot_ar_confidence_set",
]
