from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from .ar import ARConfidenceSetResult, ARTestResult, ar_confidence_set, ar_test
from .clr import clr_confidence_set, clr_test
from .data import IVData, weak_iv_dgp
from .diagnostics import (
    EffectiveFResult,
    FirstStageDiagnostics,
    effective_f,
    first_stage_diagnostics,
)
from .estimators import TSLSResult, fuller, liml, tsls
from .intervals import IntervalSet
from .model import IVModel, IVResults
from .plot_style import savefig, set_style
from .plots import plot_ar_confidence_set
from .results import ConfidenceSetResult, TestResult, WeakIVInferenceResult
from .weakiv import weakiv_inference
from .lm import lm_confidence_set, lm_test

try:
    __version__ = version("ivrobust")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

__all__ = [
    "__version__",
    # Data model
    "IVData",
    "weak_iv_dgp",
    "IVModel",
    "IVResults",
    # Inference
    "ARTestResult",
    "ARConfidenceSetResult",
    "TestResult",
    "ConfidenceSetResult",
    "ar_test",
    "ar_confidence_set",
    "lm_test",
    "lm_confidence_set",
    "clr_test",
    "clr_confidence_set",
    "weakiv_inference",
    # Estimation / diagnostics
    "TSLSResult",
    "tsls",
    "liml",
    "fuller",
    "FirstStageDiagnostics",
    "first_stage_diagnostics",
    "EffectiveFResult",
    "effective_f",
    # Intervals
    "IntervalSet",
    # Plotting
    "set_style",
    "savefig",
    "plot_ar_confidence_set",
    # Results
    "WeakIVInferenceResult",
]
