from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from .ar import ARConfidenceSetResult, ARTestResult, ar_confidence_set, ar_test
from .clr import CLRTestResult, clr_confidence_set, clr_test
from .covariance import CovSpec
from .data import IVData, weak_iv_dgp
from .diagnostics import (
    EffectiveFResult,
    FirstStageDiagnostics,
    WeakIdDiagnostics,
    cragg_donald_f,
    effective_f,
    first_stage_diagnostics,
    partial_r2,
    stock_yogo_critical_values,
    weak_id_diagnostics,
)
from .estimators import IVResults, TSLSResult, fit, fuller, kclass, liml, tsls
from .intervals import IntervalSet
from .lm import kp_lm_test, kp_rank_test, lm_confidence_set, lm_test
from .model import IVModel
from .plot_style import savefig, set_style
from .plots import plot_ar_confidence_set
from .results import ConfidenceSetResult, TestResult, WeakIVInferenceResult
from .weakiv import weakiv_inference

try:
    __version__ = version("ivrobust")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

__all__ = [
    "ARConfidenceSetResult",
    "ARTestResult",
    "CLRTestResult",
    "ConfidenceSetResult",
    "CovSpec",
    "EffectiveFResult",
    "FirstStageDiagnostics",
    "IVData",
    "IVModel",
    "IVResults",
    "IntervalSet",
    "TSLSResult",
    "TestResult",
    "WeakIVInferenceResult",
    "WeakIdDiagnostics",
    "__version__",
    "ar_confidence_set",
    "ar_test",
    "clr_confidence_set",
    "clr_test",
    "cragg_donald_f",
    "effective_f",
    "first_stage_diagnostics",
    "fit",
    "fuller",
    "kclass",
    "kp_lm_test",
    "kp_rank_test",
    "liml",
    "lm_confidence_set",
    "lm_test",
    "partial_r2",
    "plot_ar_confidence_set",
    "savefig",
    "set_style",
    "stock_yogo_critical_values",
    "tsls",
    "weak_id_diagnostics",
    "weak_iv_dgp",
    "weakiv_inference",
]
