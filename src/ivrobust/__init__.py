"""ivrobust: weak-instrument robust inference for linear IV/GMM."""

from ._version import __version__
from .benchmarks import run_benchmark_grid, run_smoke_benchmark, weak_iv_dgp
from .data import ClusterSpec, IVData, IVDesign
from .diagnostics import effective_f_statistic, first_stage_diagnostics
from .estimators import EstimatorResult, liml, tsls
from .weakiv import ARTestResult, ar_confidence_set, ar_test, clr_test, lm_test

__all__ = [
    "__version__",
    "ARTestResult",
    "ClusterSpec",
    "EstimatorResult",
    "IVData",
    "IVDesign",
    "ar_confidence_set",
    "ar_test",
    "clr_test",
    "effective_f_statistic",
    "first_stage_diagnostics",
    "liml",
    "lm_test",
    "run_benchmark_grid",
    "run_smoke_benchmark",
    "tsls",
    "weak_iv_dgp",
]
