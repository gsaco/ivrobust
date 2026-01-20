from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from ..covariance import CovSpec, CovType
from ..data import IVData
from ..diagnostics.strength import (
    effective_f,
    first_stage_diagnostics,
    weak_id_diagnostics,
)
from ..estimators.tsls import tsls
from ..results import ConfidenceSetResult, TestResult, WeakIVInferenceResult
from .ar import ar_confidence_set, ar_test
from .clr import clr_confidence_set, clr_test
from .lm import kp_rank_test, lm_confidence_set, lm_test


def _normalize_methods(methods: Iterable[str]) -> tuple[str, ...]:
    out = []
    for m in methods:
        name = m.upper()
        if name not in {"AR", "LM", "CLR"}:
            raise ValueError(f"Unknown method: {m}")
        if name not in out:
            out.append(name)
    return tuple(out)


def weakiv_inference(
    data: IVData,
    *,
    beta0: float | Sequence[float] | None = None,
    alpha: float = 0.05,
    methods: Iterable[str] = ("AR", "LM", "CLR"),
    cov: CovSpec | str | None = None,
    cov_type: CovType = "HC1",
    clusters: np.ndarray | None = None,
    grid: np.ndarray | tuple[float, float, int] | None = None,
    return_grid: bool = False,
    recommended: str = "CLR",
) -> WeakIVInferenceResult:
    """
    Unified weak-IV robust inference workflow for AR/LM/CLR.
    """
    if clusters is not None:
        data = data.with_clusters(clusters)
    methods_use = _normalize_methods(methods)
    if beta0 is None:
        beta0 = float(tsls(data, cov_type=cov_type).beta)

    beta_bounds = None
    n_grid = 2001
    grid_array: np.ndarray | None = None
    if grid is not None:
        if isinstance(grid, tuple):
            lo, hi, n_grid = grid
            beta_bounds = (float(lo), float(hi))
        else:
            grid_array = np.asarray(grid, dtype=np.float64).reshape(-1)

    tests: dict[str, TestResult] = {}
    confidence_sets: dict[str, ConfidenceSetResult] = {}

    if "AR" in methods_use:
        tests["AR"] = ar_test(
            data, beta0=beta0, cov=cov, cov_type=cov_type, alpha=alpha
        )
        confidence_sets["AR"] = ar_confidence_set(
            data,
            alpha=alpha,
            cov=cov,
            cov_type=cov_type,
            grid=grid_array,
            beta_bounds=beta_bounds,
            n_grid=n_grid,
        )

    if "LM" in methods_use:
        tests["LM"] = lm_test(
            data, beta0=beta0, cov=cov, cov_type=cov_type, clusters=clusters, alpha=alpha
        )
        confidence_sets["LM"] = lm_confidence_set(
            data,
            alpha=alpha,
            cov=cov,
            cov_type=cov_type,
            clusters=clusters,
            grid=grid_array,
            beta_bounds=beta_bounds,
            n_grid=n_grid,
        )

    if "CLR" in methods_use:
        tests["CLR"] = clr_test(
            data, beta0=beta0, cov=cov, cov_type=cov_type, clusters=clusters
        )
        confidence_sets["CLR"] = clr_confidence_set(
            data,
            alpha=alpha,
            cov=cov,
            cov_type=cov_type,
            clusters=clusters,
            grid=grid_array,
            beta_bounds=beta_bounds,
            n_grid=n_grid,
        )

    warnings: list[str] = []
    if data.k_instr / max(data.nobs, 1) > 0.2:
        warnings.append("many instruments relative to sample size (k/n > 0.2)")

    diagnostics = {
        "first_stage": first_stage_diagnostics(data),
        "effective_f": effective_f(data, cov=cov, cov_type=cov_type, clusters=clusters),
        "weak_id": weak_id_diagnostics(
            data, cov=cov, cov_type=cov_type, clusters=clusters
        ),
        "kp_rk": kp_rank_test(data, cov=cov, cov_type=cov_type, clusters=clusters),
    }

    if not return_grid:
        for cs in confidence_sets.values():
            cs.grid_info.pop("pvalues", None)

    return WeakIVInferenceResult(
        tests=tests,
        confidence_sets=confidence_sets,
        recommended=recommended.upper(),
        alpha=alpha,
        cov_type=str(cov_type),
        diagnostics=diagnostics,
        warnings=tuple(warnings),
    )


__all__ = [
    "ar_confidence_set",
    "ar_test",
    "clr_confidence_set",
    "clr_test",
    "kp_rank_test",
    "lm_confidence_set",
    "lm_test",
    "weakiv_inference",
]
