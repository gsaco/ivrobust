"""Benchmark runners for weak-IV diagnostics."""

from __future__ import annotations

from collections.abc import Iterable

from ivrobust.benchmarks.dgp import weak_iv_dgp
from ivrobust.weakiv import ar_confidence_set, ar_test


def run_smoke_benchmark(seed: int = 0) -> dict:
    """Run a small benchmark to detect regressions."""

    data, beta_true = weak_iv_dgp(n=200, k=5, strength=0.15, beta=1.0, seed=seed)
    test = ar_test(data, beta0=[beta_true])
    cs = ar_confidence_set(data, alpha=0.05)
    covers = False
    if cs.confidence_set is not None:
        for lower, upper in cs.confidence_set.intervals:
            if lower <= beta_true <= upper:
                covers = True
                break
    return {
        "seed": seed,
        "ar_stat": test.statistic,
        "ar_pvalue": test.pvalue,
        "cs_covers": covers,
        "cs_empty": cs.confidence_set.is_empty if cs.confidence_set else True,
    }


def run_benchmark_grid(
    *,
    seeds: Iterable[int],
    strengths: Iterable[float],
    n: int = 300,
    k: int = 10,
    beta: float = 1.0,
) -> list[dict]:
    """Run a grid of weak-IV benchmarks and return results as dicts."""

    results: list[dict] = []
    for seed in seeds:
        for strength in strengths:
            data, beta_true = weak_iv_dgp(
                n=n,
                k=k,
                strength=strength,
                beta=beta,
                seed=seed,
            )
            test = ar_test(data, beta0=[beta_true])
            results.append(
                {
                    "seed": seed,
                    "strength": strength,
                    "ar_stat": test.statistic,
                    "ar_pvalue": test.pvalue,
                }
            )
    return results


def results_to_dataframe(results: list[dict]):
    """Optional helper to convert results to a pandas DataFrame."""

    try:
        import pandas as pd  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("pandas is required for results_to_dataframe") from exc
    return pd.DataFrame(results)
