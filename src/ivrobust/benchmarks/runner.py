from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from ..weakiv.ar import ar_confidence_set
from ..weakiv.clr import clr_confidence_set
from ..weakiv.lm import lm_confidence_set
from .dgp import weak_iv_dgp


def _time(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> float:
    start = time.perf_counter()
    fn(*args, **kwargs)
    return time.perf_counter() - start


def run_benchmarks() -> dict[str, float]:
    dgp = weak_iv_dgp(n=400, k=10, strength=0.5, beta=1.0, seed=0)
    data = dgp.data

    results = {}
    results["ar_grid_200"] = _time(ar_confidence_set, data, n_grid=200)
    results["ar_grid_1000"] = _time(ar_confidence_set, data, n_grid=1000)
    results["lm_grid_200"] = _time(lm_confidence_set, data, n_grid=200)
    results["clr_grid_200"] = _time(clr_confidence_set, data, n_grid=200)
    return results


if __name__ == "__main__":
    out = run_benchmarks()
    for k, v in out.items():
        print(f"{k}: {v:.4f}s")
