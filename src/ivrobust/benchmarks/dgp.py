from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..data import IVData


@dataclass(frozen=True)
class BenchmarkDGP:
    data: IVData
    beta_true: float


def weak_iv_dgp(*, n: int, k: int, strength: float, beta: float, seed: int) -> BenchmarkDGP:
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(size=(n, k))
    x = np.ones((n, 1), dtype=np.float64)
    pi = (strength / np.sqrt(k)) * np.ones((k, 1), dtype=np.float64)

    u = rng.standard_normal(size=(n, 1))
    v = 0.5 * u + np.sqrt(1.0 - 0.5**2) * rng.standard_normal(size=(n, 1))
    d = z @ pi + v
    y = beta * d + u
    return BenchmarkDGP(data=IVData(y=y, d=d, x=x, z=z), beta_true=beta)
