import numpy as np

from ivrobust import fuller, kclass, liml, weak_iv_dgp


def test_liml_fuller_run() -> None:
    data, _ = weak_iv_dgp(n=300, k=3, strength=0.7, beta=1.0, seed=6)
    res_liml = liml(data, cov_type="HC1")
    res_fuller = fuller(data, alpha=1.0, cov_type="HC1")

    assert np.isfinite(res_liml.beta)
    assert np.isfinite(res_fuller.beta)


def test_kclass_runs() -> None:
    data, _ = weak_iv_dgp(n=250, k=2, strength=0.6, beta=0.8, seed=9)
    res = kclass(data, kappa=1.0, cov_type="HC1")
    assert np.isfinite(res.beta)
