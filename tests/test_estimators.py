import numpy as np

from ivrobust.benchmarks import weak_iv_dgp
from ivrobust.estimators import liml, tsls


def test_tsls_recovers_beta():
    data, beta_true = weak_iv_dgp(n=500, k=5, strength=0.8, beta=2.0, seed=42)
    result = tsls(data)
    assert np.isfinite(result.params[0])
    assert abs(result.params[0] - beta_true) < 0.3
    assert result.cov is not None
    assert result.cov.shape == (1, 1)


def test_liml_returns_point_estimate():
    data, _ = weak_iv_dgp(n=300, k=5, strength=0.6, beta=1.0, seed=7)
    result = liml(data)
    assert result.params.shape == (1,)
    assert np.isfinite(result.params[0])
