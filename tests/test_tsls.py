import numpy as np

from ivrobust import tsls, weak_iv_dgp


def test_tsls_point_estimate_reasonable() -> None:
    data, beta_true = weak_iv_dgp(n=2000, k=3, strength=1.0, beta=2.0, seed=1)

    res = tsls(data, cov_type="HC1")
    # With a reasonably strong first stage and large n, TSLS should be close.
    assert np.isfinite(res.beta)
    assert abs(res.beta - beta_true) < 0.15
