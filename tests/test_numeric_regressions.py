import numpy as np

import ivrobust as ivr


def test_near_collinear_case_stable() -> None:
    rng = np.random.default_rng(42)
    n = 200
    k = 3
    z1 = rng.standard_normal((n, 1))
    z2 = z1 + 1e-6 * rng.standard_normal((n, 1))
    z3 = rng.standard_normal((n, 1))
    Z = np.hstack([z1, z2, z3])
    X = np.ones((n, 1))

    beta = 1.0
    strength = 0.4
    pi = (strength / np.sqrt(k)) * np.ones((k, 1))

    u = rng.standard_normal((n, 1))
    v = 0.5 * u + np.sqrt(1 - 0.5**2) * rng.standard_normal((n, 1))

    D = Z @ pi + v
    Y = beta * D + u

    data = ivr.IVData(y=Y, d=D, x=X, z=Z)

    ar = ivr.ar_test(data, beta0=beta, cov_type="HC1")
    lm = ivr.lm_test(data, beta0=beta, cov_type="HC1")
    clr = ivr.clr_test(data, beta0=beta, cov_type="HC1")

    assert np.isclose(ar.statistic, 1.614205447917265, rtol=1e-6, atol=1e-6)
    assert np.isclose(lm.statistic, 0.09050846278323028, rtol=1e-6, atol=1e-6)
    assert np.isclose(clr.statistic, 0.14948700508102775, rtol=1e-6, atol=1e-6)
