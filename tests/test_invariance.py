import numpy as np

import ivrobust as ivr


def test_instrument_scaling_invariance() -> None:
    data, beta_true = ivr.weak_iv_dgp(n=220, k=4, strength=0.4, beta=1.25, seed=5)

    ar_base = ivr.ar_test(data, beta0=beta_true, cov_type="HC1")
    lm_base = ivr.lm_test(data, beta0=beta_true, cov_type="HC1")
    clr_base = ivr.clr_test(data, beta0=beta_true, cov_type="HC1")

    scaled = ivr.IVData(y=data.y, d=data.d, x=data.x, z=data.z * 7.5)
    ar_scaled = ivr.ar_test(scaled, beta0=beta_true, cov_type="HC1")
    lm_scaled = ivr.lm_test(scaled, beta0=beta_true, cov_type="HC1")
    clr_scaled = ivr.clr_test(scaled, beta0=beta_true, cov_type="HC1")

    assert np.isclose(ar_base.pvalue, ar_scaled.pvalue, rtol=1e-6, atol=1e-6)
    assert np.isclose(lm_base.pvalue, lm_scaled.pvalue, rtol=1e-6, atol=1e-6)
    assert np.isclose(clr_base.pvalue, clr_scaled.pvalue, rtol=1e-6, atol=1e-6)
