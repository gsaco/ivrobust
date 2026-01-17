import numpy as np

from ivrobust.benchmarks import weak_iv_dgp
from ivrobust.data import IVData
from ivrobust.weakiv import ar_confidence_set, ar_test


def test_ar_invariant_to_instrument_scaling():
    data, beta_true = weak_iv_dgp(n=200, k=4, strength=0.4, beta=1.0, seed=0)
    res1 = ar_test(data, beta0=[beta_true])
    scaled_data = IVData.from_arrays(
        y=data.y,
        X_endog=data.x_endog,
        X_exog=data.x_exog,
        Z=2.0 * data.z,
    )
    res2 = ar_test(scaled_data, beta0=[beta_true])
    assert np.isclose(res1.statistic, res2.statistic, rtol=1e-6, atol=1e-6)
    assert np.isclose(res1.pvalue, res2.pvalue, rtol=1e-6, atol=1e-6)


def test_ar_confidence_set_contains_true_beta():
    data, beta_true = weak_iv_dgp(n=300, k=5, strength=0.7, beta=1.5, seed=3)
    result = ar_confidence_set(data, alpha=0.1)
    cs = result.confidence_set
    assert cs is not None
    assert not cs.is_empty
    assert any(lower <= beta_true <= upper for lower, upper in cs.intervals)
