import numpy as np
import statsmodels.api as sm

from ivrobust import ar_test, weak_iv_dgp


def test_ar_hc1_matches_statsmodels_wald() -> None:
    data, beta_true = weak_iv_dgp(n=250, k=4, strength=0.35, beta=1.2, seed=123)

    res = ar_test(data, beta0=beta_true, cov_type="HC1")

    y_tilde = data.y.ravel() - beta_true * data.d.ravel()
    W = np.hstack([data.x, data.z])

    ols = sm.OLS(y_tilde, W).fit(cov_type="HC1")

    k = data.z.shape[1]
    p_x = data.x.shape[1]

    R = np.zeros((k, W.shape[1]))
    R[:, p_x : p_x + k] = np.eye(k)

    wald = ols.wald_test(R, use_f=False, scalar=False)

    stat_ref = float(np.asarray(wald.statistic).ravel()[0])
    p_ref = float(np.asarray(wald.pvalue).ravel()[0])

    assert np.isclose(res.statistic, stat_ref, rtol=1e-8, atol=1e-10)
    assert np.isclose(res.pvalue, p_ref, rtol=1e-8, atol=1e-10)
