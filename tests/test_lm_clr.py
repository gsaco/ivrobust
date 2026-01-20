import numpy as np
import pytest

from ivrobust import clr_test, lm_test, weak_iv_dgp


def test_lm_matches_ivmodels_unadjusted() -> None:
    iv_lm = pytest.importorskip("ivmodels.tests.lagrange_multiplier")

    data, beta_true = weak_iv_dgp(n=300, k=4, strength=0.6, beta=1.2, seed=12)
    beta0 = beta_true

    res = lm_test(data, beta0=beta0, cov_type="unadjusted")
    stat, pval = res.statistic, res.pvalue

    stat_ref, p_ref = iv_lm.lagrange_multiplier_test(
        Z=data.z,
        X=data.d,
        y=data.y.ravel(),
        beta=np.array([beta0]),
        C=data.x,
        fit_intercept=False,
    )

    assert np.isclose(stat, stat_ref, rtol=1e-4, atol=1e-6)
    assert np.isclose(pval, p_ref, rtol=1e-4, atol=1e-6)


def test_clr_matches_ivmodels_unadjusted() -> None:
    iv_clr = pytest.importorskip("ivmodels.tests.conditional_likelihood_ratio")

    data, beta_true = weak_iv_dgp(n=280, k=3, strength=0.5, beta=1.0, seed=5)
    beta0 = beta_true

    res = clr_test(data, beta0=beta0, cov_type="unadjusted")

    stat_ref, p_ref = iv_clr.conditional_likelihood_ratio_test(
        Z=data.z,
        X=data.d,
        y=data.y.ravel(),
        beta=np.array([beta0]),
        C=data.x,
        fit_intercept=False,
        critical_values="moreira2003conditional",
        tol=1e-6,
    )

    assert np.isclose(res.statistic, stat_ref, rtol=2e-3, atol=1e-4)
    assert np.isclose(res.pvalue, p_ref, rtol=2e-3, atol=1e-4)


def test_lm_clr_confidence_sets_contain_true_beta() -> None:
    from ivrobust import clr_confidence_set, lm_confidence_set

    data, beta_true = weak_iv_dgp(n=320, k=4, strength=0.7, beta=1.3, seed=9)

    lm_cs = lm_confidence_set(data, alpha=0.05, cov_type="HC1")
    clr_cs = clr_confidence_set(data, alpha=0.05, cov_type="HC1")

    assert lm_cs.confidence_set.contains(beta_true)
    assert clr_cs.confidence_set.contains(beta_true)
