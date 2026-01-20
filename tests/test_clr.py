from ivrobust import clr_test, weak_iv_dgp


def test_clr_method_flag() -> None:
    data, beta_true = weak_iv_dgp(n=180, k=3, strength=0.6, beta=1.2, seed=3)
    res = clr_test(data, beta0=beta_true, method="CLR", cov_type="HC1")
    assert res.method == "CLR"
    assert 0.0 <= res.pvalue <= 1.0
