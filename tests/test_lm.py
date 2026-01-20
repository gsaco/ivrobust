
from ivrobust import kp_rank_test, lm_test, weak_iv_dgp


def test_kp_rank_stat_runs() -> None:
    data, _ = weak_iv_dgp(n=150, k=3, strength=0.4, beta=1.0, seed=5)
    res = kp_rank_test(data)
    assert res.statistic >= 0.0
    assert 0.0 <= res.pvalue <= 1.0
    assert res.df == data.k_instr


def test_lm_test_nonnegative() -> None:
    data, beta_true = weak_iv_dgp(n=200, k=3, strength=0.8, beta=1.1, seed=2)
    res = lm_test(data, beta0=beta_true, cov_type="HC1")
    assert res.statistic >= 0.0
    assert 0.0 <= res.pvalue <= 1.0
