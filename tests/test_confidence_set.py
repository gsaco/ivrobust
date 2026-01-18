from ivrobust import ar_confidence_set, weak_iv_dgp


def test_ar_confidence_set_contains_true_beta() -> None:
    data, beta_true = weak_iv_dgp(n=300, k=3, strength=0.6, beta=1.5, seed=7)

    cs = ar_confidence_set(data, alpha=0.05, cov_type="HC1")

    assert cs.confidence_set.contains(beta_true)
