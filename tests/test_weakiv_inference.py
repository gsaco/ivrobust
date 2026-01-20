from ivrobust import weak_iv_dgp, weakiv_inference


def test_weakiv_inference_returns_results() -> None:
    data, beta_true = weak_iv_dgp(n=250, k=3, strength=0.6, beta=1.0, seed=3)

    res = weakiv_inference(
        data,
        beta0=beta_true,
        alpha=0.1,
        methods=("AR", "LM"),
        cov_type="HC1",
    )

    assert "AR" in res.tests
    assert "LM" in res.tests
    assert "AR" in res.confidence_sets
    assert res.confidence_sets["AR"].confidence_set.contains(beta_true)
