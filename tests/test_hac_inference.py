import ivrobust as ivr


def test_hac_inference_runs() -> None:
    data, beta_true = ivr.weak_iv_dgp(n=200, k=4, strength=0.5, beta=1.1, seed=12)

    ar = ivr.ar_test(data, beta0=beta_true, cov_type="HAC", hac_lags=2)
    lm = ivr.lm_test(data, beta0=beta_true, cov_type="HAC", hac_lags=2)
    clr = ivr.clr_test(data, beta0=beta_true, cov_type="HAC", hac_lags=2)

    for res in (ar, lm, clr):
        assert 0.0 <= res.pvalue <= 1.0

    cs = ivr.ar_confidence_set(
        data, alpha=0.05, cov_type="HAC", hac_lags=2, n_grid=401
    )
    assert cs.grid_info["hac_lags"] == 2

    res = ivr.weakiv_inference(
        data,
        beta0=beta_true,
        cov_type="HAC",
        hac_lags=2,
        methods=("AR",),
        grid=(beta_true - 1.0, beta_true + 1.0, 401),
        return_grid=True,
    )
    assert "AR" in res.tests
