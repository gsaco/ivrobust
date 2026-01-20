import ivrobust as ivr


def test_readme_imports_and_contracts() -> None:
    assert hasattr(ivr, "ar_test")
    assert hasattr(ivr, "ar_confidence_set")
    assert hasattr(ivr, "weak_iv_dgp")

    data, _beta_true = ivr.weak_iv_dgp(n=120, k=2, strength=0.6, beta=1.1, seed=4)
    cs = ivr.ar_confidence_set(data, alpha=0.1, cov_type="HC1")

    assert hasattr(cs, "intervals")
    assert cs.intervals == cs.confidence_set.intervals

    _fig, ax = ivr.plot_ar_confidence_set(cs)
    ax.set_title("contract")


def test_ivdata_from_arrays_roundtrip() -> None:
    data, _ = ivr.weak_iv_dgp(n=80, k=2, strength=0.7, beta=0.9, seed=11)
    data2 = ivr.IVData.from_arrays(y=data.y, d=data.d, z=data.z, x=data.x)
    assert data2.nobs == data.nobs
    assert data2.k_instr == data.k_instr
