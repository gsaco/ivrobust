import numpy as np

from ivrobust import stock_yogo_critical_values, weak_id_diagnostics, weak_iv_dgp


def test_weak_id_diagnostics_fields() -> None:
    data, _ = weak_iv_dgp(n=220, k=4, strength=0.5, beta=1.0, seed=7)
    diag = weak_id_diagnostics(data)
    assert diag.effective_f >= 0.0
    assert diag.first_stage_f >= 0.0
    assert 0.0 <= diag.partial_r2 <= 1.0
    assert diag.kp_rk_stat >= 0.0
    assert 0.0 <= diag.kp_rk_pvalue <= 1.0


def test_stock_yogo_table() -> None:
    val = stock_yogo_critical_values(1, 3, size_distortion=0.10)
    assert np.isfinite(val)
