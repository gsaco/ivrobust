import numpy as np
import pytest

from ivrobust import IVData, effective_f, first_stage_diagnostics, weak_iv_dgp


def test_first_stage_diagnostics_runs() -> None:
    data, _ = weak_iv_dgp(n=200, k=4, strength=0.5, beta=1.0, seed=0)
    diag = first_stage_diagnostics(data)

    assert diag.k_instr == 4
    assert diag.nobs == 200
    assert diag.f_statistic >= 0.0
    assert 0.0 <= diag.partial_r2 <= 1.0


def test_first_stage_diagnostics_errors() -> None:
    y = np.ones((3, 1))
    d = np.ones((3, 2))
    x = np.ones((3, 1))
    z = np.ones((3, 1))
    data_multi = IVData(y=y, d=d, x=x, z=z)
    with pytest.raises(NotImplementedError, match="p_endog"):
        first_stage_diagnostics(data_multi)

    y2 = np.ones((3, 1))
    d2 = np.ones((3, 1))
    x2 = np.ones((3, 1))
    z2 = np.ones((3, 2))
    data_small = IVData(y=y2, d=d2, x=x2, z=z2)
    with pytest.raises(ValueError, match="Need n > number of first-stage regressors"):
        first_stage_diagnostics(data_small)


def test_effective_f_matches_first_stage_unadjusted() -> None:
    data, _ = weak_iv_dgp(n=400, k=3, strength=0.8, beta=1.0, seed=4)
    eff = effective_f(data, cov_type="unadjusted")
    diag = first_stage_diagnostics(data)

    assert np.isclose(eff.statistic, diag.f_statistic, rtol=0.05, atol=1e-3)
