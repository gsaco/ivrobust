import numpy as np
import pytest

from ivrobust.benchmarks import weak_iv_dgp
from ivrobust.weakiv import ar_test


@pytest.mark.filterwarnings("ignore:.*ivmodels.*")
def test_ar_parity_with_ivmodels():
    pytest.importorskip("ivmodels")
    from ivmodels.tests import anderson_rubin_test

    data, beta_true = weak_iv_dgp(n=300, k=5, strength=0.4, beta=1.0, seed=5)
    ours = ar_test(data, beta0=[beta_true], cov="HOMOSKEDASTIC")
    stat_iv, p_iv = anderson_rubin_test(
        Z=data.z,
        X=data.x_endog,
        y=data.y,
        beta=np.array([beta_true]),
        C=data.x_exog,
        critical_values="chi2",
        fit_intercept=False,
    )
    assert abs(ours.pvalue - p_iv) < 0.02
