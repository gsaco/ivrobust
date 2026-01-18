import numpy as np
import pytest

from ivrobust import IVData, ar_confidence_set, ar_test, weak_iv_dgp


def test_ar_test_rejects_multiple_endog() -> None:
    y = np.ones((6, 1))
    d = np.ones((6, 2))
    x = np.ones((6, 1))
    z = np.ones((6, 1))
    data = IVData(y=y, d=d, x=x, z=z)

    with pytest.raises(NotImplementedError, match="single endogenous"):
        ar_test(data, beta0=0.0)


def test_ar_confidence_set_validates_inputs() -> None:
    data, _ = weak_iv_dgp(n=100, k=2, strength=0.4, beta=1.0, seed=0)

    with pytest.raises(ValueError, match="alpha"):
        ar_confidence_set(data, alpha=0.0)

    with pytest.raises(ValueError, match="n_grid"):
        ar_confidence_set(data, n_grid=101)

    with pytest.raises(ValueError, match="beta_bounds"):
        ar_confidence_set(data, beta_bounds=(1.0, 1.0))


def test_ar_confidence_set_rejects_multiple_endog() -> None:
    y = np.ones((6, 1))
    d = np.ones((6, 2))
    x = np.ones((6, 1))
    z = np.ones((6, 1))
    data = IVData(y=y, d=d, x=x, z=z)

    with pytest.raises(NotImplementedError, match="single endogenous"):
        ar_confidence_set(data)
