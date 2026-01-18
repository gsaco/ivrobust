import numpy as np
import pytest

from ivrobust import weak_iv_dgp


def test_weak_iv_dgp_validates_inputs() -> None:
    with pytest.raises(ValueError, match="n must be > 5"):
        weak_iv_dgp(n=5, k=1, strength=0.4, beta=1.0, seed=0)

    with pytest.raises(ValueError, match="k must be >= 1"):
        weak_iv_dgp(n=10, k=0, strength=0.4, beta=1.0, seed=0)

    with pytest.raises(ValueError, match="strength"):
        weak_iv_dgp(n=10, k=1, strength=0.0, beta=1.0, seed=0)

    with pytest.raises(ValueError, match="beta must be finite"):
        weak_iv_dgp(n=10, k=1, strength=0.4, beta=np.nan, seed=0)

    with pytest.raises(ValueError, match="rho must be finite"):
        weak_iv_dgp(n=10, k=1, strength=0.4, beta=1.0, seed=0, rho=1.0)
