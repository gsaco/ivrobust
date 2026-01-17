import numpy as np
import pytest

from ivrobust.data import IVData


def test_ivdata_shapes_and_dimensions():
    y = np.arange(5, dtype=float)
    x_endog = np.ones((5, 1))
    x_exog = np.ones((5, 1))
    z = np.ones((5, 2))
    data = IVData.from_arrays(y=y, X_endog=x_endog, X_exog=x_exog, Z=z)
    assert data.n == 5
    assert data.p == 1
    assert data.q == 1
    assert data.k == 2


def test_ivdata_shape_mismatch_raises():
    y = np.arange(5, dtype=float)
    x_endog = np.ones((5, 1))
    x_exog = np.ones((5, 1))
    z = np.ones((4, 2))
    with pytest.raises(ValueError):
        IVData.from_arrays(y=y, X_endog=x_endog, X_exog=x_exog, Z=z)


def test_ivdata_drop_missing():
    y = np.array([1.0, np.nan, 2.0])
    x_endog = np.array([1.0, 2.0, 3.0])
    x_exog = np.ones((3, 1))
    z = np.ones((3, 2))
    data = IVData.from_arrays(
        y=y,
        X_endog=x_endog,
        X_exog=x_exog,
        Z=z,
        drop_missing=True,
    )
    assert data.n == 2
    assert data.metadata.get("dropped_rows") == 1


def test_ivdata_multiway_cluster_raises():
    y = np.arange(4, dtype=float)
    x_endog = np.ones((4, 1))
    x_exog = np.ones((4, 1))
    z = np.ones((4, 2))
    clusters = np.arange(8).reshape(4, 2)
    with pytest.raises(NotImplementedError):
        IVData.from_arrays(y=y, X_endog=x_endog, X_exog=x_exog, Z=z, clusters=clusters)
