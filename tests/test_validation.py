import numpy as np
import pytest

from ivrobust import IVData


def test_ivdata_rejects_nan() -> None:
    y = np.array([1.0, np.nan, 2.0]).reshape(-1, 1)
    d = np.ones((3, 1))
    x = np.ones((3, 1))
    z = np.ones((3, 1))
    with pytest.raises(ValueError, match="contains NaN"):
        IVData(y=y, d=d, x=x, z=z)


def test_ivdata_requires_consistent_rows() -> None:
    y = np.ones((3, 1))
    d = np.ones((4, 1))
    x = np.ones((3, 1))
    z = np.ones((3, 1))
    with pytest.raises(ValueError, match="same number of rows"):
        IVData(y=y, d=d, x=x, z=z)
