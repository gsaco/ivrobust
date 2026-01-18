import numpy as np
import pytest

from ivrobust import IVData


def test_ivdata_accepts_1d_inputs_and_clusters() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0])
    d = np.array([1.5, 2.5, 3.5, 4.5])
    x = np.ones((4, 1))
    z = np.array([0.1, -0.2, 0.3, 0.4])
    clusters = np.array([0, 0, 1, 1])

    data = IVData(y=y, d=d, x=x, z=z, clusters=clusters)
    assert data.y.shape == (4, 1)
    assert data.d.shape == (4, 1)
    assert data.z.shape == (4, 1)
    assert data.clusters is not None
    assert data.nobs == 4

    data2 = data.with_clusters(np.array([1, 1, 2, 2]))
    assert data2.clusters is not None

    data_dict = data.as_dict()
    assert data_dict["y"].shape == (4, 1)


def test_ivdata_rejects_multicolumn_y() -> None:
    y = np.ones((4, 2))
    d = np.ones((4, 1))
    x = np.ones((4, 1))
    z = np.ones((4, 1))
    with pytest.raises(ValueError, match="single column"):
        IVData(y=y, d=d, x=x, z=z)


def test_ivdata_rejects_invalid_clusters() -> None:
    y = np.ones((4, 1))
    d = np.ones((4, 1))
    x = np.ones((4, 1))
    z = np.ones((4, 1))

    with pytest.raises(ValueError, match="clusters must have length"):
        IVData(y=y, d=d, x=x, z=z, clusters=np.array([0, 1]))

    with pytest.raises(ValueError, match="contains NaN"):
        IVData(y=y, d=d, x=x, z=z, clusters=np.array([0.0, 1.0, np.nan, 2.0]))

    with pytest.raises(ValueError, match="must be 1D"):
        IVData(y=y, d=d, x=x, z=z, clusters=np.array([[0, 1, 2, 3]]))

    with pytest.raises(ValueError, match="non-empty"):
        IVData(y=y, d=d, x=x, z=z, clusters=np.array([]))


def test_ivdata_rejects_empty_arrays() -> None:
    y = np.array([])
    d = np.array([])
    x = np.array([])
    z = np.array([])
    with pytest.raises(ValueError, match="non-empty"):
        IVData(y=y, d=d, x=x, z=z)


def test_ivdata_rejects_bad_dimensions_and_columns() -> None:
    y = np.ones((4, 1, 1))
    d = np.ones((4, 1))
    x = np.ones((4, 1))
    z = np.ones((4, 1))
    with pytest.raises(ValueError, match="1D or 2D"):
        IVData(y=y, d=d, x=x, z=z)

    y2 = np.ones((4, 1))
    d_bad = np.empty((4, 0))
    with pytest.raises(ValueError, match="non-empty"):
        IVData(y=y2, d=d_bad, x=x, z=z)

    z_bad = np.empty((4, 0))
    with pytest.raises(ValueError, match="non-empty"):
        IVData(y=y2, d=d, x=x, z=z_bad)

    x_bad = np.empty((4, 0))
    with pytest.raises(ValueError, match="non-empty"):
        IVData(y=y2, d=d, x=x_bad, z=z)
