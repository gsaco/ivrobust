import numpy as np
import pytest

from ivrobust import IVData, tsls, weak_iv_dgp


def test_tsls_cluster_covariance_runs() -> None:
    data, _ = weak_iv_dgp(n=240, k=3, strength=0.6, beta=1.0, seed=2)
    n_clusters = 12
    clusters = np.repeat(np.arange(n_clusters), np.ceil(data.nobs / n_clusters))[
        : data.nobs
    ]
    data = data.with_clusters(clusters)

    res = tsls(data, cov_type="cluster")
    assert res.vcov.shape[0] == res.vcov.shape[1]


def test_tsls_errors() -> None:
    y = np.ones((3, 1))
    d = np.ones((3, 2))
    x = np.ones((3, 1))
    z = np.ones((3, 1))
    data_multi = IVData(y=y, d=d, x=x, z=z)
    with pytest.raises(NotImplementedError, match="p_endog"):
        tsls(data_multi)

    y2 = np.ones((2, 1))
    d2 = np.ones((2, 1))
    x2 = np.ones((2, 1))
    z2 = np.ones((2, 1))
    data_small = IVData(y=y2, d=d2, x=x2, z=z2)
    with pytest.raises(ValueError, match="Need n > number of regressors"):
        tsls(data_small)

    data, _ = weak_iv_dgp(n=100, k=2, strength=0.5, beta=1.0, seed=1)
    with pytest.raises(ValueError, match="Cluster covariance requested"):
        tsls(data, cov_type="cluster")

    clusters_one = np.zeros(data.nobs, dtype=int)
    data_one_cluster = data.with_clusters(clusters_one)
    with pytest.raises(ValueError, match="at least 2 clusters"):
        tsls(data_one_cluster, cov_type="cluster")

    with pytest.raises(ValueError, match="Unknown cov_type"):
        tsls(data, cov_type="bad")  # type: ignore[arg-type]
