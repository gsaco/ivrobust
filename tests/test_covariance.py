import numpy as np
import pytest

from ivrobust.cov import moment_covariance
from ivrobust.utils.warnings import ClusterWarning


def test_hc0_matches_manual():
    z = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    resid = np.array([1.0, 2.0, 3.0])
    omega, _ = moment_covariance(z, resid, kind="HC0")
    ze = z * resid[:, None]
    expected = (ze.T @ ze) / z.shape[0]
    assert np.allclose(omega, expected)


def test_cluster_covariance_warns_for_few_clusters():
    z = np.ones((4, 1))
    resid = np.array([1.0, -1.0, 2.0, -2.0])
    clusters = np.array([0, 0, 1, 1])
    with pytest.warns(ClusterWarning):
        omega, _ = moment_covariance(z, resid, clusters=clusters)
    assert omega.shape == (1, 1)
