import numpy as np
import pytest

from ivrobust.covariance import cov_ols


def _sample_design() -> tuple[np.ndarray, np.ndarray]:
    x = np.column_stack([np.ones(6), np.arange(6, dtype=float)])
    resid = np.array([1.0, -1.0, 2.0, -2.0, 1.5, -0.5]).reshape(-1, 1)
    return x, resid


def test_cov_ols_hc1_scales_hc0() -> None:
    x, resid = _sample_design()
    res_hc0 = cov_ols(X=x, resid=resid, cov_type="HC0")
    res_hc1 = cov_ols(X=x, resid=resid, cov_type="HC1")

    n, p = x.shape
    df = n - p
    assert np.allclose(res_hc1.cov, res_hc0.cov * n / df)


def test_cov_ols_unadjusted_shape() -> None:
    x, resid = _sample_design()
    res = cov_ols(X=x, resid=resid, cov_type="unadjusted")
    assert res.cov.shape == (2, 2)
    assert res.df_resid == x.shape[0] - x.shape[1]


def test_cov_ols_cluster() -> None:
    x, resid = _sample_design()
    clusters = np.array([0, 0, 1, 1, 2, 2])
    with pytest.warns(RuntimeWarning):
        res = cov_ols(X=x, resid=resid, cov_type="cluster", clusters=clusters)
    assert res.cov.shape == (2, 2)
    assert res.n_clusters == 3


def test_cov_ols_cluster_requires_clusters() -> None:
    x, resid = _sample_design()
    with pytest.raises(ValueError, match="clusters must be provided"):
        cov_ols(X=x, resid=resid, cov_type="cluster")


def test_cov_ols_rejects_bad_shapes() -> None:
    x, resid = _sample_design()
    with pytest.raises(ValueError, match="same number of rows"):
        cov_ols(X=x, resid=resid[:-1], cov_type="HC0")

    x_small = np.ones((2, 2))
    resid_small = np.ones((2, 1))
    with pytest.raises(ValueError, match="Need n > p"):
        cov_ols(X=x_small, resid=resid_small, cov_type="HC0")


def test_cov_ols_cluster_shape_checks() -> None:
    x, resid = _sample_design()
    clusters_bad = np.ones((x.shape[0], 1))
    with pytest.raises(ValueError, match="1D array"):
        cov_ols(X=x, resid=resid, cov_type="cluster", clusters=clusters_bad)

    clusters_single = np.zeros(x.shape[0])
    with pytest.raises(ValueError, match="at least 2 clusters"):
        cov_ols(X=x, resid=resid, cov_type="cluster", clusters=clusters_single)


def test_cov_ols_hc2_hc3_runs() -> None:
    x, resid = _sample_design()
    res_hc2 = cov_ols(X=x, resid=resid, cov_type="HC2")
    res_hc3 = cov_ols(X=x, resid=resid, cov_type="HC3")
    assert res_hc2.cov.shape == (2, 2)
    assert res_hc3.cov.shape == (2, 2)


def test_cov_ols_unknown_cov_type() -> None:
    x, resid = _sample_design()
    with pytest.raises(ValueError, match="Unknown cov_type"):
        cov_ols(X=x, resid=resid, cov_type="bad")  # type: ignore[arg-type]


def test_cov_ols_cluster_invariance() -> None:
    x, resid = _sample_design()
    clusters1 = np.array([0, 0, 1, 1, 2, 2])
    clusters2 = np.array([10, 10, 5, 5, 7, 7])
    with pytest.warns(RuntimeWarning):
        res1 = cov_ols(X=x, resid=resid, cov_type="cluster", clusters=clusters1)
    with pytest.warns(RuntimeWarning):
        res2 = cov_ols(X=x, resid=resid, cov_type="cluster", clusters=clusters2)
    assert np.allclose(res1.cov, res2.cov)
