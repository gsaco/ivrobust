from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import warnings

import numpy as np

from ._typing import FloatArray, IntArray


CovType = Literal["unadjusted", "HC0", "HC1", "HC2", "HC3", "cluster"]
CLUSTER_WARN_THRESHOLD = 30


@dataclass(frozen=True)
class CovarianceResult:
    cov: FloatArray
    cov_type: str
    df_resid: int
    nobs: int
    n_clusters: int | None = None
    warnings: tuple[str, ...] = ()


def _pinv_sym(a: FloatArray) -> FloatArray:
    # Symmetric pseudo-inverse for numerical stability.
    vals, vecs = np.linalg.eigh(a)
    tol = np.max(vals) * 1e-12 if vals.size else 0.0
    inv_vals = np.where(vals > tol, 1.0 / vals, 0.0)
    return (vecs * inv_vals) @ vecs.T


def _leverage(X: FloatArray) -> FloatArray:
    XtX_inv = _pinv_sym(X.T @ X)
    h = np.einsum("ij,jk,ik->i", X, XtX_inv, X)
    return np.clip(h, 0.0, 1.0)


def _cluster_warnings(
    clusters: IntArray, *, k: int, threshold: int = CLUSTER_WARN_THRESHOLD
) -> tuple[str, ...]:
    warnings_list: list[str] = []
    uniq, counts = np.unique(clusters, return_counts=True)
    G = int(uniq.size)
    if G < threshold:
        warnings_list.append(f"few clusters (G={G}); inference may be unreliable")
    if np.any(counts == 1):
        warnings_list.append("clusters with size 1 detected")
    if G <= k:
        warnings_list.append("cluster covariance may be rank-deficient")
    return tuple(warnings_list)


def cov_ols(
    *,
    X: FloatArray,
    resid: FloatArray,
    cov_type: CovType,
    clusters: IntArray | None = None,
) -> CovarianceResult:
    """
    Sandwich covariance for OLS coefficients.

    Parameters
    ----------
    X
        Design matrix (n x p).
    resid
        Residual vector (n x 1) or (n,).
    cov_type
        One of {"unadjusted", "HC0", "HC1", "HC2", "HC3", "cluster"}.
    clusters
        Required when cov_type="cluster". Integer cluster labels length n.

    Returns
    -------
    CovarianceResult
        Contains covariance matrix for OLS coefficients.
    """
    X2 = np.asarray(X, dtype=np.float64)
    r = np.asarray(resid, dtype=np.float64).reshape(-1, 1)

    n, p = X2.shape
    if r.shape[0] != n:
        raise ValueError("resid must have the same number of rows as X.")
    if n <= p:
        raise ValueError("Need n > p for covariance estimation.")

    bread = _pinv_sym(X2.T @ X2)
    df_resid = n - p

    warnings_list: list[str] = []

    if cov_type == "unadjusted":
        sigma2 = float(((r.T @ r) / df_resid).item())
        cov = sigma2 * bread
        return CovarianceResult(
            cov=cov, cov_type="unadjusted", df_resid=df_resid, nobs=n
        )

    if cov_type in ("HC0", "HC1", "HC2", "HC3"):
        if cov_type in ("HC2", "HC3"):
            h = _leverage(X2).reshape(-1, 1)
            scale = 1.0 - h
            scale = np.clip(scale, 1e-12, None)
            if cov_type == "HC2":
                r_use = r / np.sqrt(scale)
            else:
                r_use = r / scale
        else:
            r_use = r

        w = r_use**2
        meat = X2.T @ (X2 * w)
        cov = bread @ meat @ bread
        if cov_type == "HC1":
            cov *= n / df_resid
        return CovarianceResult(cov=cov, cov_type=cov_type, df_resid=df_resid, nobs=n)

    if cov_type == "cluster":
        if clusters is None:
            raise ValueError("clusters must be provided when cov_type='cluster'.")
        g = np.asarray(clusters)
        if g.ndim != 1 or g.shape[0] != n:
            raise ValueError("clusters must be a 1D array of length n.")

        uniq, inv = np.unique(g, return_inverse=True)
        G = int(uniq.size)
        if G < 2:
            raise ValueError("cluster covariance requires at least 2 clusters.")

        warnings_list.extend(_cluster_warnings(g, k=p))

        meat = np.zeros((p, p), dtype=np.float64)
        for gi in range(G):
            idx = inv == gi
            Xg = X2[idx, :]
            rg = r[idx, :]
            sg = Xg.T @ rg  # p x 1
            meat += sg @ sg.T

        cov = bread @ meat @ bread

        # Finite-sample correction (common in practice)
        cov *= (G / (G - 1)) * ((n - 1) / df_resid)

        if np.linalg.matrix_rank(cov) < p:
            warnings_list.append("cluster covariance is not full rank")
        for msg in warnings_list:
            warnings.warn(msg, RuntimeWarning)

        return CovarianceResult(
            cov=cov,
            cov_type="cluster",
            df_resid=df_resid,
            nobs=n,
            n_clusters=G,
            warnings=tuple(warnings_list),
        )

    raise ValueError(f"Unknown cov_type: {cov_type}")


@dataclass(frozen=True)
class MomentCovarianceResult:
    cov: FloatArray
    cov_type: str
    df_resid: int
    nobs: int
    n_clusters: int | None = None
    warnings: tuple[str, ...] = ()


def _moment_meat(
    *,
    X: FloatArray,
    resid1: FloatArray,
    resid2: FloatArray,
    cov_type: CovType,
    clusters: IntArray | None,
) -> FloatArray:
    X2 = np.asarray(X, dtype=np.float64)
    r1 = np.asarray(resid1, dtype=np.float64).reshape(-1, 1)
    r2 = np.asarray(resid2, dtype=np.float64).reshape(-1, 1)
    n, p = X2.shape
    if r1.shape[0] != n or r2.shape[0] != n:
        raise ValueError("residuals must match X rows.")

    if cov_type in ("HC2", "HC3"):
        h = _leverage(X2).reshape(-1, 1)
        scale = 1.0 - h
        scale = np.clip(scale, 1e-12, None)
        if cov_type == "HC2":
            r1 = r1 / np.sqrt(scale)
            r2 = r2 / np.sqrt(scale)
        else:
            r1 = r1 / scale
            r2 = r2 / scale

    if cov_type == "cluster":
        if clusters is None:
            raise ValueError("clusters must be provided when cov_type='cluster'.")
        g = np.asarray(clusters)
        uniq, inv = np.unique(g, return_inverse=True)
        G = int(uniq.size)
        if G < 2:
            raise ValueError("cluster covariance requires at least 2 clusters.")
        meat = np.zeros((p, p), dtype=np.float64)
        for gi in range(G):
            idx = inv == gi
            Xg = X2[idx, :]
            s1 = Xg.T @ r1[idx, :]
            s2 = Xg.T @ r2[idx, :]
            meat += s1 @ s2.T
        return meat

    w = r1 * r2
    return X2.T @ (X2 * w)


def compute_moment_cov(
    *,
    X: FloatArray,
    resid: FloatArray,
    cov_type: CovType,
    clusters: IntArray | None = None,
    small_sample_adj: bool = True,
) -> MomentCovarianceResult:
    """
    Compute sandwich covariance for moments based on regressors X and residuals.
    """
    X2 = np.asarray(X, dtype=np.float64)
    r = np.asarray(resid, dtype=np.float64).reshape(-1, 1)
    n, p = X2.shape
    if r.shape[0] != n:
        raise ValueError("resid must have the same number of rows as X.")
    if n <= p:
        raise ValueError("Need n > p for covariance estimation.")

    bread = _pinv_sym(X2.T @ X2)
    df_resid = n - p

    warnings_list: list[str] = []
    meat = _moment_meat(X=X2, resid1=r, resid2=r, cov_type=cov_type, clusters=clusters)
    cov = bread @ meat @ bread

    if cov_type == "unadjusted":
        sigma2 = float(((r.T @ r) / df_resid).item())
        cov = sigma2 * bread
    elif cov_type == "HC1" and small_sample_adj:
        cov *= n / df_resid
    elif cov_type == "cluster" and small_sample_adj:
        g = np.asarray(clusters)
        uniq, _ = np.unique(g, return_inverse=True)
        G = int(uniq.size)
        cov *= (G / (G - 1)) * ((n - 1) / df_resid)
        warnings_list.extend(_cluster_warnings(g, k=p))
        if np.linalg.matrix_rank(cov) < p:
            warnings_list.append("cluster covariance is not full rank")
        for msg in warnings_list:
            warnings.warn(msg, RuntimeWarning)

    return MomentCovarianceResult(
        cov=cov,
        cov_type=str(cov_type),
        df_resid=df_resid,
        nobs=n,
        n_clusters=None if cov_type != "cluster" else int(np.unique(clusters).size),
        warnings=tuple(warnings_list),
    )


@dataclass(frozen=True)
class ReducedFormCovarianceResult:
    cov: FloatArray
    cov_type: str
    df_resid: int
    nobs: int
    n_clusters: int | None = None
    warnings: tuple[str, ...] = ()


def cov_reduced_form(
    *,
    X: FloatArray,
    resid_y: FloatArray,
    resid_d: FloatArray,
    cov_type: CovType,
    clusters: IntArray | None = None,
    small_sample_adj: bool = True,
    df_resid_adj: int | None = None,
) -> ReducedFormCovarianceResult:
    """
    Sandwich covariance for reduced-form coefficients of (y, d) on X.
    """
    X2 = np.asarray(X, dtype=np.float64)
    n, p = X2.shape
    if n <= p:
        raise ValueError("Need n > p for covariance estimation.")

    ry = np.asarray(resid_y, dtype=np.float64).reshape(-1, 1)
    rd = np.asarray(resid_d, dtype=np.float64).reshape(-1, 1)
    if ry.shape[0] != n or rd.shape[0] != n:
        raise ValueError("residuals must have the same number of rows as X.")

    bread = _pinv_sym(X2.T @ X2)
    df_resid = n - p
    df_adj = df_resid if df_resid_adj is None else df_resid_adj

    warnings_list: list[str] = []
    meat_yy = _moment_meat(
        X=X2, resid1=ry, resid2=ry, cov_type=cov_type, clusters=clusters
    )
    meat_dd = _moment_meat(
        X=X2, resid1=rd, resid2=rd, cov_type=cov_type, clusters=clusters
    )
    meat_yd = _moment_meat(
        X=X2, resid1=ry, resid2=rd, cov_type=cov_type, clusters=clusters
    )

    V_yy = bread @ meat_yy @ bread
    V_dd = bread @ meat_dd @ bread
    V_yd = bread @ meat_yd @ bread

    if cov_type == "unadjusted":
        sigma = np.hstack([ry, rd])
        sigma_hat = (sigma.T @ sigma) / df_adj
        V_yy = sigma_hat[0, 0] * bread
        V_dd = sigma_hat[1, 1] * bread
        V_yd = sigma_hat[0, 1] * bread
    elif cov_type == "HC1" and small_sample_adj:
        V_yy *= n / df_adj
        V_dd *= n / df_adj
        V_yd *= n / df_adj
    elif cov_type == "cluster" and small_sample_adj:
        g = np.asarray(clusters)
        uniq, _ = np.unique(g, return_inverse=True)
        G = int(uniq.size)
        adj = (G / (G - 1)) * ((n - 1) / df_adj)
        V_yy *= adj
        V_dd *= adj
        V_yd *= adj
        warnings_list.extend(_cluster_warnings(g, k=p))

    V = np.block([[V_yy, V_yd], [V_yd.T, V_dd]])
    if cov_type == "cluster" and small_sample_adj:
        if np.linalg.matrix_rank(V) < 2 * p:
            warnings_list.append("cluster covariance is not full rank")
        for msg in warnings_list:
            warnings.warn(msg, RuntimeWarning)

    return ReducedFormCovarianceResult(
        cov=V,
        cov_type=str(cov_type),
        df_resid=df_resid,
        nobs=n,
        n_clusters=None if cov_type != "cluster" else int(np.unique(clusters).size),
        warnings=tuple(warnings_list),
    )
