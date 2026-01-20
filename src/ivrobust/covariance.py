from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, cast

import numpy as np

from ._typing import FloatArray, IntArray
from .data.clusters import ClusterSpec, combine_clusters, normalize_clusters

CovType = Literal[
    "unadjusted",
    "homoskedastic",
    "HC0",
    "HC1",
    "HC2",
    "HC3",
    "hc0",
    "hc1",
    "hc2",
    "hc3",
    "cluster",
    "HAC",
    "hac",
]
CLUSTER_WARN_THRESHOLD = 10


@dataclass(frozen=True)
class CovSpec:
    cov_type: str
    clusters: ClusterSpec | None = None
    hac_lags: int | None = None
    kernel: str = "bartlett"
    bandwidth: float | None = None
    small_sample: bool = True


@dataclass(frozen=True)
class CovarianceResult:
    cov: FloatArray
    cov_type: str
    df_resid: int
    nobs: int
    n_clusters: int | None = None
    warnings: tuple[str, ...] = ()


def _normalize_cov_type(cov_type: str) -> str:
    ct = cov_type.strip().lower()
    if ct in {"homoskedastic", "unadjusted", "ols"}:
        return "unadjusted"
    if ct in {"hc0", "hc1", "hc2", "hc3"}:
        return ct.upper()
    if ct == "cluster":
        return "cluster"
    if ct == "hac":
        return "HAC"
    raise ValueError(f"Unknown cov_type: {cov_type}")


def parse_cov_spec(
    cov: CovSpec | str | Mapping[str, object] | None,
    *,
    cov_type: CovType | None,
    clusters: IntArray | ClusterSpec | None,
    nobs: int,
    hac_lags: int | None = None,
    kernel: str | None = None,
    bandwidth: float | None = None,
    small_sample: bool = True,
) -> CovSpec:
    if cov is not None and cov_type is not None:
        cov_type = None

    if isinstance(cov, CovSpec):
        spec = cov
    elif isinstance(cov, str):
        spec = CovSpec(cov_type=_normalize_cov_type(cov))
    elif isinstance(cov, Mapping):
        ct = cov.get("cov_type", cov_type or "HC1")
        clusters = cast(IntArray | ClusterSpec | None, cov.get("clusters", clusters))
        spec = CovSpec(
            cov_type=_normalize_cov_type(str(ct)),
            hac_lags=cast(int | None, cov.get("hac_lags")),
            kernel=str(cov.get("kernel", kernel or "bartlett")),
            bandwidth=cast(float | None, cov.get("bandwidth")),
            small_sample=bool(cov.get("small_sample", small_sample)),
        )
    else:
        ct = cov_type or "HC1"
        spec = CovSpec(cov_type=_normalize_cov_type(str(ct)))

    clusters_use = clusters if spec.clusters is None else spec.clusters
    if spec.cov_type == "cluster":
        if clusters_use is None:
            raise ValueError("clusters must be provided when cov_type='cluster'.")
        if isinstance(clusters_use, ClusterSpec):
            cluster_spec = clusters_use
        else:
            cluster_spec = normalize_clusters(clusters_use, nobs=nobs)
    else:
        cluster_spec = None

    return CovSpec(
        cov_type=spec.cov_type,
        clusters=cluster_spec,
        hac_lags=spec.hac_lags if hac_lags is None else hac_lags,
        kernel=spec.kernel if kernel is None else kernel,
        bandwidth=spec.bandwidth if bandwidth is None else bandwidth,
        small_sample=spec.small_sample if small_sample is None else small_sample,
    )


def _pinv_sym(a: FloatArray) -> FloatArray:
    vals, vecs = np.linalg.eigh(a)
    tol = np.max(vals) * 1e-12 if vals.size else 0.0
    inv_vals = np.where(vals > tol, 1.0 / vals, 0.0)
    return cast(FloatArray, (vecs * inv_vals) @ vecs.T)


def _leverage(X: FloatArray) -> FloatArray:
    XtX_inv = _pinv_sym(X.T @ X)
    h = np.einsum("ij,jk,ik->i", X, XtX_inv, X)
    return cast(FloatArray, np.clip(h, 0.0, 1.0))


def _cluster_warnings(
    clusters: IntArray, *, k: int, threshold: int = CLUSTER_WARN_THRESHOLD
) -> tuple[str, ...]:
    warnings_list: list[str] = []
    uniq, counts = np.unique(clusters, return_counts=True)
    G = int(uniq.size)
    if threshold > G:
        warnings_list.append(f"few clusters (G={G}); inference may be unreliable")
    if np.any(counts == 1):
        warnings_list.append("clusters with size 1 detected")
    if k >= G:
        warnings_list.append("cluster covariance may be rank-deficient")
    return tuple(warnings_list)


def _kernel_weight(lag: int, L: int, *, kernel: str) -> float:
    if L <= 0:
        return 1.0
    k = kernel.lower()
    x = lag / (L + 1)
    if k == "bartlett":
        return 1.0 - x
    if k == "parzen":
        if x <= 0.5:
            return 1.0 - 6.0 * x * x + 6.0 * x * x * x
        return 2.0 * (1.0 - x) ** 3
    raise ValueError(f"Unknown HAC kernel: {kernel}")


def _default_hac_lags(nobs: int) -> int:
    return int(np.floor(4.0 * (nobs / 100.0) ** (2.0 / 9.0)))


def _cluster_codes(spec: ClusterSpec) -> IntArray:
    if spec.is_multiway:
        return combine_clusters(spec)
    return spec.codes[0]


def _hac_meat(
    *,
    X: FloatArray,
    resid1: FloatArray,
    resid2: FloatArray,
    lags: int,
    kernel: str,
) -> FloatArray:
    X2 = np.asarray(X, dtype=np.float64)
    r1 = np.asarray(resid1, dtype=np.float64).reshape(-1, 1)
    r2 = np.asarray(resid2, dtype=np.float64).reshape(-1, 1)
    Xu1 = X2 * r1
    Xu2 = X2 * r2
    meat = Xu1.T @ Xu2
    for lag in range(1, lags + 1):
        weight = _kernel_weight(lag, lags, kernel=kernel)
        if weight <= 0:
            continue
        gamma = Xu1[lag:].T @ Xu2[:-lag]
        meat += weight * (gamma + gamma.T)
    return cast(FloatArray, meat)


def cov_ols(
    *,
    X: FloatArray,
    resid: FloatArray,
    cov_type: CovType | None = "HC1",
    clusters: IntArray | ClusterSpec | None = None,
    cov: CovSpec | str | Mapping[str, object] | None = None,
    hac_lags: int | None = None,
    kernel: str = "bartlett",
) -> CovarianceResult:
    """
    Sandwich covariance for OLS coefficients.
    """
    X2 = np.asarray(X, dtype=np.float64)
    r = np.asarray(resid, dtype=np.float64).reshape(-1, 1)

    n, p = X2.shape
    if r.shape[0] != n:
        raise ValueError("resid must have the same number of rows as X.")
    if n <= p:
        raise ValueError("Need n > p for covariance estimation.")

    spec = parse_cov_spec(
        cov,
        cov_type=cov_type,
        clusters=clusters,
        nobs=n,
        hac_lags=hac_lags,
        kernel=kernel,
    )

    bread = _pinv_sym(X2.T @ X2)
    df_resid = n - p
    warnings_list: list[str] = []

    if spec.cov_type == "unadjusted":
        sigma2 = float(((r.T @ r) / df_resid).item())
        cov_mat = sigma2 * bread
        return CovarianceResult(
            cov=cov_mat, cov_type="unadjusted", df_resid=df_resid, nobs=n
        )

    if spec.cov_type in ("HC0", "HC1", "HC2", "HC3"):
        r_use = r
        if spec.cov_type in ("HC2", "HC3"):
            h = _leverage(X2).reshape(-1, 1)
            scale = np.clip(1.0 - h, 1e-12, None)
            r_use = r / np.sqrt(scale) if spec.cov_type == "HC2" else r / scale

        w = r_use**2
        meat = X2.T @ (X2 * w)
        cov_mat = bread @ meat @ bread
        if spec.cov_type == "HC1" and spec.small_sample:
            cov_mat *= n / df_resid
        return CovarianceResult(
            cov=cov_mat,
            cov_type=spec.cov_type,
            df_resid=df_resid,
            nobs=n,
        )

    if spec.cov_type == "HAC":
        lags = spec.hac_lags if spec.hac_lags is not None else _default_hac_lags(n)
        meat = _hac_meat(X=X2, resid1=r, resid2=r, lags=lags, kernel=spec.kernel)
        cov_mat = bread @ meat @ bread
        return CovarianceResult(
            cov=cov_mat,
            cov_type="HAC",
            df_resid=df_resid,
            nobs=n,
        )

    if spec.cov_type == "cluster":
        if spec.clusters is None:
            raise ValueError("clusters must be provided when cov_type='cluster'.")
        g = _cluster_codes(spec.clusters)
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
            sg = Xg.T @ rg
            meat += sg @ sg.T

        cov_mat = bread @ meat @ bread

        if spec.small_sample:
            cov_mat *= (G / (G - 1)) * ((n - 1) / df_resid)

        if np.linalg.matrix_rank(cov_mat) < p:
            warnings_list.append("cluster covariance is not full rank")
        for msg in warnings_list:
            warnings.warn(msg, RuntimeWarning, stacklevel=2)

        return CovarianceResult(
            cov=cov_mat,
            cov_type="cluster",
            df_resid=df_resid,
            nobs=n,
            n_clusters=G,
            warnings=tuple(warnings_list),
        )

    raise ValueError(f"Unknown cov_type: {spec.cov_type}")


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
    cov_type: str,
    clusters: ClusterSpec | None,
    hac_lags: int | None,
    kernel: str,
) -> FloatArray:
    X2 = np.asarray(X, dtype=np.float64)
    r1 = np.asarray(resid1, dtype=np.float64).reshape(-1, 1)
    r2 = np.asarray(resid2, dtype=np.float64).reshape(-1, 1)
    n, p = X2.shape
    if r1.shape[0] != n or r2.shape[0] != n:
        raise ValueError("residuals must match X rows.")

    if cov_type in ("HC2", "HC3"):
        h = _leverage(X2).reshape(-1, 1)
        scale = np.clip(1.0 - h, 1e-12, None)
        if cov_type == "HC2":
            r1 = r1 / np.sqrt(scale)
            r2 = r2 / np.sqrt(scale)
        else:
            r1 = r1 / scale
            r2 = r2 / scale

    if cov_type == "cluster":
        if clusters is None:
            raise ValueError("clusters must be provided when cov_type='cluster'.")
        g = _cluster_codes(clusters)
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
        return cast(FloatArray, meat)

    if cov_type == "HAC":
        lags = hac_lags if hac_lags is not None else _default_hac_lags(n)
        return _hac_meat(X=X2, resid1=r1, resid2=r2, lags=lags, kernel=kernel)

    w = r1 * r2
    return cast(FloatArray, X2.T @ (X2 * w))


def compute_moment_cov(
    *,
    X: FloatArray,
    resid: FloatArray,
    cov_type: CovType | None = "HC1",
    clusters: IntArray | ClusterSpec | None = None,
    cov: CovSpec | str | Mapping[str, object] | None = None,
    hac_lags: int | None = None,
    kernel: str = "bartlett",
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

    spec = parse_cov_spec(
        cov,
        cov_type=cov_type,
        clusters=clusters,
        nobs=n,
        hac_lags=hac_lags,
        kernel=kernel,
        small_sample=small_sample_adj,
    )

    bread = _pinv_sym(X2.T @ X2)
    df_resid = n - p

    warnings_list: list[str] = []
    meat = _moment_meat(
        X=X2,
        resid1=r,
        resid2=r,
        cov_type=spec.cov_type,
        clusters=spec.clusters,
        hac_lags=spec.hac_lags,
        kernel=spec.kernel,
    )
    cov_mat = bread @ meat @ bread

    if spec.cov_type == "unadjusted":
        sigma2 = float(((r.T @ r) / df_resid).item())
        cov_mat = sigma2 * bread
    elif spec.cov_type == "HC1" and small_sample_adj:
        cov_mat *= n / df_resid
    elif spec.cov_type == "cluster" and small_sample_adj:
        if spec.clusters is None:
            raise ValueError("clusters must be provided when cov_type='cluster'.")
        g = _cluster_codes(spec.clusters)
        uniq, _ = np.unique(g, return_inverse=True)
        G = int(uniq.size)
        cov_mat *= (G / (G - 1)) * ((n - 1) / df_resid)
        warnings_list.extend(_cluster_warnings(g, k=p))
        if np.linalg.matrix_rank(cov_mat) < p:
            warnings_list.append("cluster covariance is not full rank")
        for msg in warnings_list:
            warnings.warn(msg, RuntimeWarning, stacklevel=2)

    return MomentCovarianceResult(
        cov=cov_mat,
        cov_type=str(spec.cov_type),
        df_resid=df_resid,
        nobs=n,
        n_clusters=None
        if spec.cov_type != "cluster" or spec.clusters is None
        else int(np.unique(_cluster_codes(spec.clusters)).size),
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
    cov_type: CovType | None = "HC1",
    clusters: IntArray | ClusterSpec | None = None,
    cov: CovSpec | str | Mapping[str, object] | None = None,
    small_sample_adj: bool = True,
    df_resid_adj: int | None = None,
    hac_lags: int | None = None,
    kernel: str = "bartlett",
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

    spec = parse_cov_spec(
        cov,
        cov_type=cov_type,
        clusters=clusters,
        nobs=n,
        hac_lags=hac_lags,
        kernel=kernel,
        small_sample=small_sample_adj,
    )

    bread = _pinv_sym(X2.T @ X2)
    df_resid = n - p
    df_adj = df_resid if df_resid_adj is None else df_resid_adj

    warnings_list: list[str] = []
    meat_yy = _moment_meat(
        X=X2,
        resid1=ry,
        resid2=ry,
        cov_type=spec.cov_type,
        clusters=spec.clusters,
        hac_lags=spec.hac_lags,
        kernel=spec.kernel,
    )
    meat_dd = _moment_meat(
        X=X2,
        resid1=rd,
        resid2=rd,
        cov_type=spec.cov_type,
        clusters=spec.clusters,
        hac_lags=spec.hac_lags,
        kernel=spec.kernel,
    )
    meat_yd = _moment_meat(
        X=X2,
        resid1=ry,
        resid2=rd,
        cov_type=spec.cov_type,
        clusters=spec.clusters,
        hac_lags=spec.hac_lags,
        kernel=spec.kernel,
    )

    V_yy = bread @ meat_yy @ bread
    V_dd = bread @ meat_dd @ bread
    V_yd = bread @ meat_yd @ bread

    if spec.cov_type == "unadjusted":
        sigma = np.hstack([ry, rd])
        sigma_hat = (sigma.T @ sigma) / df_adj
        V_yy = sigma_hat[0, 0] * bread
        V_dd = sigma_hat[1, 1] * bread
        V_yd = sigma_hat[0, 1] * bread
    elif spec.cov_type == "HC1" and small_sample_adj:
        V_yy *= n / df_adj
        V_dd *= n / df_adj
        V_yd *= n / df_adj
    elif spec.cov_type == "cluster" and small_sample_adj:
        if spec.clusters is None:
            raise ValueError("clusters must be provided when cov_type='cluster'.")
        g = _cluster_codes(spec.clusters)
        uniq, _ = np.unique(g, return_inverse=True)
        G = int(uniq.size)
        adj = (G / (G - 1)) * ((n - 1) / df_adj)
        V_yy *= adj
        V_dd *= adj
        V_yd *= adj
        warnings_list.extend(_cluster_warnings(g, k=p))

    V = np.block([[V_yy, V_yd], [V_yd.T, V_dd]])
    if spec.cov_type == "cluster" and small_sample_adj:
        if np.linalg.matrix_rank(V) < 2 * p:
            warnings_list.append("cluster covariance is not full rank")
        for msg in warnings_list:
            warnings.warn(msg, RuntimeWarning, stacklevel=2)

    return ReducedFormCovarianceResult(
        cov=V,
        cov_type=str(spec.cov_type),
        df_resid=df_resid,
        nobs=n,
        n_clusters=None
        if spec.cov_type != "cluster" or spec.clusters is None
        else int(np.unique(_cluster_codes(spec.clusters)).size),
        warnings=tuple(warnings_list),
    )
