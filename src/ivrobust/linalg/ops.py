from __future__ import annotations

from typing import cast

import numpy as np

from .._typing import FloatArray


def _as_2d(x: FloatArray) -> FloatArray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def qr_residualize(y: FloatArray, X: FloatArray) -> FloatArray:
    """
    Residualize y on X using a QR projection.
    """
    y2 = _as_2d(y)
    X2 = _as_2d(X)
    if X2.size == 0:
        return y2
    q, _ = np.linalg.qr(X2, mode="reduced")
    return cast(FloatArray, y2 - q @ (q.T @ y2))


def proj(X: FloatArray, Y: FloatArray) -> FloatArray:
    """
    Project Y onto the column space of X via QR.
    """
    Y2 = _as_2d(Y)
    X2 = _as_2d(X)
    if X2.size == 0:
        return np.zeros_like(Y2)
    q, _ = np.linalg.qr(X2, mode="reduced")
    return cast(FloatArray, q @ (q.T @ Y2))


def resid(X: FloatArray, Y: FloatArray) -> FloatArray:
    """
    Residualize Y on X via QR projections.
    """
    Y2 = _as_2d(Y)
    return Y2 - proj(X, Y2)


def sym_quadform(A: FloatArray, x: FloatArray) -> float:
    """
    Symmetric quadratic form x' A x.
    """
    A2 = np.asarray(A, dtype=np.float64)
    x2 = _as_2d(x)
    return float((x2.T @ A2 @ x2).ravel()[0])


def pinv_solve(A: FloatArray, b: FloatArray, *, rcond: float = 1e-12) -> FloatArray:
    """
    Solve A x = b via SVD-based pseudo-inverse.
    """
    A2 = np.asarray(A, dtype=np.float64)
    b2 = _as_2d(b)
    u, s, vt = np.linalg.svd(A2, full_matrices=False)
    tol = rcond * np.max(s) if s.size else 0.0
    s_inv = np.where(s > tol, 1.0 / s, 0.0)
    return cast(FloatArray, (vt.T * s_inv) @ (u.T @ b2))


def sym_solve(
    A: FloatArray, b: FloatArray, *, jitter: float = 0.0, rcond: float = 1e-12
) -> FloatArray:
    """
    Solve symmetric systems with optional jitter; fall back to pseudo-inverse.
    """
    A2 = np.asarray(A, dtype=np.float64)
    if jitter > 0:
        A2 = A2 + np.eye(A2.shape[0]) * jitter
    try:
        c = np.linalg.cholesky(A2)
        return np.linalg.solve(c.T, np.linalg.solve(c, _as_2d(b)))
    except np.linalg.LinAlgError:
        return pinv_solve(A2, b, rcond=rcond)
