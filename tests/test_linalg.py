from pathlib import Path

import numpy as np

from ivrobust.linalg.ops import proj, resid


def test_projection_idempotence() -> None:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 3))
    y = rng.standard_normal((50, 1))

    p1 = proj(X, y)
    p2 = proj(X, p1)
    assert np.allclose(p1, p2, atol=1e-10)


def test_residual_orthogonality() -> None:
    rng = np.random.default_rng(1)
    X = rng.standard_normal((40, 2))
    y = rng.standard_normal((40, 1))

    r = resid(X, y)
    ortho = X.T @ r
    assert np.allclose(ortho, 0.0, atol=1e-10)


def test_rank_deficient_projection_is_finite() -> None:
    rng = np.random.default_rng(2)
    x1 = rng.standard_normal((30, 1))
    X = np.hstack([x1, x1])
    y = rng.standard_normal((30, 1))

    p = proj(X, y)
    r = resid(X, y)
    assert np.isfinite(p).all()
    assert np.isfinite(r).all()


def test_no_explicit_inv_in_core() -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "ivrobust"
    files = list(root.rglob("*.py"))
    for path in files:
        text = path.read_text(encoding="utf-8")
        assert "np.linalg.inv" not in text
