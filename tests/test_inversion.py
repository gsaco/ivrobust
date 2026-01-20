import numpy as np

from ivrobust.weakiv.inversion import GridSpec, InversionSpec, invert_test


def test_inversion_single_interval() -> None:
    grid_spec = GridSpec(beta_bounds=(-2.0, 2.0), n_grid=401)
    inv_spec = InversionSpec(refine=False)

    def pval(b: float) -> float:
        return 1.0 if abs(b) <= 1.0 else 0.0

    cs, info = invert_test(test_fn=pval, alpha=0.5, grid_spec=grid_spec, inversion_spec=inv_spec)
    assert len(cs.intervals) == 1
    lo, hi = cs.intervals[0]
    assert lo <= -1.0 + 0.01
    assert hi >= 1.0 - 0.01
    assert info["n_grid"] == 401


def test_inversion_disjoint_intervals() -> None:
    grid = np.linspace(-3.0, 3.0, 601)
    grid_spec = GridSpec(grid=grid)
    inv_spec = InversionSpec(refine=False)

    def pval(b: float) -> float:
        return 1.0 if (-2.0 <= b <= -1.0) or (1.0 <= b <= 2.0) else 0.0

    cs, _ = invert_test(test_fn=pval, alpha=0.5, grid_spec=grid_spec, inversion_spec=inv_spec)
    assert len(cs.intervals) == 2


def test_inversion_unbounded_interval() -> None:
    grid_spec = GridSpec(beta_bounds=(-1.0, 1.0), n_grid=301)
    inv_spec = InversionSpec(refine=False)

    def pval(_: float) -> float:
        return 0.9

    cs, _ = invert_test(test_fn=pval, alpha=0.5, grid_spec=grid_spec, inversion_spec=inv_spec)
    assert cs.intervals[0][0] == float("-inf")
    assert cs.intervals[0][1] == float("inf")
