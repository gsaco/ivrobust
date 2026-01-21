import numpy as np

from ivrobust.intervals import IntervalSet


def test_interval_set_union_and_normalize() -> None:
    left = IntervalSet(intervals=[(0.0, 1.0), (-np.inf, -1.0)])
    right = IntervalSet(intervals=[(-2.0, -0.5), (1.0, np.inf)])
    union = left.union(right)

    assert len(union.intervals) == 2
    lo1, hi1 = union.intervals[0]
    lo2, hi2 = union.intervals[1]
    assert lo1 == float("-inf")
    assert np.isclose(hi1, -0.5)
    assert np.isclose(lo2, 0.0)
    assert hi2 == float("inf")


def test_interval_set_roundtrip() -> None:
    original = IntervalSet(intervals=[(-1.0, 0.0), (2.0, 3.0)])
    payload = original.to_dict()
    restored = IntervalSet.from_dict(payload)
    assert restored.intervals == original.intervals


def test_interval_set_flags() -> None:
    empty = IntervalSet(intervals=[])
    assert empty.is_empty

    real_line = IntervalSet(intervals=[(-np.inf, np.inf)])
    assert real_line.is_real_line
    assert real_line.is_unbounded
