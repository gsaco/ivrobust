import numpy as np
import pytest

from ivrobust.linalg import residualize, safe_solve
from ivrobust.utils.warnings import NumericalWarning, WarningRecord


def test_residualize_centering():
    x = np.ones((4, 1))
    y = np.array([1.0, 2.0, 3.0, 4.0])
    resid, _ = residualize(y, x)
    assert np.allclose(resid, y - y.mean())


def test_safe_solve_pinv_warning():
    a = np.array([[1.0, 2.0], [2.0, 4.0]])
    b = np.array([1.0, 2.0])
    record = WarningRecord()
    with pytest.warns(NumericalWarning):
        _, diag = safe_solve(a, b, context="singular", warnings=record)
    assert diag.used_pinv is True
    assert any("pseudoinverse" in msg for msg in record.messages)
