import json
from pathlib import Path

import numpy as np

import ivrobust as ivr


def test_replication_golden_table() -> None:
    root = Path(__file__).resolve().parents[1]
    data_path = root / "replication" / "data" / "weak_iv_fixture.csv"
    gold_path = root / "replication" / "outputs" / "golden.json"

    mat = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    y = mat[:, [0]]
    d = mat[:, [1]]
    z = mat[:, 2:5]
    x = mat[:, 5:]
    data = ivr.IVData(y=y, d=d, x=x, z=z)

    with gold_path.open("r", encoding="utf-8") as f:
        gold = json.load(f)

    tsls = ivr.tsls(data, cov_type="HC1")
    liml = ivr.liml(data, cov_type="HC1")
    fuller = ivr.fuller(data, alpha=1.0, cov_type="HC1")
    ar = ivr.ar_test(data, beta0=gold["beta_true"], cov_type="HC1")
    lm = ivr.lm_test(data, beta0=gold["beta_true"], cov_type="HC1")
    clr = ivr.clr_test(data, beta0=gold["beta_true"], cov_type="HC1")

    assert np.isclose(tsls.beta, gold["tsls"]["beta"], rtol=1e-8, atol=1e-8)
    assert np.isclose(liml.beta, gold["liml"]["beta"], rtol=1e-8, atol=1e-8)
    assert np.isclose(fuller.beta, gold["fuller"]["beta"], rtol=1e-8, atol=1e-8)
    assert np.isclose(ar.statistic, gold["ar"]["stat"], rtol=1e-8, atol=1e-8)
    assert np.isclose(lm.statistic, gold["lm"]["stat"], rtol=1e-8, atol=1e-8)
    assert np.isclose(clr.statistic, gold["clr"]["stat"], rtol=1e-8, atol=1e-8)
