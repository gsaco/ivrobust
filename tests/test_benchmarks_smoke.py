from ivrobust.benchmarks import run_smoke_benchmark


def test_run_smoke_benchmark():
    result = run_smoke_benchmark(seed=1)
    assert "ar_stat" in result
    assert "ar_pvalue" in result
    assert "cs_covers" in result
