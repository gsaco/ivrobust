# Parity and replication

ivrobust includes parity tests against reference implementations where
available:

- `tests/test_ar_vs_statsmodels.py` compares AR results to statsmodels.
- `tests/test_lm_clr.py` compares LM/CLR to ivmodels when installed.

Replication outputs are stored under `replication/outputs/` and validated by
`tests/test_golden_replication.py`.
