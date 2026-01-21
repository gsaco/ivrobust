# Parity and replication

ivrobust includes parity tests against reference implementations where
available:

- `tests/test_ar_vs_statsmodels.py` compares AR results to statsmodels.
- `tests/test_lm_clr.py` compares LM/CLR to ivmodels when installed.

Replication outputs are stored under `replication/outputs/` and validated by
`tests/test_golden_replication.py`.

## Feature gap map (Stata parity benchmark)

| Category | Status in ivrobust | Evidence | Stata expectation | Notes |
| --- | --- | --- | --- | --- |
| AR test (scalar beta) | Yes | `src/ivrobust/weakiv/ar.py` | AR test + CI by inversion | Implemented with grid inversion. |
| AR confidence set inversion | Yes | `src/ivrobust/intervals.py` | Nonstandard CI shapes (empty/union/R) | IntervalSet supports unions and unbounded sets. |
| Kleibergen LM (score) | Yes | `src/ivrobust/weakiv/lm.py` | LM/score tests with robust VCE | Implemented as KP-LM for p_endog=1. |
| CLR / CQLR | Yes | `src/ivrobust/weakiv/clr.py` | CLR test + CI | CQLR default; homoskedastic CLR available. |
| Robust covariance (HC0-3) | Yes | `src/ivrobust/covariance.py` | HC0-HC3 | Supported across tests/estimators. |
| Cluster-robust covariance | Yes | `src/ivrobust/covariance.py` | One-way clustering | Multiway clusters are combined, not CGM. |
| HAC covariance | Yes | `src/ivrobust/covariance.py` | Newey-West | HAC lags and kernel supported. |
| LIML / Fuller / k-class | Yes | `src/ivrobust/estimators/liml.py` | LIML/Fuller estimators | Provided for workflow estimates. |
| Weak-IV diagnostics | Yes | `src/ivrobust/diagnostics/strength.py` | Effective F, Stock-Yogo | Effective F and partial Stock-Yogo table. |
| Multiple endogenous regressors | Not yet | Roadmap | Joint weak-IV inference | Currently p_endog=1. |
| Many-weak instruments CLR variants | Not yet | Roadmap | Many-IV robust variants | Planned; not implemented. |
