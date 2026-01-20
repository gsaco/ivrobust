# Diagnostics and interpretation

Weak-IV robust inference is only as good as the diagnostics you report.
This page summarizes what ivrobust computes, how to interpret it, and which
outputs are valid under weak identification.

## Estimators vs tests vs confidence sets

- **Estimators** (2SLS, LIML, Fuller) provide point estimates and standard
  errors.
- **Tests** (AR, LM/K, CLR) remain valid under weak identification for a single
  endogenous regressor.
- **Confidence sets** are obtained by inverting tests and can be disjoint or
  unbounded.

If instruments are weak, rely on the weak-IV robust tests and confidence sets
rather than conventional t-tests from 2SLS.

## Core diagnostics in ivrobust

```python
import ivrobust as ivr

diag = ivr.first_stage_diagnostics(data)
eff_f = ivr.effective_f(data, cov_type="HC1")
weak = ivr.weak_id_diagnostics(data, cov_type="HC1")
kp_rk = ivr.kp_rank_test(data, cov_type="HC1")
```

**How to read them**

- **First-stage F / partial R^2**: Low values signal weak instruments. These are
  convenient summaries, but they do not guarantee valid inference under weak ID.
- **Effective F (Montiel Olea-Pflueger)**: A robust diagnostic intended for
  heteroskedastic settings. Treat small values as a warning sign.
  [@montielolea2013]
- **KP-rk test**: Tests underidentification. A weak or insignificant result
  suggests insufficient instrument relevance.

## Safe defaults

- Use `weakiv_inference(..., methods=("AR", "LM", "CLR"))` and report the full
  set-valued confidence sets.
- For most workflows, use the package-wide default `cov_type="HC1"` unless your
  design requires clustering.
- When instrument strength is low, emphasize AR/LM/CLR results and use 2SLS
  standard errors only for descriptive reporting.

## Common pitfalls

- **Treating a narrow 2SLS interval as conclusive** under weak instruments.
- **Suppressing disjoint intervals** instead of reporting the union of sets.
- **Ignoring clustering** when the data are clearly grouped.

## Scope reminders

ivrobust's weak-IV robust tests currently target a **single endogenous
regressor** (`p_endog=1`). For multiple endogenous regressors, interpret results
with care and consult the roadmap.
