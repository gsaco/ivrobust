# Diagnostics

First-stage strength reporting is standard practice.

ivrobust provides:

- Classical first-stage F-statistic
- Partial $R^2$
- Effective F (heteroskedasticity/cluster-robust)
- Kleibergen–Paap rk statistic (underidentification)
- Cragg–Donald F (scalar endogenous regressor)
- Stock–Yogo critical values (partial table)

```python
import ivrobust as ivr
data, _ = ivr.weak_iv_dgp(n=300, k=5, strength=0.4, beta=1.0, seed=0)

diag = ivr.first_stage_diagnostics(data)
diag.f_statistic, diag.partial_r2
```

## Effective F (Montiel Olea-Pflueger)

The effective F statistic is designed to remain informative under
heteroskedasticity or clustering. It is based on the robust covariance of the
reduced-form coefficients in the first stage.

```python
eff = ivr.effective_f(data, cov_type="HC1")
eff.statistic
```

Reference: Montiel Olea and Pflueger (2013), DOI: 10.1080/00401706.2013.806694.
See the References page for formatted citations.

## Weak-ID summary

```python
diag = ivr.weak_id_diagnostics(data)
diag.kp_rk_stat, diag.kp_rk_pvalue
```

The `weak_id_diagnostics` helper bundles effective F, first-stage F,
Kleibergen–Paap rk, and Cragg–Donald statistics for quick reporting.

```python
ivr.stock_yogo_critical_values(k_endog=1, k_instr=3, size_distortion=0.10)
```

## Numerical stability

ivrobust uses QR/SVD-based projections in its core linear algebra routines to
avoid explicit matrix inversion. Rank-deficient designs trigger warnings and
are handled with pseudo-inverse fallbacks where possible.
