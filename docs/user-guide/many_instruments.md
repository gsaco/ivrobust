# Many Instruments

When the number of instruments grows relative to sample size, conventional
first-stage F statistics can overstate strength. ivrobust exposes diagnostics
that remain meaningful under many-instrument asymptotics.

Key diagnostics

- Effective F (Montiel Olea–Pflueger): heteroskedastic/cluster-robust strength.
- Kleibergen–Paap rk statistic: underidentification test.
- Cragg–Donald F (scalar endog): classical counterpart for comparison.
- Stock–Yogo critical values: rule-of-thumb thresholds (partial table).

Example

```python
import ivrobust as ivr

data, _ = ivr.weak_iv_dgp(n=300, k=10, strength=0.3, beta=1.0, seed=0)

diag = ivr.weak_id_diagnostics(data)
diag.effective_f, diag.kp_rk_stat
```

Interpretation

- Effective F below ~10 signals weak identification in many settings.
- The rk test p-value near 0 indicates underidentification.
- Stock–Yogo critical values provide reference cutoffs for size distortions.
