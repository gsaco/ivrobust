# IV basics and weak instruments

## Why weak instruments matter

Weak instruments violate the usual strong-identification assumptions that justify
Wald tests and normal approximations for TSLS. In weak-ID settings, standard
errors that are "robust" to heteroskedasticity are not sufficient for valid
inference.

## Minimal example

```python
import ivrobust as ivr

data, beta_true = ivr.weak_iv_dgp(n=300, k=5, strength=0.2, beta=1.0, seed=1)

# TSLS point estimate (strong-ID approximation)
res = ivr.tsls(data)
print(res.params, res.std_errors)

# Weak-IV robust AR test
ar = ivr.ar_test(data, beta0=[beta_true])
print(ar.statistic, ar.pvalue)
```

## Interpretation guardrails

- Robust standard errors do not fix weak-ID bias.
- Set-valued confidence regions are expected under weak identification.
- Many instruments require additional diagnostics; fixed-k methods may fail.
