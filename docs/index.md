# ivrobust

Weak-instrument robust inference for linear IV/GMM in Python.

## Quickstart

```python
import ivrobust as ivr

data, beta_true = ivr.weak_iv_dgp(n=300, k=5, strength=0.4, beta=1.0, seed=0)

ar = ivr.ar_test(data, beta0=[beta_true])
print(ar.statistic, ar.pvalue)

cs = ivr.ar_confidence_set(data, alpha=0.05)
print(cs.confidence_set.intervals)
```

## Scope

- AR tests and set-valued confidence sets for scalar parameters.
- Robust covariance options: HC0/HC1 and one-way cluster-robust moments.
- Strong-ID estimators (TSLS, LIML) for workflow support; standard errors are
  not weak-IV robust.

## Next pages

- Methods: Anderson-Rubin
- Tutorials: IV basics, AR confidence sets, clustered data caveats
- Reference: API overview
