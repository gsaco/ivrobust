# Estimators

ivrobust provides 2SLS primarily for workflow support (point estimates,
conventional robust standard errors).

Important: 2SLS standard errors are not weak-instrument robust. For weak-IV
robust inference on the structural coefficient, use AR-based routines.

```python
import ivrobust as ivr
data, _ = ivr.weak_iv_dgp(n=1000, k=3, strength=0.8, beta=2.0, seed=1)

res = ivr.tsls(data, cov_type="HC1")
res.beta
```
