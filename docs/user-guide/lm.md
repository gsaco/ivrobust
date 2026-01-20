# LM/K test

The Lagrange Multiplier (LM) or Kleibergen–Paap K test is a weak-IV robust score
test for a scalar structural parameter. It has a chi-square(1) asymptotic
distribution under weak instruments.

## In ivrobust

```python
import ivrobust as ivr
data, beta_true = ivr.weak_iv_dgp(n=300, k=4, strength=0.5, beta=1.0, seed=0)

res = ivr.lm_test(data, beta0=beta_true, cov_type="HC1")
res.statistic, res.pvalue

rk = ivr.kp_rank_test(data, cov_type="HC1")
rk.statistic, rk.pvalue

cs = ivr.lm_confidence_set(data, alpha=0.05, cov_type="HC1")
cs.confidence_set.intervals
```

Notes:

- LM confidence sets can be disjoint or unbounded under weak instruments.
- Use `cov_type="cluster"` for one-way clustering (requires `data.clusters`).
- Available covariance types: `"unadjusted"`, `"HC0"`, `"HC1"`, `"HC2"`, `"HC3"`, `"cluster"`.

References:

- Kleibergen, F. (2002). Pivotal statistics for testing structural parameters in
  instrumental variables regression. Econometrica.
- Mikusheva, A. (2010). Robust confidence sets in the presence of weak
  instruments. Journal of Econometrics.
- Finlay, K., and Magnusson, L. M. (2009). Implementing weak-instrument robust
  tests for a general class of instrumental-variables models. The Stata Journal.
