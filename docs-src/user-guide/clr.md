# CLR test

The Conditional Likelihood Ratio (CLR) test (Moreira, 2003) is a canonical
weak-IV robust test with strong finite-sample properties. Its p-values are
computed from a nonstandard distribution that depends on a concentration
statistic.

## In ivrobust

```python
import ivrobust as ivr
data, beta_true = ivr.weak_iv_dgp(n=300, k=5, strength=0.4, beta=1.0, seed=0)

res = ivr.clr_test(data, beta0=beta_true, cov_type="HC1", method="CQLR")
res.statistic, res.pvalue

cs = ivr.clr_confidence_set(data, alpha=0.05, cov_type="HC1")
cs.confidence_set.intervals
```

Notes:

- CLR confidence sets can be disjoint or unbounded under weak instruments.
- Set `method="CLR"` for homoskedastic CLR; `method="CQLR"` is the robust default.
- Use `weakiv_inference` to compute AR/LM/CLR tests and sets together.
- Available covariance types: `"unadjusted"`, `"HC0"`, `"HC1"`, `"HC2"`, `"HC3"`, `"cluster"`.

References:

- Moreira, M. J. (2003). A Conditional Likelihood Ratio Test for Structural
  Models. Econometrica.
- Mikusheva, A. (2010). Robust confidence sets in the presence of weak
  instruments. Journal of Econometrics.
- Finlay, K., and Magnusson, L. M. (2009). Implementing weak-instrument robust
  tests for a general class of instrumental-variables models. The Stata Journal.
