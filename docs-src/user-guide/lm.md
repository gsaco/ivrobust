# LM/K test

The Lagrange Multiplier (LM) or Kleibergen–Paap K test is a weak-IV robust score
test for a scalar structural parameter. It has a chi-square(1) asymptotic
distribution under weak instruments.

## Purpose

LM/K provides a score-based alternative to AR. It can have higher power in some
settings while retaining weak-ID robustness.

## Statistic

The LM statistic uses the score of the reduced-form likelihood (or its
minimum-distance analog) evaluated at the null. In ivrobust it is implemented
as the Kleibergen–Paap LM statistic for a scalar endogenous regressor.

## Algorithm

1. Residualize $y$, $d$, and $z$ with respect to $x$.
2. Form the score and information terms under $H_0$.
3. Compute the LM statistic and p-value under $\chi^2_1$.

## Interpretation

LM can be more powerful than AR under some designs but can behave similarly in
very weak-ID settings.

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
- Use `cov_type="HAC"` with `hac_lags` and `kernel` for serial correlation.

References:

- Kleibergen, F. (2002). Pivotal statistics for testing structural parameters in
  instrumental variables regression. Econometrica.
- Mikusheva, A. (2010). Robust confidence sets in the presence of weak
  instruments. Journal of Econometrics.
- Finlay, K., and Magnusson, L. M. (2009). Implementing weak-instrument robust
  tests for a general class of instrumental-variables models. The Stata Journal.
