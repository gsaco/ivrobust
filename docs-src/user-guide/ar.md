# Anderson-Rubin inference

The Anderson-Rubin (AR) approach performs inference on a structural parameter by
testing whether excluded instruments predict the null-implied residual. This
yields tests that remain valid under weak instruments.

## Model and null

Consider a linear IV model with one endogenous regressor $d$:

\[
y = \beta d + x'\gamma + u
\]

Given a null $H_0: \beta = \beta_0$, define the null-implied outcome:

\[
\tilde y(\beta_0) = y - \beta_0 d
\]

The AR regression is:

\[
\tilde y(\beta_0) \sim x + z
\]

and the AR test is a joint test that coefficients on $z$ are zero.

## In ivrobust

```python
import ivrobust as ivr
data, beta_true = ivr.weak_iv_dgp(n=300, k=5, strength=0.4, beta=1.0, seed=0)

res = ivr.ar_test(data, beta0=beta_true, cov_type="HC1")
res.statistic, res.pvalue
```

Covariance options: `"unadjusted"`, `"HC0"`, `"HC1"`, `"HC2"`, `"HC3"`, and
`"cluster"` (one-way clustering with `data.clusters`).

Confidence sets return a `ConfidenceSetResult` with a union-of-intervals object:

The AR confidence set is obtained by inverting the AR test across values of
$\beta$.

```python
cs = ivr.ar_confidence_set(data, alpha=0.05, cov_type="HC1")
cs.confidence_set.intervals
```

Intervals may be disjoint or unbounded; this is not a bug. It is a feature of
weak-identification robust inference.

References:

- Anderson, T. W., and Rubin, H. (1949). Estimation of the Parameters of a
  Single Equation in a Complete System of Stochastic Equations. Annals of
  Mathematical Statistics. DOI: 10.1214/aoms/1177730090.
- Andrews, I., Stock, J. H., and Sun, L. (2019). Weak Instruments in
  Instrumental Variables Regression: Theory and Practice. Annual Review of
  Economics. DOI: 10.1146/annurev-economics-080218-025643.
- Mikusheva, A. (2010). Robust confidence sets in the presence of weak
  instruments. Journal of Econometrics. DOI: 10.1016/j.jeconom.2009.12.003.

See the References page for formatted citations.
