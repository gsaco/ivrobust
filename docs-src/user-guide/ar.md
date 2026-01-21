# Anderson-Rubin inference

The Anderson-Rubin (AR) approach performs inference on a structural parameter by
testing whether excluded instruments predict the null-implied residual. This
yields tests that remain valid under weak instruments.

## Purpose

AR provides weak-IV robust inference on a scalar structural parameter. It
remains correctly sized even when instruments are weak.

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

## Test statistic

Let $r(\beta_0) = y - \beta_0 d$. The AR statistic is the (robust) Wald test of
the coefficients on $z$ in the regression of $r(\beta_0)$ on $[x, z]$.

## Algorithm

1. Residualize $y$, $d$, and $z$ with respect to $x$.
2. Form $r(\beta_0) = y - \beta_0 d$.
3. Compute the robust covariance for moments $z r(\beta_0)$.
4. Form the Wald statistic and p-value under $\chi^2_k$.

## Confidence set by inversion

The $(1-\alpha)$ confidence set is:

\[
CS_{1-\alpha} = \{\beta : p_{AR}(\beta) \ge \alpha\}.
\]

Intervals may be empty, unbounded, or disjoint.

## Interpretation

Wide or unbounded sets indicate weak identification: the data contain limited
information about $\beta$.

## Pitfalls / numerics

- Grid bounds can affect numerical inversion; use wide bounds in weak-ID cases.
- Discontinuities in p-values can create union intervals; report them directly.

## In ivrobust

```python
import ivrobust as ivr
data, beta_true = ivr.weak_iv_dgp(n=300, k=5, strength=0.4, beta=1.0, seed=0)

res = ivr.ar_test(data, beta0=beta_true, cov_type="HC1")
res.statistic, res.pvalue
```

Covariance options: `"unadjusted"`, `"HC0"`, `"HC1"`, `"HC2"`, `"HC3"`,
`"cluster"` (one-way clustering with `data.clusters`), and `"HAC"` (Newey-West
with `hac_lags` and `kernel`).

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
