# ivrobust

Weak-instrument robust inference for linear instrumental variables (IV) models in
Python.

[![CI](https://github.com/gsaco/ivrobust/actions/workflows/ci.yml/badge.svg)](https://github.com/gsaco/ivrobust/actions/workflows/ci.yml)

## Install

For development (recommended while ivrobust is <1.0):

```bash
pip install -e ".[dev,plot]"
```

## 30-second example

```python
import ivrobust as ivr

# Synthetic weak-IV data (single endogenous regressor; intercept included)
data, beta_true = ivr.weak_iv_dgp(n=300, k=5, strength=0.4, beta=1.0, seed=0)

# Anderson-Rubin test at beta0
ar = ivr.ar_test(data, beta0=beta_true, cov_type="HC1")
print(ar.statistic, ar.pvalue)

# AR confidence set (may be disjoint or unbounded)
cs = ivr.ar_confidence_set(data, alpha=0.05, cov_type="HC1")
print(cs.confidence_set.intervals)
```

## Plotting (publication defaults)

All plotting functions and examples use a single unified style:

```python
import ivrobust as ivr

ivr.set_style()
fig, ax = ivr.plot_ar_confidence_set(cs)
ivr.savefig(fig, "artifacts/ar_confidence_set", formats=("png", "pdf"))
```

## Citing ivrobust

If you use ivrobust in academic work, please cite it using the metadata in
`CITATION.cff`.
