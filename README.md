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

# Weak-IV robust inference at beta0
res = ivr.weakiv_inference(
    data,
    beta0=beta_true,
    alpha=0.05,
    methods=("AR", "LM", "CLR"),
    cov_type="HC1",
)
print(res.tests["AR"].statistic, res.tests["AR"].pvalue)
print(res.confidence_sets["CLR"].confidence_set.intervals)
```

## Plotting (publication defaults)

All plotting functions and examples use a single unified style:

```python
import ivrobust as ivr

ivr.set_style()
cs = res.confidence_sets["AR"]
fig, ax = ivr.plot_ar_confidence_set(cs)
ivr.savefig(fig, "artifacts/ar_confidence_set", formats=("png", "pdf"))
```

## Citing ivrobust

If you use ivrobust in academic work, please cite it using the metadata in
`CITATION.cff`.
