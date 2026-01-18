# Quickstart

```python
import ivrobust as ivr

data, beta_true = ivr.weak_iv_dgp(n=300, k=5, strength=0.4, beta=1.0, seed=0)

ar = ivr.ar_test(data, beta0=beta_true, cov_type="HC1")
print("AR stat:", ar.statistic)
print("AR pval:", ar.pvalue)

cs = ivr.ar_confidence_set(data, alpha=0.05, cov_type="HC1")
print("CS intervals:", cs.confidence_set.intervals)
```

One figure:

```python
ivr.set_style()
fig, ax = ivr.plot_ar_confidence_set(cs)
ivr.savefig(fig, "artifacts/quickstart/ar_confidence_set", formats=("png", "pdf"))
```

This saves publication-friendly figures to `artifacts/quickstart/`.
