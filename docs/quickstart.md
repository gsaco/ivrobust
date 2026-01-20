# Quickstart

```python
import ivrobust as ivr

data, beta_true = ivr.weak_iv_dgp(n=300, k=5, strength=0.4, beta=1.0, seed=0)

res = ivr.weakiv_inference(
    data,
    beta0=beta_true,
    alpha=0.05,
    methods=("AR", "LM", "CLR"),
    cov_type="HC1",
)
print("AR stat:", res.tests["AR"].statistic)
print("AR pval:", res.tests["AR"].pvalue)
print("CLR intervals:", res.confidence_sets["CLR"].confidence_set.intervals)
```

One figure:

```python
ivr.set_style()
cs = res.confidence_sets["AR"]
fig, ax = ivr.plot_ar_confidence_set(cs)
ivr.savefig(fig, "artifacts/quickstart/ar_confidence_set", formats=("png", "pdf"))
```

This saves publication-friendly figures to `artifacts/quickstart/`.
