# Plotting

ivrobust uses a single, package-wide plotting style. It is intentionally
conservative:

- White background
- Grayscale defaults
- Black edges and spines
- No seaborn styling
- Vector-friendly saving (PDF) plus high-DPI PNG

## Style entrypoint

```python
import ivrobust as ivr
ivr.set_style()
```

Saving figures:

```python
paths = ivr.savefig(fig, "artifacts/figures/example", formats=("png", "pdf"))
paths
```

## Weak-IV p-value curves

```python
res = ivr.weakiv_inference(
    data,
    beta0=beta_true,
    alpha=0.05,
    methods=("AR", "LM", "CLR"),
    cov_type="HC1",
    return_grid=True,
)
fig, ax = res.plot()
```
