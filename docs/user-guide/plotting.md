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
