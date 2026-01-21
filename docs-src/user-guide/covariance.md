# Robust covariance options and what they change

Weak-IV robust tests depend on the covariance of reduced-form moments. This
page summarizes the supported covariance types and how they affect inference.

## Covariance options

| cov_type | Use case | Notes |
| --- | --- | --- |
| `unadjusted` | Homoskedastic iid | Classic formulas; used for exact CLR. |
| `HC0`-`HC3` | Heteroskedasticity | Robust sandwich estimators. |
| `cluster` | Grouped data | One-way clustered covariance; supply `clusters`. |
| `HAC` | Serial correlation | Newey-West style HAC with `hac_lags` and `kernel`. |

## Example

```python
res = ivr.weakiv_inference(
    data,
    beta0=0.0,
    cov_type="HAC",
    hac_lags=4,
    kernel="bartlett",
)
```

## Practical guidance

- Use `HC1` as a default in cross-sectional settings.
- Use `cluster` when observations are grouped (firms, regions, cohorts).
- Use `HAC` in time-series or panel settings with serial correlation.

## Diagnostics interaction

Effective F and KP-rk diagnostics are computed using the same covariance
specification, so change `cov_type` consistently across your workflow.
