# Result objects and confidence sets

ivrobust returns structured results to make reporting explicit and reproducible.

## TestResult

Fields:

- `statistic`, `pvalue`, `df`
- `method`, `cov_type`
- `cov_config` (e.g., HAC lags, kernel)
- `warnings`, `details`

Use `.summary()` for a printable block or `.as_dict()` for serialization.

## ConfidenceSetResult

Fields:

- `confidence_set` (an `IntervalSet`)
- `alpha`, `method`
- `grid_info` (grid, p-values if requested, runtime, bounds)
- `warnings`

Convenience properties:

- `intervals`
- `is_empty`, `is_unbounded`, `is_disjoint`

## IntervalSet

`IntervalSet` represents a union of real-line intervals:

```python
IntervalSet(intervals=[(-1.0, 0.0), (2.0, 3.5)])
```

Intervals may be unbounded (use `-inf` or `inf`). The set can be empty when no
values pass the inverted test.

## WeakIVInferenceResult

`weakiv_inference` returns:

- `tests`: dict of `TestResult` objects
- `confidence_sets`: dict of `ConfidenceSetResult` objects
- `diagnostics`: dictionary of strength diagnostics

When `return_grid=True`, confidence sets include p-value grids for plotting.
