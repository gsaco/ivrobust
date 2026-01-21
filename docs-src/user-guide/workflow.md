# Practitioner workflow: estimate -> test -> confidence set

This is the recommended sequence for weak-IV robust analysis.

## 1) Prepare data

```python
import ivrobust as ivr

data = ivr.IVData.from_arrays(
    y=y,
    d=d,
    z=z,
    x=x,          # include controls if needed
    add_const=True,
)
```

## 2) Report diagnostics

```python
diag = ivr.weak_id_diagnostics(data, cov_type="HC1")
diag.effective_f, diag.first_stage_f
```

## 3) Run weak-IV robust tests

```python
res = ivr.weakiv_inference(
    data,
    beta0=0.0,
    alpha=0.05,
    methods=("AR", "LM", "CLR"),
    cov_type="HC1",
)
res.tests["AR"].pvalue
```

## 4) Report confidence sets

```python
cs = res.confidence_sets["AR"]
cs.intervals
```

Intervals may be disjoint or unbounded under weak identification. Report the
full union of intervals.

## 5) Visualize p-value curves (optional)

```python
res_grid = ivr.weakiv_inference(
    data,
    beta0=0.0,
    alpha=0.05,
    methods=("AR", "LM", "CLR"),
    cov_type="HC1",
    grid=(-2.0, 2.0, 301),
    return_grid=True,
)
fig, ax = res_grid.plot()
```

## 6) Pair with point estimates (optional)

```python
tsls = ivr.tsls(data, cov_type="HC1")
tsls.beta, tsls.stderr[-1, 0]
```

Point estimates are useful for reporting but are not weak-IV robust. Always
present weak-IV robust tests and confidence sets when identification is weak.
