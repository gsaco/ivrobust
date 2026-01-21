# Estimators

ivrobust provides 2SLS/LIML/Fuller estimators primarily for workflow support
(point estimates, conventional robust standard errors).

Important: 2SLS standard errors are not weak-instrument robust. For weak-IV
robust inference on the structural coefficient, use AR-based routines.

```python
import ivrobust as ivr
data, _ = ivr.weak_iv_dgp(n=1000, k=3, strength=0.8, beta=2.0, seed=1)

res = ivr.tsls(data, cov_type="HC1")
res.beta

liml = ivr.liml(data, cov_type="HC1")
fuller = ivr.fuller(data, alpha=1.0, cov_type="HC1")

fit = ivr.fit(data, estimator="liml", cov_type="HC1")
fit.params
```

Covariance options

- `cov_type="HC0"|"HC1"|"HC2"|"HC3"` for heteroskedasticity-robust SEs
- `cov_type="cluster"` for one-way cluster robust SEs
- `cov_type="HAC"` for Newey–West-type SEs (set `hac_lags` and `kernel`)

## Model wrapper

If you prefer a model-style interface:

```python
model = ivr.IVModel.from_arrays(
    y=data.y,
    x_endog=data.d,
    z=data.z,
    x_exog=None,
    add_const=True,
)
results = model.fit(estimator="liml", cov_type="HC1")
results.params
```

You can also call weak-IV robust inference from results:

```python
weakiv = results.weakiv(methods=("AR", "LM", "CLR"), alpha=0.05)
weakiv.summary()
```
