# Clustered data caveats

Cluster-robust covariance is supported for AR inference, but validity depends on
having a sufficiently large number of clusters.

## Example

```python
import ivrobust as ivr

data, beta_true = ivr.weak_iv_dgp(
    n=300, k=5, strength=0.4, beta=1.0, seed=3, n_clusters=30
)

ar = ivr.ar_test(data, beta0=[beta_true], cov="HC1")
print(ar.pvalue)
```

## Warnings

- Few clusters trigger a warning; inference may be unreliable.
- Clustered weak-IV inference is still an active research area. Interpret with
  caution, and consider sensitivity checks.
