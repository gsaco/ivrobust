# AR confidence sets

AR confidence sets are obtained by test inversion. Under weak identification the
set can be disjoint or unbounded.

## Example

```python
import ivrobust as ivr

data, beta_true = ivr.weak_iv_dgp(n=400, k=6, strength=0.3, beta=1.0, seed=2)

cs = ivr.ar_confidence_set(data, alpha=0.05)
print(cs.confidence_set.intervals)
```

## Diagnostics

- If the acceptance set touches the grid boundary, ivrobust expands the grid.
- If expansions are exhausted, the set is marked as unbounded.
- Numerical warnings report pseudoinverse usage and conditioning.
