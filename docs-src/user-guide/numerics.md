# Numerical stability and edge cases

Weak-IV robust inference relies on projections, quadratic forms, and test
inversion. This page summarizes the numerical choices in ivrobust.

## Residualization and rank deficiency

- Projections use QR/SVD-based routines to avoid explicit matrix inversion.
- Near-collinearity in instruments or controls can lead to rank-deficient
  matrices; ivrobust falls back to pseudo-inverses and emits warnings where
  possible.

## Confidence set inversion

Confidence sets are computed by evaluating p-values on a grid and inverting the
acceptance region:

```
{ beta : p(beta) >= alpha }
```

Key numerical choices:

- Deterministic grid spacing (reproducible).
- Bracketing and refinement near acceptance boundaries.
- Explicit handling of unbounded or empty regions.

## Nonstandard set shapes

Under weak identification, confidence sets may be:

- Empty.
- Unbounded (the real line).
- A union of disjoint intervals.

These shapes are expected and should be reported without trimming.

## Reproducibility rules

- Use fixed seeds in synthetic examples.
- Save plots with `ivr.savefig` to deterministic paths.
- Keep grid definitions explicit when comparing across methods.
