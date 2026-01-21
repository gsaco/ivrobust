# Stable public API (0.x contract)

The following names are treated as stable in the 0.x series. Backwards
incompatible changes require deprecation warnings before removal.

## Core data + DGP

- `IVData`
- `IVData.from_arrays`
- `weak_iv_dgp`

## Weak-IV robust inference

- `weakiv_inference`
- `ar_test`
- `ar_confidence_set`
- `lm_test`
- `lm_confidence_set`
- `clr_test`
- `clr_confidence_set`

## Diagnostics

- `first_stage_diagnostics`
- `effective_f`
- `weak_id_diagnostics`
- `kp_rank_test`

## Estimators (workflow support)

- `tsls`
- `liml`
- `fuller`
- `fit`

## Plotting

- `set_style`
- `plot_ar_confidence_set`
- `savefig`

Notes

- Weak-IV robust procedures currently target a single endogenous regressor
  (`p_endog=1`).
- Confidence sets can be empty, unbounded, or disjoint; this is expected under
  weak identification.
