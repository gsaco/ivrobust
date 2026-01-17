# ivrobust

Weak-instrument robust inference for linear IV/GMM in Python.

**Status:** v0.1.0 implements Anderson-Rubin (AR) tests and set-valued confidence
sets with robust covariance options. CLR/LM and many-IV methods are planned but
not yet implemented (stubs raise `NotImplementedError`).

## Quickstart

```python
import numpy as np
import ivrobust as ivr

# Synthetic weak-IV data (single endogenous regressor, intercept included)
data, beta_true = ivr.weak_iv_dgp(n=300, k=5, strength=0.4, beta=1.0, seed=0)

# Anderson-Rubin test at beta0
ar = ivr.ar_test(data, beta0=[beta_true])
print(ar.statistic, ar.pvalue)

# AR confidence set (may be disjoint or unbounded)
cs = ivr.ar_confidence_set(data, alpha=0.05)
print(cs.confidence_set.intervals)
```

## Scope and assumptions

- Weak-IV robust inference currently covers AR tests and set-valued confidence
  sets for scalar parameters.
- Robust covariance options: HC0/HC1 and one-way cluster-robust moments.
- Cluster-robust inference assumes many clusters; few clusters are flagged.
- Strong-ID estimators (TSLS, LIML) are provided for workflow support; their
  standard errors are not weak-IV robust.

## Diagnostics

- First-stage F and partial R2 are computed for instrument strength reporting.
- Effective F (Montiel Olea-Pflueger) is not yet implemented and is marked as
  out of scope for v0.1.
- Many-instrument warnings are surfaced when k/n is large.

## Install

```bash
pip install -e .
```

## References

See `ivrobust_implementation_appendix.md` for curated references and links.

Key references:
- Anderson & Rubin (1949)
- Andrews, Stock & Sun (2019)
- Mikusheva (2010)
- Moreira (2003)
- Finlay & Magnusson (2009)
- Montiel Olea & Pflueger (2013)
