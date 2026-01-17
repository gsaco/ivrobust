# Anderson-Rubin (AR)

## Estimand and null

- Estimand: structural coefficient vector \(\beta\) in linear IV.
- Null: \(H_0: \beta = \beta_0\).

## Assumptions

- Instrument exogeneity: \(E[Z'u] = 0\).
- Linear IV model with correct specification.
- Weak identification is allowed; AR remains valid under weak instruments.

## Validity regime

- Fixed number of instruments (fixed-k) asymptotics.
- Robust to weak instruments.
- Cluster-robust inference assumes many clusters.

## Covariance regimes

- HC0/HC1 (heteroskedasticity-robust moment covariance).
- One-way cluster-robust moment covariance (many clusters).
- Homoskedastic option is provided for parity checks; not default.

## Procedure

1. Partial out exogenous regressors from \(y\), \(X\), and \(Z\).
2. For candidate \(\beta_0\), compute residuals \(e = y - X \beta_0\).
3. Form moments \(g = Z'e / \sqrt{n}\) and covariance \(\Omega\).
4. AR statistic: \(g'\Omega^{-1} g\).
5. Invert the test over \(\beta_0\) to obtain a set-valued confidence set.

## Failure modes and diagnostics

- \(\Omega\) can be singular or ill-conditioned; this triggers warnings and
  pseudoinverse fallback diagnostics.
- Confidence sets can be disjoint or unbounded under weak identification.
- Large k/n triggers many-instrument warnings.

## References

- Anderson & Rubin (1949)
- Andrews, Stock & Sun (2019)
- Mikusheva (2010)
