# Roadmap

This roadmap distinguishes between:

- P0: required for a stable release (current scope)
- P1: next methods and diagnostics
- P2: extended method coverage and performance work

## P0 (current)

- AR tests and AR confidence sets for a scalar structural parameter.
- HC0/HC1 and one-way cluster-robust covariance options.
- Verified implementation against a reference regression engine.

## P1 (next)

- Montiel Olea-Pflueger effective F-statistic for weak-instrument diagnostics.
- CLR test and confidence sets (single endogenous regressor).
- LM/score-type weak-IV robust tests (linear IV).

## P2 (extended)

- Many-instrument robust inference pathways (method-dependent; design carefully).
- Additional covariance structures (HAC/serial correlation) where supported.
- Formula interface (optional dependency) for applied workflows.
- Performance improvements (QR-based projections, caching repeated
  cross-products).
