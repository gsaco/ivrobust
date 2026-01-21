# Roadmap

This roadmap distinguishes between:

- P0: required for a stable release (current scope)
- P1: next methods and diagnostics
- P2: extended method coverage and performance work

## P0 (current)

- AR/LM/CLR tests and confidence sets for a scalar structural parameter.
- HC0/HC1/HC2/HC3, HAC, and one-way cluster-robust covariance options.
- Effective F diagnostics for weak instruments.
- Verified implementation against a reference regression engine.

## P1 (next)

- Multi-endogenous support with scalar-target inference.
- Many-instrument diagnostics and optional many-IV adjustments.

## P2 (extended)

- Formula interface (optional dependency) for applied workflows.
- Performance improvements (QR-based projections, caching repeated
  cross-products).
