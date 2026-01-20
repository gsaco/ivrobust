# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project adheres to Semantic
Versioning.

## [Unreleased]

### Added
- Unified covariance specification (HC/cluster/HAC) and shared covariance engine.
- Weak-IV inference modules for AR/LM/CLR with shared inversion engine.
- Diagnostics for effective F, KP rk, Cragg–Donald F, and weak-ID summaries.
- Estimator suite modules (2SLS/LIML/Fuller/k-class) with `fit` helper.
- Replication harness with golden tables and scripts.
- Benchmarks and numeric regression tests.

### Changed
- Results objects expose intervals/summary helpers and diagnostics export.
- CI includes multi-OS testing and dependency auditing.

## [0.2.0] - 2026-01-17

### Added
- Stable public data model (`IVData`) and strict input validation.
- Verified Anderson-Rubin test implementation against statsmodels (HC and
  cluster).
- Deterministic plotting style system (`set_style`, `savefig`) with regression
  tests.
- MkDocs Material documentation with references and API pages.
- Executable notebook curriculum with CI enforcement.

### Changed
- Public API returns structured result objects with documented attributes.
- CI now gates lint, format, typing, tests, and build checks.

## [0.1.0] - 2025-??-??

### Added
- Initial implementation of AR tests and set-valued confidence sets.
