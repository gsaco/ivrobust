# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project adheres to Semantic
Versioning.

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
