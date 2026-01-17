# Implementation Checklist

This checklist follows `IVROBUST_MASTER_PLAN.md` and the appendix. It is ordered
and intended to be checked off in sequence.

## Step 0 — Repo audit and migration plan
- [x] Audit current repo structure and files
- [x] Summarize current API (if any) and missing scaffolding
- [x] Draft migration plan to `src/` layout and stable public API

### Step 0 notes (current state)
- Existing code is scaffolding-only: `src/ivrobust/*` subpackages are empty stubs.
- Tests contain only a version smoke test; docs are a single landing page.
- Packaging, CI, and governance files already exist but need content updates
  aligned with the master plan.

### Step 0 migration plan (target tree)
- Keep `src/` layout (already present) and populate:
  - `ivrobust/data` (IVData, ClusterSpec, IVDesign)
  - `ivrobust/linalg` (residualization, safe solve, quadratic forms)
  - `ivrobust/cov` (HC and cluster covariance)
  - `ivrobust/estimators` (TSLS, LIML)
  - `ivrobust/weakiv` (AR test + CS)
  - `ivrobust/diagnostics` (strength + numerics)
  - `ivrobust/benchmarks` (DGP + runner)
- Add tests, docs, and examples in parallel with core methods.

## Step 1 — Packaging and quality gates
- [x] Add `pyproject.toml` with PEP 621 metadata (`ivrobust`, version `0.1.0`)
- [x] Add runtime deps (`numpy`, `scipy`) and optional extras (`test`, `dev`, `docs`, `bench`)
- [x] Add `LICENSE`, `CITATION.cff`, `CHANGELOG.md`
- [x] Add `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`
- [x] Add toolchain configs: `ruff`, `mypy`, `pytest`, `pre-commit`
- [x] Add GitHub Actions CI (lint, type, tests, docs build)

## Step 2 — Core data contracts and linear algebra utilities
- [x] Implement `IVData` with strict validation (appendix spec)
- [x] Implement `IVDesign` cache object (partialled-out variables + cross-products)
- [x] Implement stable residualization (QR/SVD-based)
- [x] Implement safe solve / conditioning diagnostics / quadratic forms

## Step 3 — Covariance estimators (HC + cluster)
- [x] Implement HC0–HC3 (at least HC0 and HC1)
- [x] Implement one-way cluster robust covariance
- [x] Implement validity checks and warnings (few clusters, rank, conditioning)

## Step 4 — Estimators (supporting workflows)
- [x] Implement TSLS estimator + result object
- [x] Implement LIML estimator + result object
- [x] Add diagnostics summary in estimator results

## Step 5 — Weak-IV robust inference: AR (release core)
- [x] Implement `ar_test` (stat + p-value) with robust covariance
- [x] Implement `ar_confidence_set` (set-valued, grid + root find)
- [x] Add invariance checks (instrument scaling)
- [x] Add diagnostics and warnings surfaced in results

## Step 6 — Diagnostics (first-class)
- [x] First-stage F and partial R2
- [x] Effective F (only if implemented/validated; otherwise mark experimental)
- [x] Many-instrument warnings (`k/n`, rank, conditioning)
- [x] Numerical diagnostics (condition numbers, fallback usage)

## Step 7 — CLR and LM/K (only if validated)
- [ ] Implement CLR (p=1, homoskedastic) with validated p-values
- [ ] Implement LM/K test variant with clear definition and validation
- [x] Add `NotImplementedError` stubs with docs if not ready

## Step 8 — Benchmarks and reproducibility harness
- [x] Implement DGP generators (weak, many, heteroskedastic, cluster)
- [x] Implement benchmark runner (JSON/CSV outputs)
- [x] Add smoke benchmark tests for CI

## Step 9 — Docs and examples
- [x] README quickstart (AR test + AR CS)
- [x] Assumption and procedure cards (AR first)
- [x] Tutorials (IV basics, AR CS, clustered caveats)
- [x] Examples (script + optional notebook)

## Step 10 — Testing strategy and CI parity
- [x] Unit tests (data validation, linalg, cov)
- [x] Integration tests (TSLS vs closed-form, AR invariance)
- [x] Golden tests vs `ivmodels` (AR stat + CS)
- [x] Docs build in CI

## Final release readiness
- [x] `pip install -e .` works
- [x] `pytest -q` passes
- [x] `ruff check .` passes
- [x] `mypy` passes (or documented exceptions)
- [x] Docs build cleanly
- [x] Examples run end-to-end
- [x] Public APIs documented with type hints
