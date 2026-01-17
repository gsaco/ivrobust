# ivrobust Implementation Appendix

This file is meant to live in the repo (e.g., `docs/` or the root) as a **design + implementation companion** to `IVROBUST_MASTER_PLAN.md`.

It contains:
1) A **repo-friendly reference list** (stable links) for grounding your docs/examples.
2) **Algorithmic/numerical implementation notes** for the core procedures (especially AR + CI inversion).
3) A **cross-ecosystem parity plan** (Stata/R/Python) that can be turned into golden tests.

> IMPORTANT: Nothing in this appendix is “marketing.” It is a specification and a checklist for correctness and reproducibility.

---

## A. Curated references with stable links (package-grounding)

### A.1 Surveys / big-picture foundations
- Andrews, Stock & Sun (2019), *Weak Instruments in Instrumental Variables Regression: Theory and Practice*, **Annual Review of Economics** (journal page):  
  https://www.annualreviews.org/doi/10.1146/annurev-economics-080218-025643  
  Author PDF (often easier to access):  
  https://lsun20.github.io/WIRev_092218-%20corrected.pdf  
  Supplementary materials:  
  https://lsun20.github.io/appendix.pdf

- Mikusheva & Sun (2023/2024), *Weak Identification with Many Instruments* (survey + extensions), arXiv:  
  https://arxiv.org/abs/2308.09535  
  MIT-hosted PDF (one version):  
  https://economics.mit.edu/sites/default/files/2023-08/paper.pdf  
  Econometrics Journal article page (if you want to cite the journal version):  
  https://academic.oup.com/ectj/article-abstract/27/2/C1/7613563

### A.2 Core weak-IV robust inference (fixed number of instruments)
- Anderson & Rubin (1949), original AR idea (classic; cite via secondary sources if you don’t have a clean PDF link).

- Moreira (2003), *A Conditional Likelihood Ratio Test for Structural Models* (Econometrica):  
  Wiley page (abstract + PDF access depends):  
  https://onlinelibrary.wiley.com/doi/abs/10.1111/1468-0262.00438  
  Alternative hosted PDF (non-official mirror; use only if needed):  
  https://drphilipshaw.com/Moreira-2003.pdf

- Mikusheva (2010), *Robust confidence sets in the presence of weak instruments* (Journal of Econometrics):  
  MIT-hosted PDF:  
  https://dspace.mit.edu/bitstream/handle/1721.1/61662/Mikusheva_Robust%20confidence.pdf?isAllowed=y&sequence=1  
  (This is extremely useful for CI shape logic + inversion algorithms.)

- Kleibergen (2002), *Pivotal statistics for testing structural parameters in instrumental variables regression* (Econometrica):  
  Wiley page:  
  https://onlinelibrary.wiley.com/doi/abs/10.1111/1468-0262.00353

- Kleibergen (2007), *Generalizing weak instrument robust IV statistics…* (Journal of Econometrics):  
  RePEc entry:  
  https://ideas.repec.org/a/eee/econom/v139y2007i1p181-216.html

- Finlay & Magnusson (2009), *Implementing Weak-Instrument Robust Tests…* (Stata Journal):  
  SAGE PDF:  
  https://journals.sagepub.com/doi/pdf/10.1177/1536867X0900900304  
  Author-hosted PDF:  
  https://kfinlay.github.io/pdf/FinlayMagnusson2009.pdf

### A.3 Diagnostics: “instrument strength” beyond naive F
- Montiel Olea & Pflueger (2013), *A Robust Test for Weak Instruments* (JBES):  
  Author PDF:  
  https://joseluismontielolea.com/Montiel-OleaJBES.pdf  
  (Journal page: https://www.tandfonline.com/doi/abs/10.1080/00401706.2013.806694)

- Sanderson & Windmeijer (2016), *A weak instrument F-test in linear IV models with multiple endogenous variables* (Journal of Econometrics):  
  PDF:  
  https://research-information.bris.ac.uk/files/65038476/1_s2.0_S0304407615001736_main.pdf

- Windmeijer (2025), *The robust F-statistic as a test for weak instruments* (Journal of Econometrics):  
  ScienceDirect page:  
  https://www.sciencedirect.com/science/article/pii/S0304407625000053  
  (Also see arXiv entry: https://arxiv.org/abs/2309.01637)

### A.4 Many instruments / many weak instruments
- Mikusheva & Sun (2022), *Inference with Many Weak Instruments* (Review of Economic Studies):  
  OUP page:  
  https://academic.oup.com/restud/article-abstract/89/5/2663/6482756  
  OUP PDF:  
  https://academic.oup.com/restud/article-pdf/89/5/2663/45764025/rdab097.pdf  
  arXiv preprint:  
  https://arxiv.org/abs/2004.12445

- manyweakiv (Stata implementation of Mikusheva–Sun style tools):  
  GitHub: https://github.com/lsun20/manyweakiv  
  SSC/RePEc entry: https://ideas.repec.org/c/boc/bocode/s459275.html

### A.5 Reference software implementations
- Stata `ivregress` postestimation manual (`estat weakrobust`):  
  https://www.stata.com/manuals/rivregresspostestimation.pdf

- Stata `weakiv` package (SSC):  
  https://ideas.repec.org/c/boc/bocode/s457684.html

- Python `ivmodels` package:
  - PyPI: https://pypi.org/project/ivmodels/
  - GitHub: https://github.com/mlondschien/ivmodels
  - Tutorial paper (2025, arXiv): https://arxiv.org/pdf/2508.12474

- R `ivmodel` package (weak-IV robust CIs among other features; also focuses on invalid IV / sensitivity):  
  PDF: https://par.nsf.gov/servlets/purl/10392069

---

## B. Core data conventions (what Codex must standardize)

**Strong recommendation for v0.1** (to avoid ambiguous degrees of freedom and to match common econometrics conventions):

- Treat `Z` as **excluded instruments only** (k columns).
- Treat `X_exog` as **included exogenous regressors** (q columns), including an intercept if desired.
- Internally, always **partial out `X_exog`** from `y`, `X_endog`, and `Z` before computing weak-IV tests and diagnostics.  
  This matches the standard “partialled out” formulation and keeps the moment dimension equal to k.

That is:
- \(M_x = I - X_x (X_x'X_x)^{-1}X_x'\)
- \(y_\perp = M_x y\), \(X_{e,\perp} = M_x X_e\), \(Z_\perp = M_x Z\)

Everything that follows (AR/CLR/LM/etc.) is computed on \((y_\perp, X_{e,\perp}, Z_\perp)\).

---

## C. Anderson–Rubin (AR) test: implementation notes (fixed-k weak-IV core)

### C.1 Estimand and null
Structural parameter vector \(\beta \in \mathbb{R}^p\).

Test:
- \(H_0: \beta = \beta_0\) (point null; for scalar case p=1 this is the primary workflow)
- More generally, allow linear restrictions \(H_0: R\beta = r\) (optional phase).

### C.2 Identification conditions (what AR needs and what it doesn’t)
- AR test only needs **exogeneity of instruments**: \(E[Z_\perp' u_\perp] = 0\).
- It **does not require strong relevance** for correct size.
- If relevance fails completely, AR confidence sets will be uninformative (often all real numbers), which is *correct behavior*.

### C.3 Core statistic (GMM form)
Define:
- residual under the null: \(e(\beta_0) = y_\perp - X_{e,\perp}\beta_0\)
- moment vector: \(g(\beta_0) = \frac{1}{\sqrt{n}} Z_\perp' e(\beta_0)\)  (dimension k)

Estimate covariance of \(\sqrt{n} g(\beta_0)\) by a sandwich / outer-product style estimator.

**Heteroskedastic (HC0-style)**
\[
\widehat{\Omega}(\beta_0) = \frac{1}{n} \sum_{i=1}^n z_{\perp i} z_{\perp i}' e_i(\beta_0)^2
\]

**One-way cluster (cluster-robust)**
\[
\widehat{\Omega}(\beta_0) = \frac{1}{n} \sum_{g=1}^G \left( \sum_{i\in g} z_{\perp i} e_i(\beta_0)\right)\left( \sum_{i\in g} z_{\perp i} e_i(\beta_0)\right)'
\]
(with finite-sample adjustments optional but must be documented.)

Then the AR test statistic:
\[
AR(\beta_0) = g(\beta_0)' \widehat{\Omega}(\beta_0)^{-1} g(\beta_0)
\]
Asymptotically under \(H_0\): \(AR(\beta_0) \overset{d}{\to} \chi^2_k\) under standard regularity conditions.

**p-value**: \(p = 1 - F_{\chi^2_k}(AR)\)

### C.4 Failure modes and numerical pathologies
- \(\widehat{\Omega}(\beta_0)\) can be singular/ill-conditioned (especially with many instruments, collinear Z, small clusters).
- Remedy (must be explicit):
  - Use symmetric eigenvalue clean-up: `Omega = (Omega + Omega.T)/2`
  - Try `scipy.linalg.cho_factor` if SPD, else fall back to `pinv` with thresholding.
  - Emit a `NumericalWarning` including condition number and rank.

### C.5 Confidence set inversion (scalar β, p=1)
Define critical value \(c_{1-\alpha}\) from chi-square distribution.

Confidence set:
\[
CS_{AR}(\alpha) = \{\beta_0 \in \mathbb{R} : AR(\beta_0) \le c_{1-\alpha}\}
\]

**Key fact:** \(CS_{AR}\) can be:
- a bounded interval,
- a union of disjoint intervals,
- unbounded (e.g., \((-\infty, a] \cup [b, \infty)\)),
- the whole line.

**Robust inversion algorithm (recommended)**
1. Require user to supply an initial `grid` or supply an automatic one based on:
   - TSLS and LIML point estimates,
   - their (strong-ID) SEs just for scale,
   - plus a fallback wide range if diagnostics indicate weak ID.
2. Evaluate `AR(beta)` on a dense grid.
3. Determine acceptance boolean vector `acc = AR <= crit`.
4. Extract connected components of `acc`.
5. Refine each boundary by root-finding on `AR(beta) - crit` using brackets from the grid.
6. If acceptance touches grid boundaries, adaptively expand the grid (log-scale outward) until either:
   - acceptance no longer touches boundary, or
   - a maximum range is reached, after which mark the CS as “unbounded” and record this in diagnostics.

**Representing the set**
Return a `ConfidenceSet` object that stores:
- list of intervals (each can be finite or ±∞),
- metadata: alpha, method, inversion grid, number of expansions,
- warnings.

### C.6 Result object contract (AR)
Suggested result schema (`ARTestResult`):
- `statistic: float`
- `df: int`
- `pvalue: float`
- `alpha: float` (if CI was requested)
- `confidence_set: ConfidenceSet | None`
- `assumptions: AssumptionSpec`
- `cov_spec: CovSpec`
- `numerics: NumericalDiagnostics`
- `diagnostics: StrengthDiagnostics`
- `reproducibility: ReproSpec`

---

## D. CLR and LM/K tests: implementation notes (what to do first)

### D.1 CLR: what to scope in v0.1 vs v0.2
CLR is most standardized for **p=1, homoskedastic** settings (classical Moreira CLR). In software practice:
- Stata computes CLR p-values exactly in the p=1 homoskedastic case, and uses a heteroskedastic generalization otherwise.  
  (See Stata manual; and Finlay–Magnusson, Kleibergen connections.)

**Recommended approach**
- v0.1: implement AR robustly + set inversion + diagnostics + benchmarks.  
  (This gives immediate practical value and prevents the project from getting stuck.)
- v0.2: implement CLR(p=1, homoskedastic) with:
  - careful numerical integration/simulation for p-values,
  - inversion algorithm per Mikusheva (2010),
  - cross-check vs Stata `estat weakrobust` in homoskedastic mode and vs `ivmodels` outputs.

### D.2 LM / K-type tests
LM/score tests (Kleibergen-style) can be implemented for fixed-k and general covariance, but you must:
- specify the exact statistic used (there are multiple closely related “K” and “LM” forms),
- validate against Stata/ivmodels,
- document when the asymptotic reference distribution is chi-square.

---

## E. Golden parity testing plan (high-value, reproducible)

### E.1 Python-vs-Python parity (CI-friendly): compare to `ivmodels`
Because `ivmodels` already implements AR/LM/CLR workflows, it is ideal for automated golden tests:
- Add `ivmodels` as a **test-only dependency** (e.g., `pip install .[test]`).
- For fixed RNG and fixed DGP, compute:
  - AR statistic and p-value at chosen β0,
  - AR confidence set at alpha=0.05 for a few DGP seeds,
  - (later) CLR and LM tests.
- Compare within tolerances and record any systematic differences.

### E.2 Python-vs-Stata parity (manual / opt-in)
Stata is not usually available in CI, but you can:
- add a `parity/` folder containing:
  - a data export script in Python that writes CSV,
  - a Stata do-file that reads CSV and runs `ivregress` + `estat weakrobust`,
  - a parser to read Stata logs or exported results.
- Document how to run it locally for gold-standard verification.

Key Stata references:
- `estat weakrobust`: https://www.stata.com/manuals/rivregresspostestimation.pdf
- `weakiv` package: https://ideas.repec.org/c/boc/bocode/s457684.html
- `manyweakiv`: https://github.com/lsun20/manyweakiv

---

## F. Repository “prettiness” checklist (release-ready hygiene)

Codex should implement these as part of making the repo “pretty” and release-ready:

- `pyproject.toml` with:
  - PEP 621 metadata,
  - minimal runtime deps (`numpy`, `scipy`),
  - optional extras: `dev`, `docs`, `bench`, `test`.
- `src/` layout.
- `README.md` with:
  - 60-second quickstart,
  - citations + links (use Section A above),
  - clear scope boundaries.
- `docs/` with:
  - assumption cards & procedure cards (templates in master plan),
  - tutorial pages,
  - API reference (auto-generated if possible).
- `tests/` with:
  - unit + integration + golden tests,
  - property tests for invariance (instrument transformations),
  - small simulation smoke tests.
- CI workflows:
  - lint/type/tests,
  - docs build.
- Governance:
  - `LICENSE`, `CITATION.cff`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `CHANGELOG.md`.

---

## G. “Don’t accidentally lie” rules for docs/examples

To keep the package scientifically trustworthy:
- Every method doc page must include:
  - *assumptions*,
  - *validity regime* (fixed-k weak-IV vs many-IV),
  - *what it does not address* (invalid IV).
- Never imply “cluster robust” is automatically valid for any test without stating the asymptotic regime (many clusters vs few clusters).
- Never present “F > 10” as a universal rule; always contextualize with effective F and many-IV warnings.

