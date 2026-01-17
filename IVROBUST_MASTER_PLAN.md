# 1. Executive Summary

- **Weak-IV / many-IV robust inference** means hypothesis tests and confidence sets for linear IV parameters that **retain correct size/coverage when instruments are weak** (first-stage near rank-deficient) and/or **numerous** (number of instruments large relative to sample size). It is not the same as “robust standard errors”.
- A “complete” weak-IV package should implement (at least) **AR (Anderson–Rubin) tests/CIs**, **CLR tests/CIs**, and **LM/score-type tests/CIs** (including Kleibergen-type statistics where relevant), plus **diagnostics** and **benchmark suites** that validate size/power/coverage under weak and many instruments.
- Weak instruments break the standard “2SLS is approximately normal” regime; **Wald-type CIs can severely undercover**, and t-tests can severely overreject even with robust/cluster SEs.
- **Anderson–Rubin (AR)** inference remains valid under weak identification in linear IV, and produces **set-valued confidence regions** that may be intervals, unions of intervals, the whole real line, or empty (rarely, depending on inversion and model). Its power can be low when identification is strong and the model is overidentified.
- **Conditional Likelihood Ratio (CLR)** inference is designed to improve power (especially in overidentified, homoskedastic settings) and is widely used in applied work; in Stata, `estat weakrobust` reports CLR by default in overidentified homoskedastic models and uses a heteroskedastic generalization (Finlay–Magnusson) otherwise. 
- For heteroskedasticity/clustered errors, some weak-IV tests have **GMM/QCLR generalizations** (e.g., Finlay–Magnusson; Kleibergen’s GMM quasi-LR connection is noted in Stata docs). 
- Many-instrument settings (k large; potentially k growing with n) introduce distinct pathologies: 2SLS bias, weak-ID diagnostics that “look strong” (e.g., large first-stage F) while identification is effectively weak, and standard weak-IV tests that rely on fixed-k approximations may fail.
- Recent theory and software emphasize **many-weak-IV robust tests and pretests** (e.g., Mikusheva–Sun) and practical guidance on when jackknife-based Wald tests are reliable. 
- In 2025, the **ivmodels** Python package exists and explicitly implements AR, LM, (conditional) LR tests/confidence sets, reduced-rank diagnostics, J tests, and k-class estimators (TSLS/LIML). It is currently the strongest directly comparable Python effort for weak-IV inference. 
- The gap is not “any IV” (Python has IV estimators), but **weak-ID robust inference with rigorous diagnostics + set-valued CIs + numerically stable inversion + benchmark parity checks**.
- A best-in-class package must be:
  - **Identification-first**: user must be guided to specify (and the library must record) assumptions (weak-ID robust vs strong-ID; iid vs cluster vs HAC).
  - **Numerically stable**: handle near-singular matrices, rank deficiency, multiple solutions in inversion, and non-interval CS shapes.
  - **Benchmark-driven**: every method validated against canonical references and cross-checked against Stata/R implementations where possible.
  - **Transparent**: return objects that show failure modes, warnings, and diagnostics; prevent misinterpretation.
- **Scope boundaries**: your package is about **weak relevance / many instruments** under standard exclusion exogeneity, not (initially) about invalid instruments (exclusion violations), though the library can include “stress test” benchmarks or warnings clearly labeled out-of-scope.

**North Star principles**
- **Identification clarity**: always specify the model, endogenous set, instrument set, and covariance regime.
- **Invariance and robustness**: implement procedures that are valid under weak ID and (where supported) heteroskedasticity/cluster structures.
- **Diagnostics-first**: report strength measures (with caveats), many-instrument warnings, and conditioning.
- **Numerical stability**: use stable decompositions, avoid gratuitous matrix inverses, implement robust root-finding/inversion.
- **Reproducibility**: config+seed registries, deterministic linear algebra, versioned benchmarks.

**High-leverage improvements to the current repo (`gsaco/ivrobust`)**
- The current repo appears **minimal / early-stage** relative to a research-grade Python package: limited docs, limited tests/benchmarks, and no visible parity harness or “assumption cards.” (Detailed repo audit is below.)
- Biggest leverage:
  - Introduce **`src/` layout + typed data contracts** (`IVData`, `ClusterSpec`) and a stable API.
  - Implement **AR test + CI inversion** robustly (including set-valued CIs and diagnostics) as v0.1 “gold core”.
  - Add **benchmark harness** with canonical weak-IV DGPs and golden comparisons vs Stata `estat weakrobust` and/or `weakiv` when feasible. 
  - Add **docs and tutorial notebooks** that teach interpretation and highlight failure modes.
  - Add strong **CI + test suite**: unit tests, integration tests, numerical regression tests, and simulation “size/coverage” tests.

---

# 2. What Counts as Weak-IV / Many-IV Robust Inference? (Definitions + Taxonomy)

## 2.1 Problem Taxonomy

### Core linear IV objects and targets
Let:
- Outcome: \(y \in \mathbb{R}^n\)
- Endogenous regressors: \(X_{e} \in \mathbb{R}^{n \times p}\)
- Exogenous regressors (controls): \(X_{x} \in \mathbb{R}^{n \times q}\)
- Instruments: \(Z \in \mathbb{R}^{n \times k}\), with \(Z\) including excluded instruments and often also including \(X_x\) (depending on convention).

Structural equation (linear IV):
\[
y = X_e \beta + X_x \gamma + u,\quad \mathbb{E}[Z_i u_i]=0
\]
Target: \(\beta\) (and possibly \(\gamma\)), plus linear contrasts and joint hypotheses.

### Taxonomy tree (crisp)
**A. Identification regime**
1. **Strong ID (classical)**: concentration parameter large; standard 2SLS asymptotics yield approximate normality.
2. **Weak ID (fixed-k weak instruments)**: first-stage coefficients local-to-zero; standard Wald inference fails; weak-IV robust tests (AR/CLR/LM/LR families) provide correct size.
3. **Many instruments (k large / k→∞)**:
   - **Many-IV strong-ish**: instruments individually weak but collectively informative; bias and variance issues depend on \(k/n\).
   - **Many-weak-IV**: both weak relevance and many instruments; requires specialized asymptotics and often jackknife/cross-fit constructions (e.g., Mikusheva–Sun). 

**B. Model shape**
1. **Just-identified** (k = p if only excluded instruments): AR=CLR equivalences; inference simplifies.
2. **Over-identified** (k > p): power comparisons and over-ID tests require care under weak ID.

**C. Endogeneity dimension**
1. **Single endogenous regressor** (p = 1): most mature theory; CLR well developed.
2. **Multiple endogenous regressors** (p > 1): robust tests exist but computation and diagnostics are more complex; “weak in some directions” matters.

**D. Error structure / covariance regime**
1. **Homoskedastic i.i.d.**: classic exact/conditional results for CLR.
2. **Heteroskedastic i.i.d.**: robust covariance; GMM-based robust tests (Finlay–Magnusson generalization; Kleibergen GMM LR/LM). 
3. **Clustered**: cluster-robust VCE; validity of weak-IV tests depends on asymptotic regime (fixed clusters vs many clusters). Stata’s weakrobust infrastructure supports clustered VCE for its tests, but rigorous theoretical boundaries should be explicit in docs. 
4. **Time series / HAC**: even more delicate; some robust tests exist in IV/HAC literature; scope should be explicit and possibly deferred unless theory is implemented carefully.

**E. Estimators vs inference**
- **Estimators**: 2SLS, LIML, Fuller, k-class, JIVE.
- **Weak-IV robust inference**: tests/CIs that remain valid regardless of weakness.
- **Robust SE** alone ≠ weak-IV robustness.

## 2.2 Targets of Inference (what is identified)

- The structural coefficient vector \(\beta\) is **identified** under:
  1) **Relevance/rank**: \(\mathrm{rank}(\mathbb{E}[Z_i X_{e,i}']) = p\) (or conditional analog after partialling out \(X_x\)),
  2) **Exogeneity**: \(\mathbb{E}[Z_i u_i]=0\).

- Under weak instruments, \(\beta\) may be **weakly identified**: relevance holds but is near-violated in finite samples or local-to-zero asymptotics. Robust tests target valid inference even when the “distance” from non-identification is small.

- Without relevance (rank failure), \(\beta\) is **not identified**; weak-IV robust tests typically yield confidence sets that are extremely wide or uninformative (often the whole real line in scalar case), appropriately reflecting lack of information.

- Subvector inference (e.g., one component of \(\beta\) when p>1) can be approached via projection or specialized methods; scope should state what is supported.

---

# 3. Literature Review (Up-to-date, cited, organized)

## 3.1 Canonical Surveys / Tutorials / Foundations

Below are foundations and surveys with direct implications for package design.

1. **Andrews, Stock & Sun (2019)**: Comprehensive review of weak instruments in linear IV with discussion of heteroskedastic/cluster settings, diagnostics, and robust inference procedures; a primary design reference for “what to implement and why.”   
2. **Montiel Olea & Pflueger (2013)**: Introduces the “effective F” statistic (robust to heteroskedasticity, clustering, etc.) as a principled weak-instrument pretest under a bias criterion; key for diagnostics reporting beyond naive F.   
3. **Moreira (2003)**: Develops the conditional likelihood ratio framework yielding CLR tests with exact similarity under homoskedastic normal models; foundational for CLR implementation and inversion.   
4. **Andrews, Moreira & Stock (2006/2007 line)**: Develop computational and optimality results for CLR and related conditional tests; provides algorithmic guidance for computing CLR p-values and CIs.   
5. **Kleibergen (2002)**: Develops pivotal statistics for IV structural parameter tests (e.g., K-statistics) and provides foundations for LM/score-type robust inference.   
6. **Kleibergen (2007)**: Generalizes weak-IV robust score (LM) and LR statistics to multiple parameters and unrestricted covariance matrices, connecting to GMM; relevant for robust/higher-dimensional inference design.   
7. **Mikusheva (2010)**: Details robust confidence sets obtained by inverting weak-IV robust tests (AR/CLR/LM), discusses possible CS shapes, and provides practical algorithms for CLR inversion; essential for numeric inversion design.   
8. **Finlay & Magnusson (2009)**: Minimum distance approach for weak-IV robust tests in general IV models and heteroskedastic settings; directly referenced by Stata’s `estat weakrobust` for heteroskedastic CLR generalization.   
9. **Sanderson & Windmeijer (2016)**: Weak-instrument F-testing in models with multiple endogenous variables (near rank reduction rather than near rank zero), shaping diagnostics for p>1 endog.   
10. **Windmeijer (2025, published)**: Extends and clarifies robust F-statistic weak-instrument testing beyond effective F; potentially affects diagnostics and reporting in modern practice.   
11. **Mikusheva & Sun (2022, Review of Economic Studies)**: Develops inference with many weak instruments under heteroskedasticity, including jackknife AR and a pretest for Wald/JIVE validity; key for many-IV robust inference and benchmark design.   
12. **Mikusheva & Sun (2023/2024 working paper / survey)**: “Weak identification with many instruments” survey and extensions including jackknife LM; sets the modern frontier for many-IV robust inference.   
13. **Londschien (2025)**: A statistician’s tutorial explicitly tied to Python `ivmodels`, providing an applied-friendly exposition and anchoring what is already implemented in Python.   

*(There are many other important papers; these are the most “package-design load-bearing” based on current tooling and the goal scope.)*

## 3.2 Methods Inventory (Core Table)

> Note: The table is long; it is meant to be “engineering actionable”: each row can become an “assumption card” + “procedure card” + tests/benchmarks.

| Family / Method | Setting | Identification assumptions | Test/CI construction sketch | What it identifies | Diagnostics | Failure modes | Compute/numerics | Canonical papers | Best modern refs | Implementation notes |
|---|---|---|---|---|---|---|---|---|---|---|
| **Anderson–Rubin (AR) test** | Linear IV; p≥1; k≥p; robust under weak ID | Exogeneity \(E[Z'u]=0\); model linear; works even if relevance weak | Test \(H_0:\beta=\beta_0\) via reduced-form regression of \(y-X_e\beta_0\) on instruments/controls; statistic is quadratic form of moment conditions; invert over \(\beta_0\) for CS | Tests structural β (or joint restrictions) | Report AR stat, p-value, CS shape; condition numbers; rank diagnostics | Low power when strong ID & over-ID; can yield whole-line CS under very weak ID | Needs stable projections; inversion may yield unions; gridding when no closed form | Anderson & Rubin (1949); weak-IV development in modern literature | Andrews–Stock–Sun (2019); Mikusheva (2010) | For homoskedastic/just-ID, inversion can be analytic; heteroskedastic over-ID often needs grid search; return set-valued CS |
| **CLR test / CS** | Best known for p=1, homoskedastic; extensions exist | Exogeneity; homoskedastic normal for exact similarity; otherwise asymptotic variants | Compute LR-type statistic conditional on sufficient statistics; p-value from nonstandard distribution; invert over β0 for CS | β scalar (often), extended variants for p>1 exist but complex | CLR stat/p-value; compare to AR; report if using generalized CLR | If assumptions violated, p-values may be invalid; numeric root finding for inversion | Nonstandard distribution computation; careful handling of boundary; inversion can be multi-interval | Moreira (2003); Andrews–Moreira–Stock (2006/2007) | Mikusheva (2010); Stata implementation notes; Finlay–Magnusson for robust generalization | Implement analytic p-value algorithm for p=1 homoskedastic; for robust, use simulation/approx methods as in Stata |
| **LM / Score / K-type tests** | IV/GMM; can handle p>1 | Exogeneity; weak-ID robust by construction of pivotal/score stats | Test based on score at β0; can be equivalent to Kleibergen’s K statistic; invert for CS | β (possibly vector) | Report LM stat/p-value; reduced rank tests | Can have poor power in some regimes; depends on covariance estimation | Requires stable covariance/inverses; may need generalized eigenvalue problems | Kleibergen (2002) | Kleibergen (2007); Mikusheva (2010) | Provide both homoskedastic and robust versions where theory supports |
| **GMM quasi-LR / robust CLR generalizations** | Heteroskedastic IV/GMM | Moment exogeneity + valid covariance estimator | Construct quasi-LR statistic; p-values via asymptotic approximations or simulation | β | Report choice of covariance; sensitivity | Wrong covariance regime → size distortions | Simulation intensive; many local minima in inversion | Finlay & Magnusson (2009) | Stata `estat weakrobust` notes Kleibergen (2007) equivalence | Provide simulation-based p-values with reproducible RNG |
| **Many-IV robust AR (jackknife / leave-one-out)** | k large; heteroskedastic allowed | Many-IV asymptotics (k→∞ with n); exogeneity | Use leave-one-out quadratic form; estimate variance via cross-fit; statistic ~ N(0,1) under H0; invert for CS | β (scalar and potentially vector depending on theory) | Report k/n; many-IV warnings; Fe pretest outputs | Implementation complex; variance estimation critical; can have poor power if naive | O(nk) or worse; must optimize linear algebra; avoid explicit n×n matrices | Mikusheva & Sun (2022) | Manyweakiv Stata package; Mikusheva–Sun survey (2023/2024) | Implement as separate module w/ explicit asymptotic regime; careful performance engineering |
| **Many-IV pretests / Fe** | Many instruments, assessing Wald validity | As in Mikusheva–Sun; bias/size control criteria | Compute Fe; compare to cutoffs to decide whether Wald/JIVE is safe; avoids naive F | Diagnostics (not inference itself) | Fe statistic + threshold + recommended path | Misuse as “instrument strength” without context; regime mismatch | Needs careful calculation and documentation | Mikusheva & Sun (2022) | manyweakivpretest (Stata) | Implement as diagnostic with explicit disclaimers |
| **First-stage F / partial R²** | Standard strong-ID diagnostics | Strong ID approximations | Compute first-stage F; partial R²; Sanderson–Windmeijer conditional F for p>1 | Diagnostics only | Report with warnings; compare to effective F | Misleading under heteroskedasticity/many instruments; Stock–Yogo not robust | Straightforward | Stock–Yogo (2005); Staiger–Stock line | Montiel Olea–Pflueger; Sanderson–Windmeijer; Windmeijer robust F | Provide “do not overinterpret” messaging |
| **Overidentification tests (J, K-J, LIML J)** | Overidentified IV | Exogeneity + relevance; interpretation depends on ID | Compute J tests; but weak ID changes distribution and meaning | Tests over-ID restrictions | Report but warn under weak ID | Under weak ID can overreject/mislead | Standard quadratic forms | Hansen J; extensions | weakiv package supports K-J tests | Make validity region explicit |

**Key point for package design:** each row must become a *procedure implementation* with explicit assumptions, plus a *result object* that reports diagnostics and failure modes.

## 3.3 “Assumption-to-Procedure Map”

Below is a high-level validity matrix. “Valid” means asymptotically correct size/coverage under the stated regime; “Maybe” means depends on additional conditions; “No” means not generally supported.

| Assumptions / Regime | AR (classic) | CLR (classic) | LM/K (classic) | Robust CLR / QLR | Many-IV jackknife AR | Wald/2SLS t-tests |
|---|---:|---:|---:|---:|---:|---:|
| Fixed k, homoskedastic, p=1 | Valid | Valid (exact/conditional) | Valid | N/A | Not needed | Only if strong ID |
| Fixed k, homoskedastic, p>1 | Valid | Nontrivial | Valid-ish | N/A | Not needed | Only if strong ID |
| Fixed k, heteroskedastic | Valid with robust covariance (AR-GMM form) | Classic CLR not exact; robust variants exist | Valid w/ robust covariance if implemented correctly | Valid per Finlay–Magnusson/Kleibergen frameworks | Not needed | Not weak-ID robust |
| Clustered (many clusters) | Potentially valid if cluster-robust moments + asymptotics | Robust variants depend; implementation-specific | Potentially valid | Potentially valid | Many-IV + clusters is frontier | Not weak-ID robust |
| Many instruments (k large) | Fixed-k AR may not be valid | Fixed-k CLR may not be valid | Fixed-k LM may not be valid | unclear | Valid per Mikusheva–Sun (under their regime) | Often fails badly |

The package should make this map explicit in docs (“assumption cards”) and enforce it via `AssumptionSpec`.

---

# 4. State of the Ecosystem: Existing Packages & Gaps (Deep scan, cited)

## 4.1 Existing Python libraries touching IV inference

### ivmodels (Python)
- **Scope**: k-class estimators (TSLS/LIML), weak-IV tests and confidence sets including AR, LM, (conditional) LR, Wald; auxiliary reduced-rank tests and J tests. 
- **Strengths**: explicit weak-IV robust methods, documented, PyPI distribution; tied to a 2025 tutorial paper. 
- **Gaps vs your goal**: unclear (from surface) whether it fully covers cluster/HAC regimes, many-IV asymptotics, and benchmark parity harness vs Stata/R. It also may have design choices you might want to differ from (e.g., data contracts, assumptions registry).

### DoubleML (Python) – weak-IV robust CIs in ML-IV context
- DoubleML provides “robust IV” examples for weak instruments in its causal ML setting, but it is not a dedicated linear weak-IV inference suite. 

### linearmodels / statsmodels (Python)
- These libraries implement IV estimators and robust covariances, but (as of my latest check up to 2025) do not provide full AR/CLR/LM weak-ID robust inference as first-class workflows in the same way as Stata’s weakrobust/weakiv or ivmodels. **(Uncertain)**: verify their current feature sets before claiming absence; they evolve. *(I did not find primary sources in this quick scan.)*

### Your repo: `gsaco/ivrobust`
- I could not retrieve a full structured API and extensive documentation from the repository snapshot used in this review; the repo appears early-stage relative to the target “release-ready scientific package.”
- **Actionable conclusion**: treat your repo as a *seed* and migrate toward a full package structure (see Blueprint).

## 4.2 Reference implementations in other ecosystems

### Stata: `estat weakrobust` (built-in postestimation for ivregress)
- Stata’s ivregress postestimation manual states:
  - In overidentified homoskedastic models, it reports **Moreira (2003) CLR** by default.
  - Under non-homoskedastic VCE, it uses **Finlay–Magnusson (2009) generalization**, noted as equivalent to Kleibergen’s GMM quasi-LR in the linear IV case.
  - Confidence intervals are obtained by **test inversion**; AR can be inverted analytically in some cases; otherwise gridding is used. 
- This is an ideal “gold standard” for output parity checks in Python.

### Stata: `weakiv` (SSC)
- The `weakiv` command (SSC) supports AR, CLR, LM K test, J test, K-J combinations, with MD/Wald and LM versions, intended for weak-IV robust inference. 

### Stata: `manyweakiv` (Sun)
- `manyweakiv` and `manyweakivpretest` implement many-weak-IV robust inference and strength assessment based on Mikusheva–Sun frameworks; valid under heteroskedasticity and many instruments per the repo documentation. 

### R: `ivmodel` / other
- The R package `ivmodel` provides AR and CLR confidence intervals robust to weak instruments (among other features), making it another plausible parity target. 
- R’s `estimatr::iv_robust` provides IV estimation with robust VCE and some diagnostics, but it is not a full weak-IV robust inference suite; discussions indicate diagnostics limitations with fixed effects. 

## 4.3 Gap analysis

What’s missing in Python (relative to Stata’s weakrobust/weakiv and modern many-weak-IV work):
- **End-to-end weak-ID robust CIs** (especially set-valued, inversion-based) with reliable numerics and full reporting.
- **Cluster/HAC support** with explicit theoretical validity boundaries and careful implementation (avoid “it runs” ≠ “it’s valid”).
- **Many-IV robust inference** (jackknife AR/LM and pretests) is not mainstream in Python outside niche efforts; Stata has `manyweakiv`. 
- **Diagnostics-first UX**: effective F, conditional F, reduced-rank tests, and warnings integrated into result objects.
- **Benchmark harness**: reproducible simulation tests of size/power/coverage plus golden parity checks vs Stata/R/Matlab implementations.

---

# 5. Package Vision: What “Best Complete Weak-IV Package” Means

**Definition of “complete” (for your scope)**
1. **Method coverage**
   - Fixed-k weak-IV robust inference:
     - AR tests + inversion CIs (scalar and joint tests).
     - CLR tests + inversion CIs (at least p=1 homoskedastic exact; robust generalizations where defensible).
     - LM/Score/K tests + inversion CIs (at least for common cases).
   - Many-IV robust inference:
     - Many-weak-IV AR (jackknife / leave-one-out variants) + associated variance estimators.
     - Many-IV “strength” pretests (Fe and/or analogous) and reporting.
   - Diagnostics:
     - First-stage and reduced-rank diagnostics.
     - Effective-F and modern robust strength reporting.
     - Many-IV warnings and estimator guidance (LIML/Fuller/JIVE contexts).
2. **Numerical stability**
   - Stable projections and decompositions.
   - Robust inversion logic for set-valued CIs.
   - Deterministic simulation p-value computation when required.
3. **Benchmark coverage**
   - Size/power/coverage curves across weak-ID strength parameters, heteroskedasticity, clustering, many instruments.
   - Golden comparisons vs Stata `estat weakrobust`, `weakiv`, and `manyweakiv` where possible.
4. **Documentation**
   - Assumption cards / procedure cards.
   - Tutorials with interpretation guardrails.
   - Reproducible benchmark recipes.

**Explicit non-goals (early)**
- “Invalid IV” (exclusion violations) robust inference is a different literature (e.g., sensitivity analysis, partial identification, MR, etc.). You can include stress-test simulations, but do not claim validity unless you implement that theory.

---

# 6. Blueprint: Package Architecture & Folder Structure (NO repo creation, but fully specified plan)

## 6.1 Proposed top-level tree

Target state (proposed):

```text
ivrobust/
  pyproject.toml
  README.md
  LICENSE
  CITATION.cff
  CHANGELOG.md
  CONTRIBUTING.md
  CODE_OF_CONDUCT.md

  src/ivrobust/
    __init__.py
    _version.py

    data/
      __init__.py
      ivdata.py          # IVData, validation, pandas adapters
      clusters.py        # ClusterSpec, cluster validation
      weights.py         # WeightSpec, WLS conventions
      design.py          # IVDesign: cached projections, residualizers
      checks.py          # rank/conditioning checks

    linalg/
      __init__.py
      projections.py     # stable residualization, QR/SVD helpers
      solvers.py         # safe solve, pinv, symmetrize, cond diagnostics
      quadforms.py       # compute quadratic forms stably

    estimators/
      __init__.py
      kclass.py          # TSLS, LIML, Fuller (k-class base)
      jive.py            # JIVE / UJIVE (optional for many-IV diagnostics)
      results.py         # EstimatorResult objects

    cov/
      __init__.py
      hc.py              # HC0-HC3
      cluster.py         # one-way / two-way cluster (start w/ one-way)
      hac.py             # HAC/Newey-West (optional/phase)
      sandwich.py        # shared sandwich patterns
      validity.py        # notes and checks: what is supported where

    weakiv/
      __init__.py
      ar.py              # AR stats, p-values, inversion
      clr.py             # CLR stats/p-values/inversion
      lm.py              # LM/K stats/p-values/inversion
      inversion.py       # generic inversion engine (grid+root find)
      sets.py            # set-valued CS representations
      results.py         # WeakIVTestResult, ConfidenceSet objects

    manyiv/
      __init__.py
      jackknife_ar.py    # Mikusheva–Sun style AR; variance estimators
      pretests.py        # Fe and other many-IV diagnostics
      notes.py           # explicit assumptions & references

    diagnostics/
      __init__.py
      strength.py        # F, effective F, partial R2, cond F
      reduced_rank.py    # Anderson (1951), KP, SVD-based rank tests
      overid.py          # J tests, LIML J
      numerics.py        # condition numbers, warnings
      reporting.py       # tables, summaries

    plots/
      __init__.py
      cs.py              # plot confidence sets (interval unions, etc.)
      power.py           # benchmark visualization
      diagnostics.py

    benchmarks/
      __init__.py
      dgp.py             # DGP generators (weak, many, heterosked, clusters)
      runners.py         # simulation harness
      metrics.py         # size/power/coverage/length
      registry.py        # config snapshots + results storage

    utils/
      __init__.py
      rng.py             # reproducible RNG utilities
      typing.py
      warnings.py

  tests/
    test_data_contracts.py
    test_linalg.py
    test_estimators.py
    test_ar.py
    test_clr.py
    test_lm.py
    test_manyiv_ar.py
    test_diagnostics.py
    test_benchmarks_smoke.py

  examples/
    README.md
    minimal_ar_ci.py
    card_1995_schooling.ipynb
    anglist_krueger_manyiv.ipynb
    clustered_example.ipynb

  docs/
    mkdocs.yml (or docs/ + config)
    index.md
    assumptions/
      ar.md
      clr.md
      lm.md
      manyiv_ar.md
      effective_f.md
    procedures/
      ar_ci.md
      clr_ci.md
      lm_ci.md
      diagnostics.md
    tutorials/
      01_iv_basics.md
      02_weak_iv_why.md
      03_ar_confidence_sets.md
      04_clr_workflow.md
      05_many_iv.md
      06_clustered.md
      07_benchmarks.md
    reference/
      api.md
      numerics.md
      reproducibility.md
```

### Mapping from current repo → target tree (minimal disruption)
- Keep repo name `ivrobust`.
- Introduce `src/ivrobust/` and gradually migrate existing code into the appropriate modules.
- Preserve any existing public API functions by re-exporting through `ivrobust/__init__.py`, but mark them as deprecated if necessary.
- Add docs scaffolding and tests early, even before full method coverage.

## 6.2 Data Contracts (critical)

### Canonical dataset object
**`IVData`** (immutable-ish, validated container)

```python
IVData(
  y: ArrayLike[n],
  X_endog: ArrayLike[n, p],
  X_exog: ArrayLike[n, q],
  Z: ArrayLike[n, k],
  weights: Optional[ArrayLike[n]] = None,
  clusters: Optional[ArrayLike[n] or ArrayLike[n, g]] = None,
  time: Optional[ArrayLike[n]] = None,    # for HAC/time series extensions
  metadata: dict = {}
)
```

**Validation rules**
- `y` length n matches all others.
- Enforce 2D matrices for X/Z (even if p=1).
- Handle missingness: either forbid NaNs by default or provide `drop_missing="listwise"` option that returns a new `IVData` and logs action.
- Rank checks:
  - `rank([X_exog, Z])` ≥ `q + p` for identification (after partialling out exog).
  - compute condition numbers and singular values for warnings.
- Intercept conventions:
  - By default, do not auto-add intercept; require user to include in `X_exog` or offer `add_intercept=True` helper explicitly.
- Weights:
  - Define whether weights are analytic/frequency/probability; for v0.1, support analytic weights (WLS) with clear formulas and disclaimers.
- Clusters:
  - `clusters` may be 1D (one-way) or 2D (multi-way). Start with one-way cluster robust; multi-way later.
- Memory layout:
  - Convert inputs to `np.asarray(..., order="F")` or consistent layout for BLAS friendliness; document.

### `IVDesign` (cached linear algebra)
Create an object that caches:
- residual-maker for exog: \(M_{X_x}\)
- projected instruments after partialling out exog: \(Z_\perp = M_{X_x} Z\)
- projected endog: \(X_{e,\perp} = M_{X_x} X_e\)
- key cross-products: \(Z'Z\), \(Z'X_e\), etc.
- stable QR/SVD decompositions for repeated computations in inversion.

## 6.3 API Design Principles

- **Separate**:
  1) model/data (`IVData`, `IVDesign`),
  2) estimation (`TSLS`, `LIML`, etc.),
  3) inference (`ARTest`, `CLRTest`, `LMTest`, `ConfidenceSet` inversion),
  4) covariance choices (`HC`, `Cluster`, `HAC`) and their validity.
- Provide an explicit `AssumptionSpec`:
  - `id_regime = {"strong", "weak_fixed_k", "many_weak_iv"}`
  - `error = {"homoskedastic", "heteroskedastic", "cluster", "hac"}`
  - `asymptotics = {"fixed_k", "many_k"}`
  - `notes` and `citations`.
- Result objects must include:
  - statistic, p-value, df/critical value info,
  - confidence set (possibly non-interval),
  - inversion diagnostics (grid density, root-finding status),
  - numerical warnings and conditioning metrics,
  - reproducibility metadata (seed, version, config).
- Provide small composable functions and also “workflow” entry points:
  - `ivrobust.ar_test(data, beta0, cov="HC1")`
  - `ivrobust.ar_confidence_set(data, alpha=0.05, grid=..., cov="HC1")`
  - but back them by classes for extensibility.

---

# 7. Diagnostics & Reporting: Required First-Class Features

## Diagnostics Playbook

### 7.1 Instrument strength reporting (with limitations)
**What to compute**
- Classical first-stage F-statistics (and partial R²) for each endogenous regressor.
- Robust/effective strength:
  - **Effective F** (Montiel Olea–Pflueger) for single endogenous regressor and robust covariance regimes where applicable. 
  - For multiple endog: conditional F tests à la Sanderson–Windmeijer, with explicit interpretation. 
- Reduced-rank tests:
  - KP-type rank tests / Anderson reduced rank; in ivmodels these appear as “Anderson’s test of reduced rank”. 

**Outputs**
- `StrengthDiagnostics` object:
  - `first_stage_F`, `partial_R2`, `effective_F` (if supported),
  - `k`, `n`, `k/n`, `rank_ZX`, `min_singular_value`,
  - recommended caution flags.

**Pitfalls to guard against**
- “F>10” is not a universal guarantee; robust/many-IV settings can violate it.
- Under many instruments, large classical F can coincide with weak identification; must warn.

### 7.2 Many-instrument warnings and recommended alternatives
- Detect large k relative to n (e.g., k/n > 0.1 or user-configurable thresholds) and warn that:
  - 2SLS can be biased,
  - weak-IV fixed-k tests may be invalid,
  - recommend many-IV tools (jackknife AR, Fe pretest) if implemented. 

### 7.3 Numerical conditioning reports
- Always compute and store:
  - condition numbers of \(Z'Z\), \(X'P_Z X\), etc.
  - smallest singular values.
- Provide `NumericalDiagnostics`:
  - warnings for near singularity,
  - fallback strategies used (pinv, ridge).
- Avoid silently switching to pseudoinverse without warning.

### 7.4 Sensitivity to covariance choices
- Provide a function to recompute inference under different covariance regimes:
  - `result.sensitivity(cov=["HC0","HC1","cluster"])`
- But document that covariance regime changes **assumptions**; do not present as purely “robustness”.

### 7.5 Interpretation guardrails
- Explicitly surface messages like:
  - “Robust SE does not fix weak-IV bias; use AR/CLR/LM.”
  - “Overidentification tests are not reliable indicators of validity under weak ID.”
  - “Confidence set may be disjoint or unbounded; this is expected under weak ID.”

---

# 8. Benchmark Suite Design (Ambitious but Realistic)

## 8.1 Benchmark categories (DGP families)

1. **Canonical weak-IV (fixed k)**
   - Single endogenous regressor:
     - Vary concentration parameter / first-stage strength.
     - Compare AR vs CLR vs LM size/power/coverage.
   - Multiple endogenous regressors:
     - Weak in one direction, strong in another.
     - Near-rank-reduction designs.

2. **Many instruments**
   - k increasing with n (e.g., k = floor(n^a) or k = floor(c n)).
   - Designs where classical F is misleading.
   - Compare fixed-k tests vs many-IV robust jackknife AR.

3. **Heteroskedasticity and clustering**
   - Heteroskedastic designs (variance depends on Z or X).
   - Clustered designs with varying number of clusters (G) and cluster sizes.
   - Document validity assumptions (many clusters asymptotics).

4. **Optional invalid IV stress tests** (clearly out-of-scope)
   - Simulate exclusion violations to show that weak-IV robust tests don’t address invalidity.

## 8.2 Metrics
- Size (type I error) under weak-ID.
- Power curves vs local and distant alternatives.
- Coverage of confidence sets; include non-interval sets.
- Expected length/volume of CS.
- Numerical failure rates (non-convergence, inversion failure).
- Runtime and scaling (n, k).

## 8.3 Reproducibility standards
- Each benchmark run saves:
  - RNG seed, config, package version, BLAS/LAPACK info.
- Deterministic mode:
  - stable linear algebra paths (QR/SVD), fixed tolerances.
- Cross-check harness:
  - For select DGPs, generate data and export to Stata/R scripts for parity of AR/CLR outputs.

---

# 9. Documentation Plan (What to write, not how to code)

## Assumption Cards (template)
Each `docs/assumptions/<proc>.md` should include:
- Procedure name.
- Validity regime (weak fixed-k, many-IV, etc.).
- Required assumptions (exogeneity, linearity, covariance structure).
- What is robust to (weak ID, heteroskedasticity, clustering).
- What is not robust to (invalid IV, too-few clusters).
- References (author-year + links).

## Procedure Cards (template)
Each `docs/procedures/<proc>.md`:
- What it tests.
- Inputs and outputs.
- Confidence set shape possibilities.
- Diagnostics to report.
- Common pitfalls.
- Examples snippet.

## Tutorials
- IV basics + why weak-IV matters.
- AR confidence sets (including disjoint/unbounded).
- CLR workflow; differences vs AR.
- Many-IV and jackknife AR; pretests.
- Clustered data: what is / isn’t supported.
- Benchmark replication tutorial.

## Reference docs
- API reference pages.
- Numerics notes (projection stability, inversion algorithms).
- Reproducibility notes.

## Troubleshooting guides
- Singular matrices.
- Empty or whole-line confidence sets.
- Grid inversion issues.
- Interpretation pitfalls.

---

# 10. Engineering Quality Gates (What makes it trustworthy)

## Testing ladder
1. Unit tests:
   - linear algebra utilities (projections, solves).
   - covariance estimators (HC/cluster).
2. Integration tests:
   - TSLS vs known closed-form.
   - AR test invariance to instrument transformations.
3. Golden tests:
   - compare selected outputs to saved reference numbers (from Stata/R/ivmodels).
4. Benchmark regression tests:
   - small simulation runs to catch drift in size/coverage.

## Type checking / linting / docs build
- Type hints across public API.
- `ruff`/`black` (or ruff format) for style.
- `mypy` for types.
- Doc build as CI job.

## Versioning and deprecation
- SemVer.
- Deprecation warnings with 2 minor releases grace period.

## Numerical QA
- Use stable decompositions (QR/SVD) instead of direct inverses.
- Centralize tolerances; expose `NumericsSpec`.
- Provide deterministic seeding and random streams.

## Governance
- LICENSE, CITATION.cff, CONTRIBUTING, CODE_OF_CONDUCT.
- Reference policy: how to cite papers; ensure docs are grounded.

---

# 11. Roadmap (Staged, with deliverables and acceptance criteria)

## Phase 0: Literature + design freeze artifacts
Work items (acceptance criteria in parentheses)
1. Write Assumption Card templates (docs compile).
2. Write Procedure Card templates (docs compile).
3. Freeze `IVData` contract and `AssumptionSpec` (typed, documented).
4. Define benchmark DGP spec (doc + config schema).
5. Identify reference implementations for parity (Stata weakrobust/weakiv, ivmodels) and specify parity tests. 
6. Choose numerical primitives (QR/SVD policy) and tolerances (documented).
7. Establish CI pipeline skeleton (lint, type, tests).
8. Add CONTRIBUTING and governance docs.

## Phase 1: Minimal research-grade core + AR inference + diagnostics + golden tests
1. Implement `IVData` + validation (unit tests).
2. Implement projections/residualization utilities (unit tests).
3. Implement TSLS + LIML baseline estimators (integration tests vs known results).
4. Implement HC0–HC3 covariance for reduced-form moments (unit tests).
5. Implement one-way cluster covariance (unit tests + warnings for small clusters).
6. Implement AR test (scalar and joint) + analytic inversion where possible; grid otherwise.
7. Implement confidence set object that can represent unions of intervals.
8. Implement diagnostics (F, partial R², rank checks, conditioning).
9. Add minimal examples (script + notebook).
10. Add golden parity: replicate a few Stata weakrobust outputs (manual/CI skip if Stata unavailable).

Acceptance: `pip install -e.` works; `pytest` passes; docs build; example runs; AR CI inversion returns set-valued CS with diagnostics.

## Phase 2: CLR + LM/LR families + many-IV support + expanded covariance support
1. Implement CLR (p=1, homoskedastic exact) with validated p-value computation.
2. Implement CLR inversion per Mikusheva algorithm (unit tests on known shapes). 
3. Implement LM/K tests (p=1 and p>1 where supported).
4. Implement robust CLR/QLR variants (simulation-based p-values) aligned with Finlay–Magnusson and Stata behavior. 
5. Many-IV module:
   - jackknife AR + variance estimator variants (Mikusheva–Sun).
   - Fe pretest and reporting. 
6. Benchmark expansions: many-IV and heteroskedastic/clusters.

Acceptance: CLR/LM tests validated on canonical DGPs; many-IV AR passes size benchmarks; docs include clear assumption boundaries.

## Phase 3: Benchmark suite polish, cross-ecosystem parity checks, docs, stability, adoption
1. Comprehensive benchmark suite with plots and reports.
2. Cross-ecosystem parity harness:
   - generate data and compare to Stata `estat weakrobust` and `weakiv` outputs where feasible. 
3. Add more tutorials and troubleshooting.
4. Add performance profiling and optimization for many-IV routines.
5. API stability review and v1.0 readiness checklist.
6. Release automation: build wheels, publish docs, version tags.

Acceptance: stable API; benchmarks reproducible; parity checks documented; release artifacts build cleanly.

---

# 12. Risk Register (Be brutally honest)

## Identification risks
- Users confuse weak instruments with invalid instruments.
  - Mitigation: assumption cards and explicit guardrails; optional “invalid IV” stress test examples labeled out-of-scope.
- Users misinterpret effective F / pretests as proof of validity.
  - Mitigation: documentation and warnings; always report limitations.

## Methodological risks
- Using covariance estimators outside their theoretical validity (few clusters, HAC mis-specification).
  - Mitigation: require `AssumptionSpec`, warn on small number of clusters, and document asymptotic regimes.
- Overidentification tests misused under weak ID.
  - Mitigation: “do not interpret as instrument validity” warnings and references.

## Numerical risks
- Inversion yields disjoint/unbounded sets and users think it’s a bug.
  - Mitigation: explicit set objects + plotting + tutorial.
- Near-singular matrices cause unstable stats.
  - Mitigation: stable decompositions, ridge options with explicit warnings, numeric diagnostics.

## Engineering risks
- Scope explosion (supporting every covariance regime and many-IV frontier).
  - Mitigation: staged roadmap; keep many-IV module separate; “supported/experimental” flags.
- Maintenance burden of simulation benchmarks and parity harness.
  - Mitigation: keep benchmark suite modular; small smoke tests in CI; full benchmarks in nightly workflow.

## Adoption risks
- API too complex for applied users.
  - Mitigation: provide “quickstart” wrappers and defaults; keep advanced controls accessible but optional.
- Users expect magic “weak-IV fix” without understanding.
  - Mitigation: docs that teach; warnings that force awareness.

---

## Repo review note

I did not embed a line-by-line dump of your repo because the web scan here focused on the wider landscape and referenced implementations; in the next iteration, the repo audit should list:
- current folder structure,
- current public API (functions/classes),
- test coverage,
- CI workflows,
- docs/examples status,
and map them directly to the target tree above.

However, the upgrade plan above is designed to be **minimally disruptive**: introduce the architecture in parallel, then migrate existing code gradually while maintaining compatibility layers.
