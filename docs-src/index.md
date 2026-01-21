<div class="ivr-hero" markdown>
<div class="ivr-hero__layout" markdown>

<div class="ivr-hero__copy" markdown>
<span class="ivr-hero__badge">Research Software</span>

<h1 class="ivr-hero__title">Weak-IV robust inference for linear IV</h1>

<p class="ivr-hero__subtitle">
ivrobust is a Python library for weak-instrument robust inference in linear IV
models. It implements Anderson-Rubin, LM/K, and CLR tests with robust
covariance options, set-valued confidence sets, and instrument-strength
diagnostics designed for applied research workflows.
</p>

<div class="ivr-hero__actions">
<a href="quickstart/" class="md-button md-button--primary">Get Started</a>
<a href="reference/api/" class="md-button">API Reference</a>
<a href="https://github.com/gsaco/ivrobust" class="md-button">View on GitHub</a>
</div>

<div class="ivr-hero__install">
<span class="ivr-hero__install-label">Install</span>
<code>pip install "ivrobust[plot]"</code>
</div>

<div class="ivr-hero__features">
<div class="ivr-feature">
<div class="ivr-feature__icon">01</div>
<div class="ivr-feature__text">
<strong>Unified workflow</strong>
<span>AR, LM, and CLR tests and confidence sets in one call.</span>
</div>
</div>
<div class="ivr-feature">
<div class="ivr-feature__icon">02</div>
<div class="ivr-feature__text">
<strong>Robust covariance</strong>
<span>HC0-HC3, cluster, and HAC options across tests and diagnostics.</span>
</div>
</div>
<div class="ivr-feature">
<div class="ivr-feature__icon">03</div>
<div class="ivr-feature__text">
<strong>Diagnostics first</strong>
<span>Effective F, KP rk, and weak-ID summaries for reporting.</span>
</div>
</div>
</div>
</div>

<div class="ivr-hero__figure" markdown>
<figure class="ivr-figure" markdown>
![Weak-IV p-value curves](assets/figures/pvalue_curve.png)
<figcaption>
<strong>Weak-IV p-value curves.</strong>
AR, LM, and CLR p-values computed across a beta grid using
<code>weakiv_inference</code>.
</figcaption>
</figure>
</div>

</div>
</div>

<div class="ivr-badge-row">
<a href="https://github.com/gsaco/ivrobust/actions/workflows/ci.yml"><img src="https://github.com/gsaco/ivrobust/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
<img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python 3.10+">
<img src="https://img.shields.io/badge/license-MIT-green" alt="MIT License">
</div>

---

<section class="ivr-section" markdown>

## Why ivrobust

<div class="ivr-split" markdown>
<div class="ivr-split__content" markdown>
ivrobust targets weak instruments in linear IV models and emphasizes
research-grade reporting. Standard 2SLS t-tests can be severely distorted when
instruments are weak. The AR, LM, and CLR procedures in ivrobust remain valid
under weak identification, following the guidance of [@andrews2019].

Key properties:

- Set-valued confidence sets from test inversion (disjoint or unbounded is
  expected under weak identification).
- Robust covariance regimes that match applied settings (HC0-HC3, cluster, HAC).
- Strength diagnostics and underidentification tests in the same workflow.
</div>

<div class="ivr-split__media" markdown>
<figure class="ivr-figure" markdown>
![AR confidence set](assets/figures/ar_confidence_set.png)
<figcaption>
<strong>AR confidence set.</strong>
Inversion yields set-valued intervals without trimming or ad hoc fixes.
</figcaption>
</figure>
</div>
</div>

<div class="ivr-callout ivr-callout--accent" markdown>
**Scope:** weak-IV robust inference currently targets a single endogenous
regressor (<code>p_endog=1</code>) with optional exogenous controls.
</div>

</section>

---

<section class="ivr-section" markdown>

## Core methods

<div class="ivr-methods-grid" markdown>

<div class="ivr-method-card" markdown>

### Anderson-Rubin (AR)

Joint test on instruments under the null. Inversion yields confidence sets that
are valid regardless of instrument strength.

<div class="ivr-method-card__equation">
AR(beta0) = (Y - X beta0)' Pz (Y - X beta0) / s^2
</div>

<span class="ivr-method-card__cite">Reference: Anderson and Rubin (1949)</span>
</div>

<div class="ivr-method-card" markdown>

### Lagrange Multiplier (LM/K)

Kleibergen-Paap score test with robust covariance choices. Often higher power
than AR while retaining weak-ID validity.

<div class="ivr-method-card__equation">
LM(beta0) = S(beta0)' Omega^-1 S(beta0)
</div>

<span class="ivr-method-card__cite">Reference: Kleibergen (2002)</span>
</div>

<div class="ivr-method-card" markdown>

### Conditional Likelihood Ratio (CLR)

Conditional likelihood ratio inference (CQLR by default) with grid inversion for
confidence sets.

<div class="ivr-method-card__equation">
CLR(beta0) = 0.5 (AR - rk + sqrt((AR - rk)^2 + 4 * LM * rk))
</div>

<span class="ivr-method-card__cite">Reference: Moreira (2003)</span>
</div>

</div>

</section>

---

<section class="ivr-section" markdown>

## Quick example

```python
import ivrobust as ivr

# Synthetic weak-IV data
# (single endogenous regressor; intercept included)
data, beta_true = ivr.weak_iv_dgp(n=300, k=5, strength=0.4, beta=1.0, seed=0)

# Weak-IV robust inference
res = ivr.weakiv_inference(
    data,
    beta0=beta_true,
    alpha=0.05,
    methods=("AR", "LM", "CLR"),
    cov_type="HC1",
)

# Results
print(f"AR p-value: {res.tests['AR'].pvalue:.4f}")
print(f"CLR confidence set: {res.confidence_sets['CLR'].intervals}")
```

<div class="ivr-button-row">
<a href="quickstart/" class="md-button md-button--primary">Full Quickstart Guide</a>
<a href="notebooks/" class="md-button">Notebook Gallery</a>
</div>

</section>

---

<section class="ivr-section" markdown>

## Core functionality

<div class="grid cards iv-grid" markdown>

-   **Weak-IV inference**

    Unified workflow via `weakiv_inference` plus individual entrypoints for
    `ar_test`, `lm_test`, `clr_test`, and their inverted confidence sets.

-   **Diagnostics**

    `first_stage_diagnostics`, `effective_f`, `weak_id_diagnostics`, `kp_rank_test`,
    and `stock_yogo_critical_values` for instrument strength reporting.

-   **Estimators**

    `tsls`, `liml`, `fuller`, and `fit` for point estimates and conventional
    robust standard errors (workflow support).

-   **Plotting**

    `plot_ar_confidence_set`, `res.plot()` for p-value curves, plus `set_style`
    and `savefig` for publication-ready figures.

</div>

</section>

---

<section class="ivr-section" markdown>

## Practitioner workflow

<div class="ivr-workflow">
<div class="ivr-workflow__step">
<div class="ivr-workflow__icon">1</div>
<h4>Prepare data</h4>
<p>Use `IVData` or `IVModel.from_arrays`</p>
</div>
<div class="ivr-workflow__step">
<div class="ivr-workflow__icon">2</div>
<h4>Diagnose strength</h4>
<p>Report effective F and KP rk</p>
</div>
<div class="ivr-workflow__step">
<div class="ivr-workflow__icon">3</div>
<h4>Test and invert</h4>
<p>Run AR, LM, and CLR with a shared covariance spec</p>
</div>
<div class="ivr-workflow__step">
<div class="ivr-workflow__icon">4</div>
<h4>Report and plot</h4>
<p>Use p-value curves and set-valued intervals</p>
</div>
</div>

<a href="user-guide/workflow/" class="md-button">Complete Workflow Guide</a>

</section>

---

<section class="ivr-section" markdown>

## Gallery highlights

<div class="ivr-gallery" markdown>

<figure class="ivr-figure" markdown>
![P-value curves](assets/figures/pvalue_curve.png)
<figcaption>
<strong>P-value curves across the parameter space.</strong>
AR, LM, and CLR p-values showing where the null is rejected.
</figcaption>
</figure>

<figure class="ivr-figure" markdown>
![Rejection rates](assets/figures/rejection_vs_strength.png)
<figcaption>
<strong>Monte Carlo rejection rates.</strong>
AR maintains correct size while 2SLS t-tests over-reject under weak instruments.
</figcaption>
</figure>

</div>

<a href="gallery/" class="md-button">View Full Gallery</a>

</section>

---

<section class="ivr-section ivr-team-section" markdown>

## Research team

<p class="section-lead">
ivrobust is developed and maintained by researchers committed to rigorous,
reproducible econometric software.
</p>

<div class="ivr-team-grid">
<div class="ivr-team-card">
<div class="ivr-team-card__avatar">GS</div>
<div class="ivr-team-card__info">
<h4>Gabriel Saco</h4>
<p>Lead Developer • Econometrics Research</p>
</div>
</div>
<div class="ivr-team-card">
<div class="ivr-team-card__avatar">+</div>
<div class="ivr-team-card__info">
<h4>Contributors</h4>
<p>Open source community</p>
</div>
</div>
</div>

</section>

---

<section class="ivr-section" markdown>

## Citing ivrobust

<div class="ivr-citation-block" markdown>
<span class="ivr-citation-block__label">BibTeX</span>

```bibtex
@software{ivrobust,
  title = {ivrobust: Weak-IV Robust Inference in Python},
  author = {Saco, Gabriel and contributors},
  year = {2026},
  url = {https://github.com/gsaco/ivrobust},
  version = {0.2.0}
}
```

</div>

When using ivrobust, please also cite the methodological references for the
specific tests you employ (see [References](references.md)).

</section>

---

<section class="ivr-section" markdown>

## Trust and reproducibility

<div class="grid cards iv-grid" markdown>

-   **Continuous integration**

    Linting, type checks, unit tests, notebook execution, and docs builds run
    on every change.

-   **Reproducible figures**

    Every figure in the documentation is generated from committed code with
    fixed random seeds.

-   **Clear scope**

    Focused implementation of weak-IV robust inference for a single endogenous
    regressor with comprehensive documentation and examples.

</div>

</section>

---

<div class="ivr-button-row" style="justify-content: center; margin-top: 3rem;">
<a href="quickstart/" class="md-button md-button--primary">Get Started</a>
<a href="user-guide/" class="md-button">User Guide</a>
<a href="reference/api/" class="md-button">API Reference</a>
</div>
