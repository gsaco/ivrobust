<div class="iv-hero" markdown>
<div>

<span class="iv-hero__eyebrow">Weak-IV robust inference in Python</span>

# ivrobust keeps inference credible when instruments are weak.

<div class="iv-hero__lead" markdown>
Research-grade Anderson-Rubin, LM/K, and CLR inference with robust covariance
options, set-valued confidence sets, and diagnostics that match the modern weak
instrument literature.
</div>

<div class="iv-hero__actions">

[Quickstart](quickstart.md){ .md-button .md-button--primary }
[User guide](user-guide/index.md){ .md-button }
[Notebooks](notebooks.md){ .md-button }

</div>

<div class="iv-hero__stats">
  <div class="iv-stat"><strong>AR / LM / CLR</strong><br />Unified weak-IV workflow.</div>
  <div class="iv-stat"><strong>Robust covariance</strong><br />HC0-HC3 and cluster options.</div>
  <div class="iv-stat"><strong>Set-valued CIs</strong><br />Disjoint intervals supported.</div>
</div>

</div>

<div class="iv-hero__code" markdown>

```python
import ivrobust as ivr

data, beta_true = ivr.weak_iv_dgp(n=300, k=5, strength=0.4, beta=1.0, seed=0)

res = ivr.weakiv_inference(
    data,
    beta0=beta_true,
    alpha=0.05,
    methods=("AR", "LM", "CLR"),
    cov_type="HC1",
)
print(res.tests["AR"].pvalue)
print(res.confidence_sets["CLR"].intervals)
```

</div>
</div>

<section class="iv-section" markdown>
## Why this matters

<div class="iv-callout" markdown>
Weak instruments break classical IV inference. ivrobust keeps tests valid under
weak identification, reports diagnostics alongside inference, and produces
publication-quality figures with a single, consistent plotting style.
</div>
</section>

<section class="iv-section" markdown>
## Methods at a glance

<div class="grid cards iv-grid" markdown>

-   :material-scale-balance: **Anderson-Rubin (AR)**

    Invert the joint test on instruments for weak-IV robust inference and
    confidence sets. [@anderson1949; @andrews2019]

-   :material-waveform: **LM/K test**

    Kleibergen-Paap LM statistics for scalar structural parameters with robust
    covariance support. [@kleibergen2002]

-   :material-chart-bell-curve: **CLR / CQLR**

    Conditional likelihood ratio inference with grid inversion for confidence
    sets. [@moreira2003]

-   :material-vector-combine: **Set-valued confidence sets**

    Disjoint or unbounded intervals are allowed and reported explicitly. [@mikusheva2010]

-   :material-shield-check: **Robust covariance**

    HC0-HC3 and one-way clustering for reliable inference under heteroskedasticity.
    [@finlay2009]

-   :material-stethoscope: **Diagnostics**

    Effective F, weak-ID diagnostics, and rank tests for instrument strength.
    [@montielolea2013]

</div>
</section>

<section class="iv-section" markdown>
## Start here

<div class="iv-button-row">

[Quickstart](quickstart.md){ .md-button .md-button--primary }
[User guide](user-guide/index.md){ .md-button }
[Notebooks](notebooks.md){ .md-button }

</div>
</section>

<section class="iv-section" markdown>
## Gallery highlights

<div class="grid" markdown>

<figure class="iv-figure" markdown>

![AR confidence set](assets/figures/ar_confidence_set.png)

<figcaption>Weak-IV confidence set (AR), generated with ivrobust.</figcaption>
</figure>

<figure class="iv-figure" markdown>

![Weak-IV p-value curves](assets/figures/pvalue_curve.png)

<figcaption>AR, LM, and CLR p-values across the beta grid.</figcaption>
</figure>

</div>

Explore more in the [Gallery](gallery.md).
</section>

<section class="iv-section" markdown>
## Reproducible by design

- Deterministic data generators for weak-IV benchmarks.
- Figures saved with ivrobust plotting conventions for papers and reports.
- Notebooks curated for applied workflows and reproducibility checks.
</section>
