# Methods at a glance

ivrobust focuses on weak-IV robust inference for a single endogenous regressor.
Use the unified workflow when you want consistent defaults and comparable output
across AR, LM/K, and CLR procedures.

```python
import ivrobust as ivr

res = ivr.weakiv_inference(data, beta0=1.0, methods=("AR", "LM", "CLR"))
res.tests.keys(), res.confidence_sets.keys()
```

## Core tests and confidence sets

<div class="grid cards iv-grid" markdown>

-   **Anderson-Rubin (AR)**

    Joint test on instruments under the null with weak-ID robustness. Invert the
    AR test for set-valued confidence intervals. [@anderson1949; @andrews2019]

-   **LM/K test**

    Kleibergen-Paap LM statistic with covariance choices matched to your
    application. [@kleibergen2002]

-   **CLR / CQLR**

    Conditional likelihood ratio inference with grid inversion for confidence
    sets. [@moreira2003]

</div>

## Scope and assumptions

- The weak-IV robust tests in ivrobust target **one endogenous regressor**
  (`p_endog=1`).
- Confidence sets are produced by test inversion and can be disjoint or
  unbounded; report the full union of intervals.
- Robust covariance options (HC0-HC3, clustering, HAC) apply across tests and
  diagnostics.

## Robust covariance and diagnostics

<div class="grid cards iv-grid" markdown>

-   **Robust covariance**

    HC0-HC3 and one-way cluster robust options across tests and diagnostics.
    [@finlay2009]

-   **Set-valued confidence sets**

    Disjoint, unbounded intervals are reported directly rather than trimmed.
    [@mikusheva2010]

-   **Diagnostics**

    Effective F, KP-rk, and weak-ID diagnostics for instrument strength.
    [@montielolea2013]

</div>

## Related API entrypoints

```python
ivr.ar_test(data, beta0=1.0)
ivr.lm_test(data, beta0=1.0)
ivr.clr_test(data, beta0=1.0)

ivr.ar_confidence_set(data, alpha=0.05)
ivr.lm_confidence_set(data, alpha=0.05)
ivr.clr_confidence_set(data, alpha=0.05)
```

See the [User guide](../user-guide/index.md) for method-specific details.
