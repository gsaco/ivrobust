# User guide

The user guide focuses on weak-IV robust inference workflows, from data setup to
publication-ready plots.

## What you will learn

- How `IVData` structures the model and assumptions.
- When to prefer AR, LM/K, or CLR inference.
- How to interpret diagnostics and effective F statistics.
- How to interpret diagnostics and safe defaults for reporting.
- How to report set-valued confidence sets and p-value curves.

## Guide map

<div class="grid cards iv-grid" markdown>

-   **Anderson-Rubin inference**

    Robust inference under weak identification.

    [Read more](ar.md)

-   **LM/K test**

    Kleibergen-Paap LM statistics and confidence sets.

    [Read more](lm.md)

-   **CLR test**

    Conditional likelihood ratio inference and grids.

    [Read more](clr.md)

-   **Estimators**

    2SLS, LIML, Fuller, and k-class estimators.

    [Read more](estimators.md)

-   **Diagnostics**

    Effective F, weak-ID diagnostics, and rank tests.

    [Read more](diagnostics.md)

-   **Diagnostics & interpretation**

    Practical guidance on reading outputs and avoiding pitfalls.

    [Read more](../diagnostics-interpretation.md)

-   **Many instruments**

    Guidance when k is large relative to n.

    [Read more](many_instruments.md)

-   **Plotting**

    Style conventions and figure output for papers.

    [Read more](plotting.md)

-   **Choosing a method**

    Decision table for AR/LM/CLR selection and covariance regimes.

    [Read more](choosing.md)

-   **Workflow**

    Estimate -> diagnose -> test -> confidence set workflow.

    [Read more](workflow.md)

-   **Covariance choices**

    HC, cluster, and HAC options and what they change.

    [Read more](covariance.md)

-   **Numerics**

    Grid inversion, edge cases, and stability considerations.

    [Read more](numerics.md)

</div>
