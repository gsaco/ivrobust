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

-   :material-scale-balance: **Anderson-Rubin inference**

    Robust inference under weak identification.

    [Read more](ar.md)

-   :material-waveform: **LM/K test**

    Kleibergen-Paap LM statistics and confidence sets.

    [Read more](lm.md)

-   :material-chart-bell-curve: **CLR test**

    Conditional likelihood ratio inference and grids.

    [Read more](clr.md)

-   :material-atom: **Estimators**

    2SLS, LIML, Fuller, and k-class estimators.

    [Read more](estimators.md)

-   :material-stethoscope: **Diagnostics**

    Effective F, weak-ID diagnostics, and rank tests.

    [Read more](diagnostics.md)

-   :material-compass: **Diagnostics & interpretation**

    Practical guidance on reading outputs and avoiding pitfalls.

    [Read more](../diagnostics-interpretation.md)

-   :material-vector-combine: **Many instruments**

    Guidance when k is large relative to n.

    [Read more](many_instruments.md)

-   :material-palette: **Plotting**

    Style conventions and figure output for papers.

    [Read more](plotting.md)

-   :material-directions: **Choosing a method**

    Decision table for AR/LM/CLR selection and covariance regimes.

    [Read more](choosing.md)

-   :material-route: **Workflow**

    Estimate -> diagnose -> test -> confidence set workflow.

    [Read more](workflow.md)

-   :material-shield-check: **Covariance choices**

    HC, cluster, and HAC options and what they change.

    [Read more](covariance.md)

-   :material-numeric: **Numerics**

    Grid inversion, edge cases, and stability considerations.

    [Read more](numerics.md)

</div>
