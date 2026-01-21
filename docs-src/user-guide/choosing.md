# Choosing a weak-IV robust method

This guide helps you pick an inference method based on your identification
concerns, covariance regime, and reporting needs.

## Inputs to the decision

- Suspected instrument strength (weak vs strong).
- Covariance regime (heteroskedastic, cluster, HAC/serial correlation).
- Whether you need a test, a confidence set, or both.
- Willingness to report nonstandard confidence sets (disjoint or unbounded).

## Decision table

| Scenario | Recommended method(s) | Notes |
| --- | --- | --- |
| Instruments likely weak, scalar endogenous regressor | AR or CLR | AR is simple and robust; CLR often has higher power. |
| You need a single test with good power | CLR (CQLR) | Use CQLR under heteroskedasticity or clustering. |
| You need a quick robustness check | AR + LM | LM can complement AR when power differs. |
| Robust covariance required (cluster/HAC) | AR/LM/CLR with cov_type set | Ensure the method supports the desired covariance regime. |
| Many instruments (k/n not negligible) | Report diagnostics + LIML/Fuller | Weak-IV tests are still valid for fixed k; warn when k/n is large. |

## Nonstandard confidence sets

Weak-IV robust confidence sets are obtained by inversion and can be:

- Empty (no values pass the test at alpha).
- Unbounded (the full real line).
- A union of disjoint intervals.

These shapes are expected under weak identification and should be reported as
they are.

## Stata parity expectations

Stata's weak-IV postestimation advertises AR/CLR inference, robust covariance
options, and nonstandard confidence sets. ivrobust mirrors this workflow and
exposes the same objects in Python.
