# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # ivrobust - Quickstart
#
# This notebook runs the smallest end-to-end workflow:
#
# 1. Simulate a weak-IV dataset
# 2. Run weak-IV robust tests (AR/LM/CLR)
# 3. Compute confidence sets
# 4. Save one publication-style figure
#
# ## Implementation context (for contributors)
#
# - What to build: a single-call weak-IV inference workflow with AR/LM/CLR tests
#   and set-valued confidence sets.
# - Why it matters: applied users want one entry point that makes weak-IV robust
#   inference explicit and reproducible.
# - Literature/benchmarks: Moreira (2003) CLR; Kleibergen (2002) LM/K; Mikusheva
#   (2010) confidence set shapes; Stata weak-IV reporting for CI behavior.
# - Codex-ready tasks: implement `weakiv_inference`, add `lm_test`/`clr_test`,
#   wire plotting helpers, and expose results in the public API.
# - Tests/docs: unit tests against reference implementations + notebooks showing
#   disjoint/unbounded sets with reproducible seeds.

# %%
from pathlib import Path

import ivrobust as ivr

ART = Path("artifacts") / "00_quickstart"
ART.mkdir(parents=True, exist_ok=True)

ivr.set_style()


# %%
data, beta_true = ivr.weak_iv_dgp(n=300, k=5, strength=0.4, beta=1.0, seed=0)
beta_true

# %%
res = ivr.weakiv_inference(
    data,
    beta0=beta_true,
    alpha=0.05,
    methods=("AR", "LM", "CLR"),
    cov_type="HC1",
)
res.tests["AR"]

# %% [markdown]
# ## Interpretation
#
# - At the true beta, the AR test should not reject, so the p-value should be
#   comfortably above common significance levels.
# - The confidence set can be wide or even disjoint under weak instruments; this
#   is expected behavior rather than a numerical bug.

# %%
cs = res.confidence_sets["AR"]
cs.confidence_set.intervals

# %%
fig, ax = ivr.plot_ar_confidence_set(cs)
ivr.savefig(fig, ART / "ar_confidence_set", formats=("png", "pdf"))

# %% [markdown]
# ## P-value curve
#
# Plot p-values across a beta grid to visualize how acceptance changes.

# %%
res_grid = ivr.weakiv_inference(
    data,
    beta0=beta_true,
    alpha=0.05,
    methods=("AR", "LM", "CLR"),
    cov_type="HC1",
    grid=(beta_true - 2.0, beta_true + 2.0, 301),
    return_grid=True,
)
fig, ax = res_grid.plot()
ivr.savefig(fig, ART / "pvalue_curve", formats=("png", "pdf"))
