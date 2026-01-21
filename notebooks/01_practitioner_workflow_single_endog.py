# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     notebook_metadata_filter: jupytext,kernelspec,language_info
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Practitioner workflow (single endogenous regressor)
#
# ## Context
#
# This notebook shows an end-to-end weak-IV robust workflow using AR/LM/CLR
# inference for a single endogenous regressor.
#
# ## Model and estimand
#
# We consider a linear IV model with one endogenous regressor and seek
# inference on the structural coefficient beta.
#
# ## Procedure
#
# - Generate data with `weak_iv_dgp`
# - Run AR/LM/CLR tests at a null beta
# - Invert to confidence sets
# - Visualize p-value curves and confidence sets
#
# ## Results
#
# We save three figures:
#
# - AR confidence set
# - AR/LM/CLR p-value curves
# - Union-of-intervals diagram
#
# ## Caveats
#
# Weak-IV robust confidence sets can be disjoint or unbounded. Report the full
# set rather than trimming.
#
# ## Key takeaways
#
# - AR/LM/CLR remain valid under weak identification.
# - Confidence sets can be nonstandard but still informative.
# - P-value curves help communicate uncertainty.

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import ivrobust as ivr

ART = Path("artifacts") / "01_practitioner_workflow_single_endog"
ART.mkdir(parents=True, exist_ok=True)

ivr.set_style()

# %%
data, beta_true = ivr.weak_iv_dgp(n=400, k=6, strength=0.35, beta=1.0, seed=1)

# %%
res = ivr.weakiv_inference(
    data,
    beta0=beta_true,
    alpha=0.05,
    methods=("AR", "LM", "CLR"),
    cov_type="HC1",
)
res.tests["AR"].pvalue, res.tests["LM"].pvalue, res.tests["CLR"].pvalue

# %% [markdown]
# ## Confidence sets

# %%
cs_ar = res.confidence_sets["AR"]
cs_ar.intervals

# %%
fig, ax = ivr.plot_ar_confidence_set(cs_ar)
ivr.savefig(fig, ART / "ar_confidence_set", formats=("png", "pdf"))

# %% [markdown]
# ![AR confidence set](artifacts/01_practitioner_workflow_single_endog/ar_confidence_set.png)

# %% [markdown]
# ## P-value curves

# %%
grid_res = ivr.weakiv_inference(
    data,
    beta0=beta_true,
    alpha=0.05,
    methods=("AR", "LM", "CLR"),
    cov_type="HC1",
    grid=(beta_true - 2.0, beta_true + 2.0, 301),
    return_grid=True,
)
fig, ax = grid_res.plot()
ax.set_title("AR/LM/CLR p-value curves")
ivr.savefig(fig, ART / "pvalue_curves_ar_lm_clr", formats=("png", "pdf"))

# %% [markdown]
# ![AR/LM/CLR p-value curves](artifacts/01_practitioner_workflow_single_endog/pvalue_curves_ar_lm_clr.png)

# %% [markdown]
# ## Confidence set diagram

# %%
intervals = cs_ar.intervals
fig, ax = plt.subplots(figsize=(6.0, 1.6))
for lo, hi in intervals:
    ax.plot([lo, hi], [0.0, 0.0], solid_capstyle="butt")
ax.set_yticks([])
ax.set_xlabel(r"$\beta$")
ax.set_title("Union-of-intervals confidence set")
ivr.savefig(fig, ART / "ci_interval_diagram", formats=("png", "pdf"))

# %% [markdown]
# ![Union-of-intervals diagram](artifacts/01_practitioner_workflow_single_endog/ci_interval_diagram.png)
