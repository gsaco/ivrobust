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
# 2. Run an Anderson-Rubin (AR) test
# 3. Compute an AR confidence set
# 4. Save one publication-style figure
#

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
ar = ivr.ar_test(data, beta0=beta_true, cov_type="HC1")
ar

# %% [markdown]
# ## Interpretation
#
# - At the true beta, the AR test should not reject, so the p-value should be
#   comfortably above common significance levels.
# - The confidence set can be wide or even disjoint under weak instruments; this
#   is expected behavior rather than a numerical bug.

# %%
cs = ivr.ar_confidence_set(data, alpha=0.05, cov_type="HC1", beta_bounds=(-10, 10))
cs.confidence_set.intervals

# %%
fig, ax = ivr.plot_ar_confidence_set(cs)
ivr.savefig(fig, ART / "ar_confidence_set", formats=("png", "pdf"))
