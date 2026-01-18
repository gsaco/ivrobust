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
# # Diagnostics and inference
#
# This notebook focuses on:
#
# - First-stage reporting (F-statistic, partial R^2)
# - Interpreting weak-IV robust confidence sets
# - Comparing conventional and weak-IV robust outputs
#

# %%
from pathlib import Path
import ivrobust as ivr

ART = Path("artifacts") / "02_diagnostics_and_inference"
ART.mkdir(parents=True, exist_ok=True)

ivr.set_style()


# %%
data, beta_true = ivr.weak_iv_dgp(n=500, k=5, strength=0.2, beta=1.0, seed=2)
diag = ivr.first_stage_diagnostics(data)
diag

# %% [markdown]
# A low first-stage F-statistic and partial R^2 indicate weak instruments. In
# this regime, conventional standard errors can be misleading, so we rely on AR
# inference for valid statements about beta.

# %% [markdown]
# ## Conventional estimator (2SLS)
#

# %%
tsls_res = ivr.tsls(data, cov_type="HC1")
tsls_res.beta, tsls_res.stderr[-1, 0]

# %% [markdown]
# This point estimate is useful for reporting, but interpret the standard error
# with caution when instrument strength is weak.

# %% [markdown]
# ## Weak-IV robust inference
#

# %%
cs = ivr.ar_confidence_set(data, alpha=0.05, cov_type="HC1", beta_bounds=(-10, 10))
cs.confidence_set.intervals

# %% [markdown]
# The AR confidence set is robust to weak identification and may be wider than
# conventional confidence intervals, reflecting true uncertainty.

# %%
fig, ax = ivr.plot_ar_confidence_set(cs)
ivr.savefig(fig, ART / "ar_confidence_set", formats=("png", "pdf"))
