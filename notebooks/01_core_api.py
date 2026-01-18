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
# # Core API
#
# This notebook introduces:
#
# - The `IVData` layout
# - `tsls` for workflow point estimates
# - `ar_test` and `ar_confidence_set` for weak-IV robust inference
#

# %%
from pathlib import Path
import ivrobust as ivr

ART = Path("artifacts") / "01_core_api"
ART.mkdir(parents=True, exist_ok=True)

ivr.set_style()


# %%
data, beta_true = ivr.weak_iv_dgp(n=600, k=4, strength=0.3, beta=1.25, seed=1)
data.nobs, data.p_exog, data.p_endog, data.k_instr

# %% [markdown]
# ## Workflow estimator: 2SLS
#
# 2SLS standard errors are conventional (strong-ID). Use AR inference if instrument strength is questionable.

# %%
tsls_res = ivr.tsls(data, cov_type="HC1")
tsls_res.beta, beta_true

# %% [markdown]
# The 2SLS estimate should be close to the true value on average in this DGP,
# but its standard errors are not weak-IV robust. Use AR inference for validity
# under weak identification.

# %% [markdown]
# ## Weak-IV robust inference: AR test
#

# %%
ar = ivr.ar_test(data, beta0=beta_true, cov_type="HC1")
ar.statistic, ar.pvalue

# %% [markdown]
# A non-rejection at the true beta is expected. Under weak instruments, the AR
# test remains correctly sized, unlike conventional Wald tests.

# %% [markdown]
# ## AR confidence set
#

# %%
cs = ivr.ar_confidence_set(data, alpha=0.05, cov_type="HC1", beta_bounds=(-10, 10))
cs.confidence_set.intervals

# %% [markdown]
# The AR confidence set may be wide or disjoint in weak-ID settings. This is a
# feature of weak-IV robust inference, not a numerical issue.

# %%
fig, ax = ivr.plot_ar_confidence_set(cs)
ivr.savefig(fig, ART / "ar_confidence_set", formats=("png", "pdf"))
