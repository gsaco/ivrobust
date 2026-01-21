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
# # Diagnostics: effective F and Stock-Yogo
#
# ## Context
#
# This notebook shows how instrument-strength diagnostics vary with first-stage
# strength.
#
# ## Model and estimand
#
# Scalar endogenous regressor with weak instruments.
#
# ## Procedure
#
# - Simulate data across strengths
# - Compute classical first-stage F and effective F
# - Overlay Stock-Yogo thresholds (partial table)
#
# ## Key takeaways
#
# - Effective F tracks strength under heteroskedasticity and clustering.
# - Stock-Yogo values provide a rough reference under homoskedasticity.

# %%
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import ivrobust as ivr

ART = Path("artifacts") / "03_diagnostics_effectiveF_and_stockyogo"
ART.mkdir(parents=True, exist_ok=True)

ivr.set_style()

# %%
strength_grid = [0.05, 0.1, 0.2, 0.4, 0.8]
n = 240
k = 5
beta_true = 1.0

first_stage_f = []
effective_f = []

for idx, strength in enumerate(strength_grid):
    data, _ = ivr.weak_iv_dgp(n=n, k=k, strength=strength, beta=beta_true, seed=idx)
    diag = ivr.first_stage_diagnostics(data)
    eff = ivr.effective_f(data, cov_type="HC1")
    first_stage_f.append(diag.f_statistic)
    effective_f.append(eff.statistic)

# %% [markdown]
# ## First-stage F vs strength

# %%
fig, ax = plt.subplots(figsize=(6.0, 3.8))
ax.plot(strength_grid, first_stage_f, marker="o")
ax.set_xlabel("strength")
ax.set_ylabel("first-stage F")
ax.set_title("First-stage F vs strength")
ivr.savefig(fig, ART / "first_stage_f_vs_strength", formats=("png", "pdf"))

# %% [markdown]
# ## Effective F vs strength

# %%
fig, ax = plt.subplots(figsize=(6.0, 3.8))
ax.plot(strength_grid, effective_f, marker="o")
ax.set_xlabel("strength")
ax.set_ylabel("effective F")
ax.set_title("Effective F vs strength")
ivr.savefig(fig, ART / "effective_f_vs_strength", formats=("png", "pdf"))

# %% [markdown]
# ## Stock-Yogo thresholds

# %%
try:
    sy10 = ivr.stock_yogo_critical_values(k_endog=1, k_instr=k, size_distortion=0.10)
except Exception:
    sy10 = None

fig, ax = plt.subplots(figsize=(6.0, 3.8))
ax.plot(strength_grid, first_stage_f, marker="o", label="First-stage F")
if sy10 is not None:
    ax.axhline(sy10, color="black", linestyle="--", label="Stock-Yogo 10%")
ax.set_xlabel("strength")
ax.set_ylabel("statistic")
ax.set_title("Stock-Yogo threshold overlay")
ax.legend(frameon=False)
ivr.savefig(fig, ART / "stock_yogo_threshold_overlay", formats=("png", "pdf"))
