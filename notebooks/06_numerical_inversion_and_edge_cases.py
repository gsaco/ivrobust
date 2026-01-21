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
# # Numerical inversion and edge cases
#
# ## Context
#
# Weak-IV confidence sets can have nonstandard shapes. This notebook illustrates
# edge cases and how grid refinement affects inversion.
#
# ## Model and estimand
#
# Scalar endogenous regressor with weak or collinear instruments.
#
# ## Procedure
#
# - Create very weak and near-collinear instrument designs
# - Plot p-value curves
# - Compare coarse vs refined inversion grids
#
# ## Key takeaways
#
# - Weak instruments can yield flat p-value curves.
# - Grid refinement stabilizes interval boundaries.

# %%
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import ivrobust as ivr

ART = Path("artifacts") / "06_numerical_inversion_and_edge_cases"
ART.mkdir(parents=True, exist_ok=True)

ivr.set_style()

# %% [markdown]
# ## Case 1: Very weak instruments

# %%
data_weak, beta_true = ivr.weak_iv_dgp(n=220, k=4, strength=0.05, beta=1.0, seed=7)

grid_res = ivr.weakiv_inference(
    data_weak,
    beta0=beta_true,
    methods=("AR",),
    cov_type="HC1",
    grid=(beta_true - 3.0, beta_true + 3.0, 301),
    return_grid=True,
)

fig, ax = plt.subplots(figsize=(6.2, 3.6))
ax.plot(
    grid_res.confidence_sets["AR"].grid_info["grid"],
    grid_res.confidence_sets["AR"].grid_info["pvalues"],
)
ax.axhline(0.05, color="black", linestyle="--", linewidth=1.0)
ax.set_xlabel(r"$\beta$")
ax.set_ylabel("AR p-value")
ax.set_title("Weak instruments: flat p-value curve")
ivr.savefig(fig, ART / "pvalue_curve_edge_cases", formats=("png", "pdf"))

# %% [markdown]
# ## Case 2: Near-collinear instruments

# %%
rng = np.random.default_rng(42)
n = 220
z1 = rng.standard_normal((n, 1))
z2 = z1 + 1e-6 * rng.standard_normal((n, 1))
z3 = rng.standard_normal((n, 1))
z = np.hstack([z1, z2, z3])
x = np.ones((n, 1))

pi = (0.3 / np.sqrt(z.shape[1])) * np.ones((z.shape[1], 1))
u = rng.standard_normal((n, 1))
v = 0.5 * u + np.sqrt(1 - 0.5**2) * rng.standard_normal((n, 1))
d = z @ pi + v
y = beta_true * d + u

data_col = ivr.IVData(y=y, d=d, x=x, z=z)

grid_res = ivr.weakiv_inference(
    data_col,
    beta0=beta_true,
    methods=("AR",),
    cov_type="HC1",
    grid=(beta_true - 3.0, beta_true + 3.0, 301),
    return_grid=True,
)

fig, ax = plt.subplots(figsize=(6.2, 3.6))
ax.plot(
    grid_res.confidence_sets["AR"].grid_info["grid"],
    grid_res.confidence_sets["AR"].grid_info["pvalues"],
)
ax.axhline(0.05, color="black", linestyle="--", linewidth=1.0)
ax.set_xlabel(r"$\beta$")
ax.set_ylabel("AR p-value")
ax.set_title("Near-collinear instruments")
ivr.savefig(fig, ART / "acceptance_region_union", formats=("png", "pdf"))

# %% [markdown]
# ## Case 3: Grid refinement

# %%
cs_coarse = ivr.ar_confidence_set(
    data_weak, alpha=0.05, cov_type="HC1", n_grid=101, refine=False
)
cs_refined = ivr.ar_confidence_set(
    data_weak, alpha=0.05, cov_type="HC1", n_grid=2001, refine=True
)

fig, ax = plt.subplots(figsize=(6.2, 2.2))
for lo, hi in cs_coarse.intervals:
    ax.plot([lo, hi], [0.5, 0.5], label="coarse")
for lo, hi in cs_refined.intervals:
    ax.plot([lo, hi], [0.0, 0.0], label="refined")
ax.set_yticks([0.0, 0.5], ["refined", "coarse"])
ax.set_xlabel(r"$\beta$")
ax.set_title("Grid refinement demo")
ivr.savefig(fig, ART / "grid_refinement_demo", formats=("png", "pdf"))
