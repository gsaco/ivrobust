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
# # Heteroskedasticity and clustering
#
# ## What you will learn
#
# - How HC vs cluster-robust covariance choices change inference
# - How to flag weak instruments with effective F under clustering
#
# ## Implementation context (for contributors)
#
# - What to build: shared covariance engine with HC/cluster parity across tests.
# - Why it matters: robust covariance is required for credible weak-IV inference.
# - Literature/benchmarks: Finlay & Magnusson (2009); Andrews–Stock–Sun (2019).
# - Codex-ready tasks: implement HC2/HC3 + standardized cluster warnings.
# - Tests/docs: unit tests for covariance variants + notebook comparisons.

# %%
from pathlib import Path

import numpy as np
import ivrobust as ivr

ART = Path("artifacts") / "07_heteroskedasticity_and_clustering"
ART.mkdir(parents=True, exist_ok=True)

ivr.set_style()

# %%
data, beta_true = ivr.weak_iv_dgp(n=400, k=4, strength=0.4, beta=1.0, seed=21)

# Create artificial clusters
n_clusters = 20
clusters = np.repeat(np.arange(n_clusters), np.ceil(data.nobs / n_clusters))[
    : data.nobs
]
data_clustered = data.with_clusters(clusters)

# %%
hc = ivr.weakiv_inference(
    data,
    beta0=beta_true,
    alpha=0.05,
    methods=("AR", "LM", "CLR"),
    cov_type="HC1",
)

cl = ivr.weakiv_inference(
    data_clustered,
    beta0=beta_true,
    alpha=0.05,
    methods=("AR", "LM", "CLR"),
    cov_type="cluster",
)

hc.tests["AR"].pvalue, cl.tests["AR"].pvalue

# %%
hc.diagnostics["effective_f"], cl.diagnostics["effective_f"]

# %% [markdown]
# ## P-value curves: HC1 vs cluster

# %%
import matplotlib.pyplot as plt

hc_grid = ivr.weakiv_inference(
    data,
    beta0=beta_true,
    alpha=0.05,
    methods=("AR",),
    cov_type="HC1",
    grid=(beta_true - 2.0, beta_true + 2.0, 301),
    return_grid=True,
)
cl_grid = ivr.weakiv_inference(
    data_clustered,
    beta0=beta_true,
    alpha=0.05,
    methods=("AR",),
    cov_type="cluster",
    grid=(beta_true - 2.0, beta_true + 2.0, 301),
    return_grid=True,
)

fig, ax = plt.subplots(figsize=(6.2, 3.8))
ax.plot(
    hc_grid.confidence_sets["AR"].grid_info["grid"],
    hc_grid.confidence_sets["AR"].grid_info["pvalues"],
    label="HC1",
)
ax.plot(
    cl_grid.confidence_sets["AR"].grid_info["grid"],
    cl_grid.confidence_sets["AR"].grid_info["pvalues"],
    label="Cluster",
)
ax.axhline(0.05, color="black", linestyle="--", linewidth=1.0)
ax.set_xlabel(r"$\beta$")
ax.set_ylabel("AR p-value")
ax.set_title("Covariance choice and AR p-values")
ax.legend(frameon=False)
ivr.savefig(fig, ART / "covariance_pvalues", formats=("png", "pdf"))

# %% [markdown]
# ## Effective F comparison

# %%
eff_hc = hc.diagnostics["effective_f"].statistic
eff_cl = cl.diagnostics["effective_f"].statistic

fig, ax = plt.subplots(figsize=(4.8, 3.4))
ax.bar(["HC1", "Cluster"], [eff_hc, eff_cl])
ax.set_ylabel("Effective F")
ax.set_title("Instrument strength under clustering")
ivr.savefig(fig, ART / "effective_f_comparison", formats=("png", "pdf"))
