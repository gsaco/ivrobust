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
# # Robust covariance: HC, cluster, HAC
#
# ## Context
#
# Covariance choices can materially change weak-IV robust inference. This
# notebook compares HC, cluster-robust, and HAC options.
#
# ## Model and estimand
#
# Scalar endogenous regressor with correlated errors.
#
# ## Procedure
#
# - Simulate data with mild serial correlation
# - Compare AR/LM/CLR p-values under HC, cluster, and HAC
# - Compare AR confidence sets across covariance choices
#
# ## Key takeaways
#
# - Covariance regime changes test statistics and confidence sets.
# - HAC is appropriate when serial correlation is present.

# %%
from pathlib import Path
import os

import numpy as np
import matplotlib.pyplot as plt
import ivrobust as ivr

ART = Path("artifacts") / "05_robust_vcov_hc_cluster_hac"
ART.mkdir(parents=True, exist_ok=True)

ivr.set_style()

# %%
rng = np.random.default_rng(0)
n = 260
k = 5
beta_true = 1.0

z = rng.standard_normal((n, k))
x = np.ones((n, 1))
pi = (0.35 / np.sqrt(k)) * np.ones((k, 1))

# AR(1) errors for serial correlation
rho = 0.5
eps_u = rng.standard_normal((n, 1))
eps_v = rng.standard_normal((n, 1))
u = np.zeros((n, 1))
v = np.zeros((n, 1))
for t in range(1, n):
    u[t] = rho * u[t - 1] + eps_u[t]
    v[t] = rho * v[t - 1] + eps_v[t]

d = z @ pi + v
y = beta_true * d + u

data = ivr.IVData(y=y, d=d, x=x, z=z)

# Cluster labels
clusters = np.repeat(np.arange(20), np.ceil(n / 20))[:n]
data_clustered = data.with_clusters(clusters)

# %%
cov_types = ["HC1", "cluster", "HAC"]
results = {}
for cov_type in cov_types:
    if cov_type == "cluster":
        res = ivr.weakiv_inference(
            data_clustered,
            beta0=beta_true,
            methods=("AR", "LM", "CLR"),
            cov_type=cov_type,
        )
    elif cov_type == "HAC":
        res = ivr.weakiv_inference(
            data,
            beta0=beta_true,
            methods=("AR", "LM", "CLR"),
            cov_type=cov_type,
            hac_lags=4,
        )
    else:
        res = ivr.weakiv_inference(
            data,
            beta0=beta_true,
            methods=("AR", "LM", "CLR"),
            cov_type=cov_type,
        )
    results[cov_type] = res

# %% [markdown]
# ## P-values by covariance type

# %%
methods = ["AR", "LM", "CLR"]
fig, ax = plt.subplots(figsize=(6.4, 3.8))
for method in methods:
    vals = [results[cov].tests[method].pvalue for cov in cov_types]
    ax.plot(cov_types, vals, marker="o", label=method)
ax.set_ylabel("p-value")
ax.set_title("P-values by covariance choice")
ax.legend(frameon=False)
ivr.savefig(fig, ART / "pvalues_by_vcov_type", formats=("png", "pdf"))

# %% [markdown]
# ## Confidence sets by covariance type (AR)

# %%
fig, ax = plt.subplots(figsize=(6.4, 2.2))
ypos = np.arange(len(cov_types))
for i, cov in enumerate(cov_types):
    cs = results[cov].confidence_sets["AR"]
    for lo, hi in cs.intervals:
        ax.plot([lo, hi], [ypos[i], ypos[i]], solid_capstyle="butt")
ax.set_yticks(ypos, cov_types)
ax.set_xlabel(r"$\beta$")
ax.set_title("AR confidence sets by covariance type")
ivr.savefig(fig, ART / "ci_by_vcov_type", formats=("png", "pdf"))

# %% [markdown]
# ## Rejection rates under serial correlation

# %%
R = int(os.getenv("IVROBUST_MC_REPS", "30"))
rej = {cov: 0 for cov in cov_types}
for r in range(R):
    rng = np.random.default_rng(r)
    z = rng.standard_normal((n, k))
    eps_u = rng.standard_normal((n, 1))
    eps_v = rng.standard_normal((n, 1))
    u = np.zeros((n, 1))
    v = np.zeros((n, 1))
    for t in range(1, n):
        u[t] = rho * u[t - 1] + eps_u[t]
        v[t] = rho * v[t - 1] + eps_v[t]
    d = z @ pi + v
    y = beta_true * d + u
    data = ivr.IVData(y=y, d=d, x=x, z=z)
    data_clustered = data.with_clusters(clusters)

    ar_hc = ivr.ar_test(data, beta0=beta_true, cov_type="HC1")
    ar_cl = ivr.ar_test(data_clustered, beta0=beta_true, cov_type="cluster")
    ar_hac = ivr.ar_test(data, beta0=beta_true, cov_type="HAC", hac_lags=4)
    rej["HC1"] += int(ar_hc.pvalue < 0.05)
    rej["cluster"] += int(ar_cl.pvalue < 0.05)
    rej["HAC"] += int(ar_hac.pvalue < 0.05)

fig, ax = plt.subplots(figsize=(5.0, 3.4))
ax.bar(rej.keys(), [v / R for v in rej.values()])
ax.axhline(0.05, color="black", linestyle="--", linewidth=1.0)
ax.set_ylabel("rejection rate at true beta")
ax.set_title("Rejection under serial correlation")
ivr.savefig(fig, ART / "rejection_rate_under_misspec", formats=("png", "pdf"))
