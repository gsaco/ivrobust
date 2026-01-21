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
# # Many instruments: bias in TSLS vs LIML vs Fuller
#
# ## Context
#
# When k grows relative to n, TSLS can exhibit finite-sample bias. LIML and
# Fuller are often less biased in many-instrument settings.
#
# ## Model and estimand
#
# Scalar endogenous regressor with varying number of instruments.
#
# ## Procedure
#
# - Vary k relative to n
# - Compare TSLS, LIML, and Fuller bias and RMSE
#
# ## Key takeaways
#
# - Bias can grow as k/n increases.
# - LIML and Fuller often reduce bias compared to TSLS.

# %%
from pathlib import Path
import os

import numpy as np
import matplotlib.pyplot as plt
import ivrobust as ivr

ART = Path("artifacts") / "04_many_instruments_bias_tsls_liml_fuller"
ART.mkdir(parents=True, exist_ok=True)

ivr.set_style()

# %%
n = 250
k_grid = [2, 5, 10, 20]
strength = 0.3
beta_true = 1.0
R = int(os.getenv("IVROBUST_MC_REPS", "40"))

tsls_bias = []
liml_bias = []
fuller_bias = []
tsls_rmse = []
liml_rmse = []
fuller_rmse = []

dist_data = {}

for k in k_grid:
    tsls_est = []
    liml_est = []
    fuller_est = []
    for r in range(R):
        data, _ = ivr.weak_iv_dgp(
            n=n, k=k, strength=strength, beta=beta_true, seed=r
        )
        tsls_est.append(ivr.tsls(data, cov_type="HC1").beta)
        liml_est.append(ivr.liml(data, cov_type="HC1").beta)
        fuller_est.append(ivr.fuller(data, alpha=1.0, cov_type="HC1").beta)
    tsls_est = np.array(tsls_est, dtype=float)
    liml_est = np.array(liml_est, dtype=float)
    fuller_est = np.array(fuller_est, dtype=float)

    tsls_bias.append(float(np.mean(tsls_est - beta_true)))
    liml_bias.append(float(np.mean(liml_est - beta_true)))
    fuller_bias.append(float(np.mean(fuller_est - beta_true)))

    tsls_rmse.append(float(np.sqrt(np.mean((tsls_est - beta_true) ** 2))))
    liml_rmse.append(float(np.sqrt(np.mean((liml_est - beta_true) ** 2))))
    fuller_rmse.append(float(np.sqrt(np.mean((fuller_est - beta_true) ** 2))))

    dist_data[k] = (tsls_est, liml_est, fuller_est)

k_over_n = [k / n for k in k_grid]

# %% [markdown]
# ## Bias vs k/n

# %%
fig, ax = plt.subplots(figsize=(6.0, 3.8))
ax.plot(k_over_n, tsls_bias, marker="o", label="TSLS")
ax.plot(k_over_n, liml_bias, marker="s", label="LIML")
ax.plot(k_over_n, fuller_bias, marker="^", label="Fuller")
ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
ax.set_xlabel("k/n")
ax.set_ylabel("bias")
ax.set_title("Estimator bias vs k/n")
ax.legend(frameon=False)
ivr.savefig(fig, ART / "bias_vs_k_over_n", formats=("png", "pdf"))

# %% [markdown]
# ![Bias vs k/n](artifacts/04_many_instruments_bias_tsls_liml_fuller/bias_vs_k_over_n.png)

# %% [markdown]
# ## RMSE vs k/n

# %%
fig, ax = plt.subplots(figsize=(6.0, 3.8))
ax.plot(k_over_n, tsls_rmse, marker="o", label="TSLS")
ax.plot(k_over_n, liml_rmse, marker="s", label="LIML")
ax.plot(k_over_n, fuller_rmse, marker="^", label="Fuller")
ax.set_xlabel("k/n")
ax.set_ylabel("RMSE")
ax.set_title("Estimator RMSE vs k/n")
ax.legend(frameon=False)
ivr.savefig(fig, ART / "rmse_vs_k_over_n", formats=("png", "pdf"))

# %% [markdown]
# ![RMSE vs k/n](artifacts/04_many_instruments_bias_tsls_liml_fuller/rmse_vs_k_over_n.png)

# %% [markdown]
# ## Sampling distributions (largest k)

# %%
max_k = max(k_grid)
tsls_est, liml_est, fuller_est = dist_data[max_k]

fig, ax = plt.subplots(figsize=(6.4, 3.8))
ax.hist(tsls_est, bins=18, alpha=0.6, label="TSLS", density=True)
ax.hist(liml_est, bins=18, alpha=0.6, label="LIML", density=True)
ax.hist(fuller_est, bins=18, alpha=0.6, label="Fuller", density=True)
ax.axvline(beta_true, color="black", linestyle="--", linewidth=1.0)
ax.set_title(f"Sampling distributions (k={max_k})")
ax.set_xlabel("beta estimate")
ax.legend(frameon=False)
ivr.savefig(fig, ART / "sampling_distributions", formats=("png", "pdf"))

# %% [markdown]
# ![Sampling distributions](artifacts/04_many_instruments_bias_tsls_liml_fuller/sampling_distributions.png)
