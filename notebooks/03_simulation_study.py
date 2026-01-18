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
# # Simulation study
#
# We run a small Monte Carlo experiment to illustrate how AR test behavior changes with instrument strength.
#
# This is a teaching notebook: the goal is reproducibility and interpretation, not exhaustive benchmarking.
#

# %%
from pathlib import Path
import numpy as np
import ivrobust as ivr

ART = Path("artifacts") / "03_simulation_study"
ART.mkdir(parents=True, exist_ok=True)

ivr.set_style()


# %%
strength_grid = [0.1, 0.2, 0.4, 0.8]
R = 200
n = 300
k = 5
beta_true = 1.0
alpha = 0.05

reject_rates = []
for s in strength_grid:
    rej = 0
    for r in range(R):
        data, _ = ivr.weak_iv_dgp(n=n, k=k, strength=s, beta=beta_true, seed=r)
        ar = ivr.ar_test(data, beta0=beta_true, cov_type="HC1")
        rej += int(ar.pvalue < alpha)
    reject_rates.append(rej / R)

reject_rates

# %% [markdown]
# In a well-sized test, rejection rates at the true beta should be close to the
# nominal alpha (here 5%). Some variation is expected due to Monte Carlo error
# because R is finite.

# %% [markdown]
# ## Plot rejection rates
#

# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6.0, 4.0))
ax.plot(strength_grid, reject_rates, marker="o")
ax.set_xlabel("first-stage strength (DGP parameter)")
ax.set_ylabel(r"AR rejection rate at true $\beta$")
ax.set_title("Monte Carlo illustration (should be near size)")
ivr.savefig(fig, ART / "ar_rejection_rates", formats=("png", "pdf"))
