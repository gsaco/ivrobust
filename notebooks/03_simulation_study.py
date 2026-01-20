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
# ## Implementation context (for contributors)
#
# - What to build: simulation tests that validate size/coverage under strong vs weak IV.
# - Why it matters: reviewers expect empirical sanity checks beyond unit tests.
# - Literature/benchmarks: Andrews–Stock–Sun (2019) guidance on weak-IV diagnostics.
# - Codex-ready tasks: add lightweight Monte Carlo tests with fixed seeds.
# - Tests/docs: keep runtime small; pin randomness for reproducibility.
#

# %%
from pathlib import Path
import os
import numpy as np
from scipy.stats import norm
import ivrobust as ivr

ART = Path("artifacts") / "03_simulation_study"
ART.mkdir(parents=True, exist_ok=True)

ivr.set_style()


# %%
strength_grid = [0.1, 0.2, 0.4, 0.8]
R = int(os.getenv("IVROBUST_MC_REPS", "120"))
n = 300
k = 5
beta_true = 1.0
alpha = 0.05

reject_rates = []
tsls_rates = []
for s in strength_grid:
    ar_rej = 0
    tsls_rej = 0
    for r in range(R):
        data, _ = ivr.weak_iv_dgp(n=n, k=k, strength=s, beta=beta_true, seed=r)
        ar = ivr.ar_test(data, beta0=beta_true, cov_type="HC1")
        ar_rej += int(ar.pvalue < alpha)
        tsls = ivr.tsls(data, cov_type="HC1")
        t_stat = (tsls.beta - beta_true) / tsls.stderr[-1, 0]
        tsls_pval = 2.0 * norm.sf(abs(float(t_stat)))
        tsls_rej += int(tsls_pval < alpha)
    reject_rates.append(ar_rej / R)
    tsls_rates.append(tsls_rej / R)

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

# %% [markdown]
# ## Robust vs conventional rejection rates
#
# Compare AR (robust) to a conventional 2SLS Wald test.

# %%
fig, ax = plt.subplots(figsize=(6.2, 3.8))
ax.plot(strength_grid, reject_rates, marker="o", label="AR (robust)")
ax.plot(strength_grid, tsls_rates, marker="s", label="2SLS t-test")
ax.axhline(alpha, color="black", linestyle="--", linewidth=1.0)
ax.set_xlabel("first-stage strength (DGP parameter)")
ax.set_ylabel("Rejection rate at true beta")
ax.set_title("Robust vs conventional size behavior")
ax.legend(frameon=False)
ivr.savefig(fig, ART / "robust_vs_conventional", formats=("png", "pdf"))
