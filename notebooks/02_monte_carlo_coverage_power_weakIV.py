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
# # Monte Carlo: coverage and power under weak IV
#
# ## Context
#
# We compare AR and CLR behavior across weak-instrument strengths. The focus is
# on size, power, and confidence set coverage.
#
# ## Model and estimand
#
# Linear IV model with a single endogenous regressor; target is the scalar beta.
#
# ## Procedure
#
# - Simulate weak-IV data across strengths
# - Compute AR/CLR tests at the null and a local alternative
# - Invert to confidence sets and track coverage and length
#
# ## Caveats
#
# This notebook keeps Monte Carlo sizes small for speed. Increase
# IVROBUST_MC_REPS for more precision.
#
# ## Key takeaways
#
# - AR maintains size under weak instruments.
# - CLR can be more powerful but may yield nonstandard confidence sets.

# %%
from pathlib import Path
import os

import numpy as np
import matplotlib.pyplot as plt
import ivrobust as ivr

ART = Path("artifacts") / "02_monte_carlo_coverage_power_weakIV"
ART.mkdir(parents=True, exist_ok=True)

ivr.set_style()

# %%
strength_grid = [0.05, 0.1, 0.2, 0.4, 0.8]
n = 220
k = 5
beta_true = 1.0
beta_alt = 1.5
alpha = 0.05
R = int(os.getenv("IVROBUST_MC_REPS", "40"))


def interval_length(intervals: list[tuple[float, float]]) -> float:
    length = 0.0
    for lo, hi in intervals:
        if not np.isfinite(lo) or not np.isfinite(hi):
            return float("inf")
        length += hi - lo
    return length


def summarize_strength(strength: float) -> dict[str, float]:
    ar_rej = 0
    clr_rej = 0
    ar_pow = 0
    clr_pow = 0
    ar_cov = 0
    clr_cov = 0
    ar_len = []
    clr_len = []
    ar_empty = 0
    ar_unbounded = 0
    ar_disjoint = 0
    clr_empty = 0
    clr_unbounded = 0
    clr_disjoint = 0

    for r in range(R):
        data, _ = ivr.weak_iv_dgp(
            n=n, k=k, strength=strength, beta=beta_true, seed=r
        )
        ar = ivr.ar_test(data, beta0=beta_true, cov_type="HC1")
        clr = ivr.clr_test(data, beta0=beta_true, cov_type="HC1")
        ar_rej += int(ar.pvalue < alpha)
        clr_rej += int(clr.pvalue < alpha)

        ar_alt = ivr.ar_test(data, beta0=beta_alt, cov_type="HC1")
        clr_alt = ivr.clr_test(data, beta0=beta_alt, cov_type="HC1")
        ar_pow += int(ar_alt.pvalue < alpha)
        clr_pow += int(clr_alt.pvalue < alpha)

        ar_cs = ivr.ar_confidence_set(
            data, alpha=alpha, cov_type="HC1", n_grid=201
        )
        clr_cs = ivr.clr_confidence_set(
            data, alpha=alpha, cov_type="HC1", n_grid=201
        )

        ar_cov += int(ar_cs.confidence_set.contains(beta_true))
        clr_cov += int(clr_cs.confidence_set.contains(beta_true))
        ar_len.append(interval_length(ar_cs.intervals))
        clr_len.append(interval_length(clr_cs.intervals))

        ar_empty += int(ar_cs.is_empty)
        ar_unbounded += int(ar_cs.is_unbounded)
        ar_disjoint += int(ar_cs.is_disjoint)
        clr_empty += int(clr_cs.is_empty)
        clr_unbounded += int(clr_cs.is_unbounded)
        clr_disjoint += int(clr_cs.is_disjoint)

    return {
        "ar_size": ar_rej / R,
        "clr_size": clr_rej / R,
        "ar_power": ar_pow / R,
        "clr_power": clr_pow / R,
        "ar_cov": ar_cov / R,
        "clr_cov": clr_cov / R,
        "ar_len": float(np.mean(ar_len)),
        "clr_len": float(np.mean(clr_len)),
        "ar_empty": ar_empty / R,
        "ar_unbounded": ar_unbounded / R,
        "ar_disjoint": ar_disjoint / R,
        "clr_empty": clr_empty / R,
        "clr_unbounded": clr_unbounded / R,
        "clr_disjoint": clr_disjoint / R,
    }


summaries = [summarize_strength(s) for s in strength_grid]

# %%
ar_size = [s["ar_size"] for s in summaries]
clr_size = [s["clr_size"] for s in summaries]
ar_power = [s["ar_power"] for s in summaries]
clr_power = [s["clr_power"] for s in summaries]
ar_cov = [s["ar_cov"] for s in summaries]
clr_cov = [s["clr_cov"] for s in summaries]
ar_len = [s["ar_len"] for s in summaries]
clr_len = [s["clr_len"] for s in summaries]

# %% [markdown]
# ## Size by strength

# %%
fig, ax = plt.subplots(figsize=(6.0, 3.8))
ax.plot(strength_grid, ar_size, marker="o", label="AR")
ax.plot(strength_grid, clr_size, marker="s", label="CLR")
ax.axhline(alpha, color="black", linestyle="--", linewidth=1.0)
ax.set_xlabel("strength")
ax.set_ylabel("rejection rate at true beta")
ax.set_title("Size by strength")
ax.legend(frameon=False)
ivr.savefig(fig, ART / "size_by_strength", formats=("png", "pdf"))

# %% [markdown]
# ## Power by strength

# %%
fig, ax = plt.subplots(figsize=(6.0, 3.8))
ax.plot(strength_grid, ar_power, marker="o", label="AR")
ax.plot(strength_grid, clr_power, marker="s", label="CLR")
ax.set_xlabel("strength")
ax.set_ylabel("rejection rate at alternative")
ax.set_title("Power by strength")
ax.legend(frameon=False)
ivr.savefig(fig, ART / "power_by_strength", formats=("png", "pdf"))

# %% [markdown]
# ## Coverage by strength

# %%
fig, ax = plt.subplots(figsize=(6.0, 3.8))
ax.plot(strength_grid, ar_cov, marker="o", label="AR")
ax.plot(strength_grid, clr_cov, marker="s", label="CLR")
ax.axhline(1.0 - alpha, color="black", linestyle="--", linewidth=1.0)
ax.set_xlabel("strength")
ax.set_ylabel("coverage")
ax.set_title("Confidence set coverage")
ax.legend(frameon=False)
ivr.savefig(fig, ART / "coverage_by_strength", formats=("png", "pdf"))

# %% [markdown]
# ## Average confidence set length

# %%
fig, ax = plt.subplots(figsize=(6.0, 3.8))
ax.plot(strength_grid, ar_len, marker="o", label="AR")
ax.plot(strength_grid, clr_len, marker="s", label="CLR")
ax.set_xlabel("strength")
ax.set_ylabel("avg length (finite only)")
ax.set_title("Average confidence set length")
ax.legend(frameon=False)
ivr.savefig(fig, ART / "ci_length_by_strength", formats=("png", "pdf"))

# %% [markdown]
# ## Nonstandard confidence set frequency

# %%
labels = ["empty", "unbounded", "disjoint"]
ar_flags = [
    np.mean([s["ar_empty"] for s in summaries]),
    np.mean([s["ar_unbounded"] for s in summaries]),
    np.mean([s["ar_disjoint"] for s in summaries]),
]
clr_flags = [
    np.mean([s["clr_empty"] for s in summaries]),
    np.mean([s["clr_unbounded"] for s in summaries]),
    np.mean([s["clr_disjoint"] for s in summaries]),
]

x = np.arange(len(labels))
width = 0.35
fig, ax = plt.subplots(figsize=(6.0, 3.6))
ax.bar(x - width / 2, ar_flags, width, label="AR")
ax.bar(x + width / 2, clr_flags, width, label="CLR")
ax.set_xticks(x, labels)
ax.set_ylabel("frequency")
ax.set_title("Nonstandard confidence set frequency")
ax.legend(frameon=False)
ivr.savefig(fig, ART / "nonstandard_ci_frequency", formats=("png", "pdf"))
