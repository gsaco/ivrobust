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
# - Effective F (Montiel Olea–Pflueger)
# - Interpreting weak-IV robust confidence sets
# - Comparing conventional and weak-IV robust outputs
#
# ## Implementation context (for contributors)
#
# - What to build: effective F diagnostics and unified inference reporting.
# - Why it matters: classical first-stage F can be misleading under
#   heteroskedasticity or clustering.
# - Literature/benchmarks: Montiel Olea & Pflueger (2013); Andrews–Stock–Sun (2019).
# - Codex-ready tasks: add `effective_f` and integrate into results summaries.
# - Tests/docs: unit tests comparing effective F to classical F under iid settings.
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

# %%
eff = ivr.effective_f(data, cov_type="HC1")
eff

# %% [markdown]
# ## Diagnostic summary plot
#
# Compare classical first-stage F and effective F side-by-side.

# %%
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
axes[0].bar(["First-stage F", "Effective F"], [diag.f_statistic, eff.statistic])
axes[0].set_ylabel("Statistic")
axes[0].set_title("Instrument strength diagnostics")
axes[1].bar(["Partial R^2"], [diag.partial_r2])
axes[1].set_ylim(0.0, 1.0)
axes[1].set_title("Partial R^2")
ivr.savefig(fig, ART / "diagnostics_summary", formats=("png", "pdf"))

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
weakiv = ivr.weakiv_inference(
    data,
    beta0=beta_true,
    alpha=0.05,
    methods=("AR", "LM", "CLR"),
    cov_type="HC1",
)
weakiv.confidence_sets["AR"].confidence_set.intervals

# %% [markdown]
# The AR confidence set is robust to weak identification and may be wider than
# conventional confidence intervals, reflecting true uncertainty.

# %%
fig, ax = ivr.plot_ar_confidence_set(weakiv.confidence_sets["AR"])
ivr.savefig(fig, ART / "ar_confidence_set", formats=("png", "pdf"))
