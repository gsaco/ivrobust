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
# # LM/K test in practice
#
# ## What you will learn
#
# - How the LM/K test compares to AR and CLR
# - How to interpret LM confidence sets
# - How to compute LM p-values over a beta grid
#
# ## Implementation context (for contributors)
#
# - What to build: `lm_test` + `lm_confidence_set` + shared inversion utilities.
# - Why it matters: LM provides a powerful weak-IV robust score test.
# - Literature/benchmarks: Kleibergen (2002), Mikusheva (2010).
# - Codex-ready tasks: implement LM statistic and inversion with interval unions.
# - Tests/docs: compare LM p-values to reference implementations on fixed datasets.

# %%
from pathlib import Path

import ivrobust as ivr

ART = Path("artifacts") / "06_lm_in_practice"
ART.mkdir(parents=True, exist_ok=True)

ivr.set_style()

# %%
data, beta_true = ivr.weak_iv_dgp(n=350, k=4, strength=0.4, beta=1.2, seed=13)

# %%
res = ivr.weakiv_inference(
    data,
    beta0=beta_true,
    alpha=0.05,
    methods=("AR", "LM"),
    cov_type="HC1",
    return_grid=True,
)
res.tests["AR"], res.tests["LM"]

# %%
res.confidence_sets["LM"].confidence_set.intervals

# %% [markdown]
# ## Interpretation
#
# - LM confidence sets can be nonstandard (disjoint or unbounded).
# - Report the full union of intervals for transparency.

# %%
fig, ax = res.plot(methods=("AR", "LM"))
ivr.savefig(fig, ART / "lm_pvalues", formats=("png", "pdf"))

# %% [markdown]
# ## LM confidence set shape

# %%
fig, ax = res.confidence_sets["LM"].plot()
ivr.savefig(fig, ART / "lm_confidence_set", formats=("png", "pdf"))
