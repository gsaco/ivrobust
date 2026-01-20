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
# # CLR in practice
#
# ## What you will learn
#
# - How CLR differs from AR in weak-IV settings
# - How to report set-valued CLR confidence sets
# - How to visualize p-value curves over beta
#
# ## Implementation context (for contributors)
#
# - What to build: `clr_test` + `clr_confidence_set` + p-value curve plotting.
# - Why it matters: CLR often yields tighter yet still valid inference under weak IV.
# - Literature/benchmarks: Moreira (2003), Mikusheva (2010); Stata weak-IV CI shapes.
# - Codex-ready tasks: implement CLR stat, conditional p-value, and grid inversion.
# - Tests/docs: golden p-values vs reference implementation + notebook examples.

# %%
from pathlib import Path

import ivrobust as ivr

ART = Path("artifacts") / "05_clr_in_practice"
ART.mkdir(parents=True, exist_ok=True)

ivr.set_style()

# %%
data, beta_true = ivr.weak_iv_dgp(n=400, k=5, strength=0.35, beta=1.0, seed=11)

# %%
res = ivr.weakiv_inference(
    data,
    beta0=beta_true,
    alpha=0.05,
    methods=("AR", "CLR"),
    cov_type="HC1",
    return_grid=True,
)
res.tests["AR"], res.tests["CLR"]

# %%
res.confidence_sets["AR"].confidence_set.intervals

# %%
res.confidence_sets["CLR"].confidence_set.intervals

# %% [markdown]
# ## Interpretation
#
# - CLR often yields narrower acceptance regions than AR while preserving
#   weak-IV robustness.
# - Report CLR intervals as unions of intervals (possibly disjoint).

# %%
fig, ax = res.plot(methods=("AR", "CLR"))
ivr.savefig(fig, ART / "clr_pvalues", formats=("png", "pdf"))

# %% [markdown]
# ## CLR confidence set shape

# %%
fig, ax = res.confidence_sets["CLR"].plot()
ivr.savefig(fig, ART / "clr_confidence_set", formats=("png", "pdf"))
