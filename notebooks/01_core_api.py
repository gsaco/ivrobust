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
# # Core API
#
# This notebook introduces:
#
# - The `IVData` layout
# - `tsls`/`liml`/`fuller` for workflow point estimates
# - `ar_test`/`lm_test`/`clr_test` for weak-IV robust inference
# - `weakiv_inference` as a unified entry point
#
# ## Implementation context (for contributors)
#
# - What to build: a consistent, front-door API (`IVModel` + `weakiv_inference`)
#   with standardized result objects.
# - Why it matters: reduces friction for applied users and makes weak-IV robust
#   inference visible in routine workflows.
# - Literature/benchmarks: Stata weak-IV postestimation; estimatr's regression UX.
# - Codex-ready tasks: add `IVModel.from_arrays`, `IVResults.weakiv`, and
#   plotting helpers for p-value curves.
# - Tests/docs: unit tests for API surfaces + notebook examples with pinned seeds.
#

# %%
from pathlib import Path
import ivrobust as ivr

ART = Path("artifacts") / "01_core_api"
ART.mkdir(parents=True, exist_ok=True)

ivr.set_style()


# %%
data, beta_true = ivr.weak_iv_dgp(n=600, k=4, strength=0.3, beta=1.25, seed=1)
data.nobs, data.p_exog, data.p_endog, data.k_instr

# %% [markdown]
# ## Workflow estimator: 2SLS
#
# 2SLS standard errors are conventional (strong-ID). Use AR inference if instrument strength is questionable.

# %%
tsls_res = ivr.tsls(data, cov_type="HC1")
tsls_res.beta, beta_true

# %%
liml_res = ivr.liml(data, cov_type="HC1")
fuller_res = ivr.fuller(data, alpha=1.0, cov_type="HC1")
liml_res.beta, fuller_res.beta

# %% [markdown]
# ## Model-style API
#
# The model wrapper mirrors familiar stats interfaces.

# %%
model = ivr.IVModel.from_arrays(
    y=data.y,
    x_endog=data.d,
    z=data.z,
    x_exog=None,
    add_const=True,
)
model_results = model.fit(estimator="liml", cov_type="HC1")
model_results.params.ravel()[:3]

# %% [markdown]
# The 2SLS estimate should be close to the true value on average in this DGP,
# but its standard errors are not weak-IV robust. Use AR inference for validity
# under weak identification.

# %% [markdown]
# ## Weak-IV robust inference: AR/LM/CLR tests
#

# %%
weakiv = ivr.weakiv_inference(
    data,
    beta0=beta_true,
    alpha=0.05,
    methods=("AR", "LM", "CLR"),
    cov_type="HC1",
)
weakiv.tests["AR"], weakiv.tests["LM"], weakiv.tests["CLR"]

# %% [markdown]
# A non-rejection at the true beta is expected. Under weak instruments, the AR
# test remains correctly sized, unlike conventional Wald tests.

# %% [markdown]
# ## Confidence sets
#

# %%
cs = weakiv.confidence_sets["AR"]
cs.confidence_set.intervals

# %% [markdown]
# The AR confidence set may be wide or disjoint in weak-ID settings. This is a
# feature of weak-IV robust inference, not a numerical issue.

# %%
fig, ax = ivr.plot_ar_confidence_set(cs)
ivr.savefig(fig, ART / "ar_confidence_set", formats=("png", "pdf"))

# %% [markdown]
# ## P-value curves for AR/LM/CLR
#
# Comparing the three weak-IV robust tests on a common grid.

# %%
weakiv_grid = ivr.weakiv_inference(
    data,
    beta0=beta_true,
    alpha=0.05,
    methods=("AR", "LM", "CLR"),
    cov_type="HC1",
    grid=(beta_true - 2.0, beta_true + 2.0, 301),
    return_grid=True,
)
fig, ax = weakiv_grid.plot()
ivr.savefig(fig, ART / "pvalue_curve", formats=("png", "pdf"))
