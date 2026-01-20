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
# # Real data example (open dataset)
#
# This notebook demonstrates an end-to-end IV workflow on an openly available dataset downloaded in-notebook.
#
# We use the `AER::CollegeDistance` dataset via Rdatasets.
#
# ## Implementation context (for contributors)
#
# - What to build: real-data workflow with weak-IV robust inference outputs.
# - Why it matters: applied users want an end-to-end example they can adapt.
# - Literature/benchmarks: Stata weak-IV reporting; estimatr-style regression UX.
# - Codex-ready tasks: add `weakiv_inference` and diagnostic reporting.
# - Tests/docs: keep runtime short; guard against network failures.
#

# %%
from pathlib import Path
import numpy as np
import pandas as pd
import ivrobust as ivr

ART = Path("artifacts") / "04_real_data_example"
ART.mkdir(parents=True, exist_ok=True)

ivr.set_style()


# %%
url = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/AER/CollegeDistance.csv"
df = pd.read_csv(url)
df.head()


# %% [markdown]
# ## Define a simple IV specification
#
# This example is intentionally minimal: one endogenous regressor and one excluded instrument.
#
# - Outcome: wage (hourly wage)
# - Endogenous regressor: education (years of schooling)
# - Instrument: distance to college (distance)
# - Exogenous controls: intercept plus selected numeric controls
#
# Column names vary slightly across distributions; we defensively select available variables.

# %%
def pick(colnames):
    for c in colnames:
        if c in df.columns:
            return c
    raise KeyError(f"None of {colnames} found")

y_col = pick(["wage"])
d_col = pick(["education"])
z_col = pick(["distance"])

controls = [
    c
    for c in ["score", "unemp", "tuition"]
    if c in df.columns
]

use = [y_col, d_col, z_col] + controls
dff = df[use].dropna().copy()

y = dff[y_col].to_numpy(dtype=float).reshape(-1, 1)
d = dff[d_col].to_numpy(dtype=float).reshape(-1, 1)
z = dff[z_col].to_numpy(dtype=float).reshape(-1, 1)

x_list = [np.ones((len(dff), 1))]
for c in controls:
    x_list.append(dff[c].to_numpy(dtype=float).reshape(-1, 1))
x = np.hstack(x_list)

data = ivr.IVData(y=y, d=d, x=x, z=z)
data.nobs

# %% [markdown]
# This specification is intentionally minimal and uses only numeric controls to
# keep the example transparent and easy to reproduce. In applied work, you would
# typically add richer controls and consider alternative instruments.


# %% [markdown]
# ## 2SLS estimate (workflow)
#

# %%
tsls_res = ivr.tsls(data, cov_type="HC1")
tsls_res.beta, tsls_res.stderr[-1, 0]

# %% [markdown]
# The 2SLS estimate provides a conventional point estimate. Its standard error
# is not weak-IV robust, so treat it as descriptive when identification is
# uncertain.

# %% [markdown]
# ## Weak-IV robust AR confidence set
#

# %%
beta_hat = float(tsls_res.beta)
cs = ivr.ar_confidence_set(data, alpha=0.05, cov_type="HC1", beta_bounds=(beta_hat - 2.0, beta_hat + 2.0))
cs.confidence_set.intervals

# %% [markdown]
# The AR confidence set reflects weak-IV uncertainty and can be wider than a
# conventional interval. If the set is broad, the data provide limited
# information about the causal effect.

# %%
fig, ax = ivr.plot_ar_confidence_set(cs)
ivr.savefig(fig, ART / "ar_confidence_set", formats=("png", "pdf"))

# %% [markdown]
# ## P-value curve for AR/LM/CLR
#
# Visualize weak-IV p-values over a beta grid.

# %%
weakiv_grid = ivr.weakiv_inference(
    data,
    beta0=beta_hat,
    alpha=0.05,
    methods=("AR", "LM", "CLR"),
    cov_type="HC1",
    grid=(beta_hat - 2.0, beta_hat + 2.0, 301),
    return_grid=True,
)
fig, ax = weakiv_grid.plot()
ivr.savefig(fig, ART / "pvalue_curve", formats=("png", "pdf"))
