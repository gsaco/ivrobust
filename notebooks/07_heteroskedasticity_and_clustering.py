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
# # Heteroskedasticity and clustering
#
# ## What you will learn
#
# - How HC vs cluster-robust covariance choices change inference
# - How to flag weak instruments with effective F under clustering
#
# ## Implementation context (for contributors)
#
# - What to build: shared covariance engine with HC/cluster parity across tests.
# - Why it matters: robust covariance is required for credible weak-IV inference.
# - Literature/benchmarks: Finlay & Magnusson (2009); Andrews–Stock–Sun (2019).
# - Codex-ready tasks: implement HC2/HC3 + standardized cluster warnings.
# - Tests/docs: unit tests for covariance variants + notebook comparisons.

# %%
from pathlib import Path

import numpy as np
import ivrobust as ivr

ART = Path("artifacts") / "07_heteroskedasticity_and_clustering"
ART.mkdir(parents=True, exist_ok=True)

ivr.set_style()

# %%
data, beta_true = ivr.weak_iv_dgp(n=400, k=4, strength=0.4, beta=1.0, seed=21)

# Create artificial clusters
n_clusters = 20
clusters = np.repeat(np.arange(n_clusters), np.ceil(data.nobs / n_clusters))[
    : data.nobs
]
data_clustered = data.with_clusters(clusters)

# %%
hc = ivr.weakiv_inference(
    data,
    beta0=beta_true,
    alpha=0.05,
    methods=("AR", "LM", "CLR"),
    cov_type="HC1",
)

cl = ivr.weakiv_inference(
    data_clustered,
    beta0=beta_true,
    alpha=0.05,
    methods=("AR", "LM", "CLR"),
    cov_type="cluster",
)

hc.tests["AR"].pvalue, cl.tests["AR"].pvalue

# %%
hc.diagnostics["effective_f"], cl.diagnostics["effective_f"]
