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
# # Runtime scaling
#
# This notebook gives a lightweight view of runtime scaling for weak-IV
# inference grids. It is intentionally small to keep CI runtime manageable.

# %%
from pathlib import Path
import time
import numpy as np
import ivrobust as ivr

ART = Path("artifacts") / "08_runtime_scaling"
ART.mkdir(parents=True, exist_ok=True)

ivr.set_style()

# %%
import matplotlib.pyplot as plt

rng = np.random.default_rng(31)
configs = [
    (150, 3),
    (250, 5),
    (400, 7),
]

timings = []
for n, k in configs:
    data, beta_true = ivr.weak_iv_dgp(
        n=n,
        k=k,
        strength=0.4,
        beta=1.0,
        seed=int(rng.integers(0, 1_000_000)),
    )
    start = time.perf_counter()
    _ = ivr.weakiv_inference(
        data,
        beta0=beta_true,
        alpha=0.05,
        methods=("AR", "LM", "CLR"),
        cov_type="HC1",
        grid=(beta_true - 1.5, beta_true + 1.5, 301),
        return_grid=False,
    )
    timings.append(time.perf_counter() - start)

timings = np.asarray(timings)

fig, ax = plt.subplots(figsize=(6.2, 3.6))
labels = [f"n={n}, k={k}" for n, k in configs]
ax.plot(range(len(labels)), timings, marker="o")
ax.set_xticks(range(len(labels)), labels)
ax.set_ylabel("Runtime (seconds)")
ax.set_title("Runtime vs sample size and instruments")
ivr.savefig(fig, ART / "runtime_by_nk", formats=("png", "pdf"))

# %% [markdown]
# ## Runtime vs grid size
#
# Hold n and k fixed, then vary the grid length used for inversion.

# %%
base_data, beta_true = ivr.weak_iv_dgp(
    n=250,
    k=5,
    strength=0.4,
    beta=1.0,
    seed=123,
)

grid_sizes = [301, 401, 501]
size_times = []
for n_grid in grid_sizes:
    start = time.perf_counter()
    _ = ivr.weakiv_inference(
        base_data,
        beta0=beta_true,
        alpha=0.05,
        methods=("AR", "LM", "CLR"),
        cov_type="HC1",
        grid=(beta_true - 1.5, beta_true + 1.5, n_grid),
        return_grid=False,
    )
    size_times.append(time.perf_counter() - start)

fig, ax = plt.subplots(figsize=(6.0, 3.4))
ax.plot(grid_sizes, size_times, marker="o")
ax.set_xlabel("Grid size")
ax.set_ylabel("Runtime (seconds)")
ax.set_title("Runtime vs grid length")
ivr.savefig(fig, ART / "runtime_by_grid", formats=("png", "pdf"))
