# Quickstart

Get weak-IV robust inference in minutes with ivrobust.

---

## Installation

=== "With plotting support (recommended)"

    ```bash
    pip install "ivrobust[plot]"
    ```

=== "Core only"

    ```bash
    pip install ivrobust
    ```

=== "Development"

    ```bash
    git clone https://github.com/gsaco/ivrobust.git
    cd ivrobust
    pip install -e ".[dev,plot]"
    ```

---

## Your First Analysis

### Step 1: Prepare Your Data

ivrobust uses an `IVData` container to structure your IV model:

```python
import ivrobust as ivr
import numpy as np

# Option A: Use built-in data generator for testing
data, beta_true = ivr.weak_iv_dgp(n=300, k=5, strength=0.4, beta=1.0, seed=0)

# Option B: Create from your own arrays
# data = ivr.IVData(Y=your_y, X=your_x, Z=your_z, W=your_controls)
```

### Step 2: Run Weak-IV Robust Inference

The unified workflow computes all three test statistics in one call:

```python
res = ivr.weakiv_inference(
    data,
    beta0=beta_true,      # Null hypothesis value
    alpha=0.05,           # Significance level
    methods=("AR", "LM", "CLR"),
    cov_type="HC1",       # Heteroskedasticity-robust
)
```

### Step 3: Examine Results

```python
# Test results
print(res.tests["AR"].summary())
# Output: AR statistic: 1.234, p-value: 0.267, df: 5

# Confidence sets
print(res.confidence_sets["CLR"].intervals)
# Output: [(-0.42, 2.31)]  # Can be disjoint or unbounded!
```

---

## Visualization

### P-Value Curves

Visualize how p-values change across the parameter space:

```python
# Request grid computation
res = ivr.weakiv_inference(
    data,
    beta0=beta_true,
    alpha=0.05,
    methods=("AR", "LM", "CLR"),
    cov_type="HC1",
    grid=(beta_true - 2.0, beta_true + 2.0, 301),
    return_grid=True,
)

# Generate publication-ready figure
ivr.set_style()
fig, ax = res.plot()
ivr.savefig(fig, "artifacts/pvalue_curve", formats=("png", "pdf"), dpi=300)
```

### Confidence Set Plots

```python
cs = ivr.ar_confidence_set(data, alpha=0.05, cov_type="HC1")
fig, ax = ivr.plot_ar_confidence_set(cs)
ivr.savefig(fig, "artifacts/ar_cs", formats=("png", "pdf"), dpi=300)
```

---

## Key Concepts

<div class="ivr-methods-grid" markdown>

<div class="ivr-method-card" markdown>
<div class="ivr-method-card__icon">⚠️</div>

### Why Not Just Use 2SLS?

Standard 2SLS t-tests can have severe size distortions when instruments are weak. 
The AR/LM/CLR tests in ivrobust remain valid regardless of instrument strength.
</div>

<div class="ivr-method-card" markdown>
<div class="ivr-method-card__icon">📊</div>

### Set-Valued Confidence Sets

Under weak identification, confidence sets may be disjoint (multiple intervals) 
or unbounded (extending to ±∞). ivrobust reports these directly without trimming.
</div>

<div class="ivr-method-card" markdown>
<div class="ivr-method-card__icon">🛡️</div>

### Robust Covariance

All tests support HC0-HC3 heteroskedasticity-robust and cluster-robust covariance 
estimation via the `cov_type` parameter.
</div>

</div>

---

## Next Steps

<div class="ivr-button-row">

[User Guide](user-guide/index.md){ .md-button .md-button--primary }
[Choosing a Method](user-guide/choosing.md){ .md-button }
[Interactive Notebooks](notebooks.md){ .md-button }

</div>

!!! tip "For Practitioners"
    See the [Workflow Guide](user-guide/workflow.md) for a complete estimate → diagnose → test → report pipeline.
