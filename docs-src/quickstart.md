# Quickstart

Get a weak-IV robust workflow in minutes.

## Install

```bash
python -m pip install "ivrobust[plot]"
```

## Run weak-IV robust inference

=== "Core workflow"

    ```python
    import ivrobust as ivr

    data, beta_true = ivr.weak_iv_dgp(n=300, k=5, strength=0.4, beta=1.0, seed=0)

    res = ivr.weakiv_inference(
        data,
        beta0=beta_true,
        alpha=0.05,
        methods=("AR", "LM", "CLR"),
        cov_type="HC1",
    )

    print(res.tests["AR"].summary())
    print(res.confidence_sets["CLR"].intervals)
    ```

=== "Plot p-value curves"

    ```python
    res = ivr.weakiv_inference(
        data,
        beta0=beta_true,
        alpha=0.05,
        methods=("AR", "LM", "CLR"),
        cov_type="HC1",
        grid=(beta_true - 2.0, beta_true + 2.0, 301),
        return_grid=True,
    )

    fig, ax = res.plot()
    ivr.savefig(fig, "artifacts/quickstart/pvalue_curve", dpi=500)
    ```

=== "Confidence sets"

    ```python
    cs = ivr.ar_confidence_set(data, alpha=0.05, cov_type="HC1")
    fig, ax = ivr.plot_ar_confidence_set(cs)
    ivr.savefig(fig, "artifacts/quickstart/ar_confidence_set", dpi=500)
    ```

!!! tip
    For more detail, see the [User guide](user-guide/index.md) and the curated
    [Notebooks](notebooks.md).
