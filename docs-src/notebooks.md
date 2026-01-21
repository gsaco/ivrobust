# Notebooks

The notebook suite lives in `notebooks/` (paired `.ipynb` and `.py` via
Jupytext). The documentation renders a curated subset inline and links to the
full directory for additional materials.

## Featured notebooks (rendered here)

<div class="grid cards iv-grid" markdown>

-   **00 Quickstart**

    Weak instruments in 10 minutes. Unified API, confidence sets, and plots.

    [Open notebook](00_quickstart/)
    [View source](https://github.com/gsaco/ivrobust/blob/main/notebooks/00_quickstart.ipynb)

-   **01 Practitioner workflow (single endog)**

    End-to-end applied workflow with AR/LM/CLR tests and confidence sets.

    [Open notebook](01_practitioner_workflow_single_endog/)
    [View source](https://github.com/gsaco/ivrobust/blob/main/notebooks/01_practitioner_workflow_single_endog.ipynb)

-   **02 Diagnostics and inference**

    Effective F, weak-ID diagnostics, and interpretation guidance.

    [Open notebook](02_diagnostics_and_inference/)
    [View source](https://github.com/gsaco/ivrobust/blob/main/notebooks/02_diagnostics_and_inference.ipynb)

-   **04 Many instruments bias**

    TSLS vs LIML vs Fuller bias and RMSE as k/n grows.

    [Open notebook](04_many_instruments_bias_tsls_liml_fuller/)
    [View source](https://github.com/gsaco/ivrobust/blob/main/notebooks/04_many_instruments_bias_tsls_liml_fuller.ipynb)

-   **04 Real data example**

    Real-data IV workflow with robust inference and plots.

    [Open notebook](04_real_data_example/)
    [View source](https://github.com/gsaco/ivrobust/blob/main/notebooks/04_real_data_example.ipynb)

-   **08 Runtime scaling**

    Lightweight runtime checks for grid inversion and inference.

    [Open notebook](08_runtime_scaling/)
    [View source](https://github.com/gsaco/ivrobust/blob/main/notebooks/08_runtime_scaling.ipynb)

</div>

## Full notebook index

Additional notebooks (core API, CLR/LM in practice, robust covariance, and more)
live in the repository:

- [notebooks/ on GitHub](https://github.com/gsaco/ivrobust/tree/main/notebooks)

!!! note
    Execute notebooks with `pytest --nbmake notebooks/*.ipynb`.
