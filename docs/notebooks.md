# Notebooks

The notebook suite lives in `notebooks/` (paired `.ipynb` and `.py` via
Jupytext). Each notebook is reproducible and aligned with the package API.

<div class="grid cards iv-grid" markdown>

-   :material-rocket-launch: **00 Quickstart**

    Weak instruments in 10 minutes. Learn the unified API and confidence sets.

    [Open notebook](https://github.com/gsaco/ivrobust/blob/main/notebooks/00_quickstart.ipynb)

-   :material-book-open-variant: **01 Core API**

    Data model, estimators, and the main entrypoints for inference.

    [Open notebook](https://github.com/gsaco/ivrobust/blob/main/notebooks/01_core_api.ipynb)

-   :material-stethoscope: **02 Diagnostics and inference**

    Effective F, weak-ID diagnostics, and how they inform inference.

    [Open notebook](https://github.com/gsaco/ivrobust/blob/main/notebooks/02_diagnostics_and_inference.ipynb)

-   :material-chart-line: **03 Simulation study**

    Simulation checks to understand weak-IV behavior and coverage.

    [Open notebook](https://github.com/gsaco/ivrobust/blob/main/notebooks/03_simulation_study.ipynb)

-   :material-domain: **04 Real data example**

    End-to-end workflow on a real dataset with reporting conventions.

    [Open notebook](https://github.com/gsaco/ivrobust/blob/main/notebooks/04_real_data_example.ipynb)

-   :material-chart-bell-curve: **05 CLR in practice**

    CLR confidence sets and diagnostics for applied settings.

    [Open notebook](https://github.com/gsaco/ivrobust/blob/main/notebooks/05_clr_in_practice.ipynb)

-   :material-waveform: **06 LM in practice**

    LM/K confidence sets, p-value curves, and workflow patterns.

    [Open notebook](https://github.com/gsaco/ivrobust/blob/main/notebooks/06_lm_in_practice.ipynb)

-   :material-vector-combine: **07 Heteroskedasticity and clustering**

    Compare covariance choices and clustering behavior.

    [Open notebook](https://github.com/gsaco/ivrobust/blob/main/notebooks/07_heteroskedasticity_and_clustering.ipynb)

</div>

!!! note
    Run notebooks with `pytest --nbmake` and launch locally with JupyterLab.
