# Notebooks

The repository ships a reproducible notebook suite in `notebooks/` (both `.py`
and `.ipynb` via Jupytext). Each notebook includes purpose, literature context,
and testing/implementation notes.

Current notebooks:

- `notebooks/00_quickstart.ipynb` — Weak instruments in 10 minutes
- `notebooks/01_core_api.ipynb` — Core API and unified inference
- `notebooks/02_diagnostics_and_inference.ipynb` — Diagnostics + effective F
- `notebooks/03_simulation_study.ipynb` — Simulation sanity checks
- `notebooks/04_real_data_example.ipynb` — Real-data workflow
- `notebooks/05_clr_in_practice.ipynb` — CLR confidence sets
- `notebooks/06_lm_in_practice.ipynb` — LM/K confidence sets
- `notebooks/07_heteroskedasticity_and_clustering.ipynb` — Robust covariance choices

Build docs locally with `mkdocs serve` and run notebooks with `pytest --nbmake`.
