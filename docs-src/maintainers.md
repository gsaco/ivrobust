# Maintainer guide

This page summarizes the release and documentation workflow for ivrobust.

## Install docs tooling

```bash
pip install -e ".[docs,plot,dev]"
```

## Sync notebooks into docs

```bash
python scripts/sync_docs_notebooks.py
```

The curated notebook list lives inside the script. It copies `.ipynb` files and
their artifacts into `docs-src/notebooks/`.

## Build docs locally

```bash
make docs
```

This runs the sync step and then executes `mkdocs build --strict`.

## Update figures

```bash
make figures
```

Figures are saved under `docs-src/assets/figures/` and referenced in the gallery
and homepage.

## Run notebooks

```bash
pytest --nbmake notebooks/*.ipynb
```

Use `IVROBUST_MC_REPS` to scale Monte Carlo loops when needed.

## Deploy docs

Docs are deployed by GitHub Actions on pushes to `main` and tags:

- Workflow: `.github/workflows/release.yml`
- Command: `mkdocs gh-deploy --force`

You can deploy locally with the same command if needed.
