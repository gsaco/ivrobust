# Reproducibility

ivrobust is designed for reproducible, researcher-grade workflows. This page
summarizes how the figures and notebooks in the documentation are generated.

## Seeds and determinism

- Synthetic data in the docs use `weak_iv_dgp` with fixed random seeds.
- Figure generation scripts save both plots and cached data summaries.
- CI uses a fixed `PYTHONHASHSEED` to reduce non-determinism.

## Rebuild figures

```bash
python scripts/build_figures.py
```

Figures are saved to `docs-src/assets/figures/` and cached data are stored in
`docs-src/assets/data/` as `.npz` files.

Using the Makefile shortcut:

```bash
make figures
```

## Execute notebooks

```bash
pytest --nbmake notebooks/*.ipynb
```

Notebooks are intentionally lightweight and use small sample sizes by default.
Set environment variables or edit the notebook parameters if you need larger
simulation sizes.

## Build documentation

```bash
mkdocs build --strict
```

The MkDocs source lives in `docs-src/` and the built site is committed to
`docs/` for GitHub Pages branch-based deployment.

Makefile shortcut:

```bash
make docs
```

## Link checking

```bash
python scripts/check_links.py --site-dir docs
make links
```
