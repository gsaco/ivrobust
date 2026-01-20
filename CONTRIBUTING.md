# Contributing

Thanks for your interest in contributing to ivrobust.

## Development setup

1. Clone the repository.
2. Create and activate a virtual environment.
3. Install development dependencies:

```bash
pip install -e ".[dev,plot,docs]"
pre-commit install
```

## Running checks locally

Lint + format:

```bash
ruff check .
ruff format .
black .
```

Type checking:

```bash
mypy src/ivrobust
pyright
```

Tests:

```bash
pytest
```

Docs:

```bash
mkdocs build --strict
make docs
```

Notebooks (must run top-to-bottom):

```bash
pytest --nbmake notebooks/*.ipynb
```

Figures (docs assets):

```bash
python scripts/build_figures.py
make figures
```

## Pull request checklist

- Tests added/updated and passing
- Type checking passes
- Documentation updated (docstrings + user guide + references if needed)
- Plotting outputs use ivrobust.set_style() and ivrobust.savefig()
- Changelog entry added (if user-facing change)
