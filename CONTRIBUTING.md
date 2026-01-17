# Contributing

Thanks for your interest in contributing to ivrobust. We welcome issues, bug
reports, documentation improvements, and code contributions.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev,test,docs]
```

## Quality checks

```bash
ruff check .
ruff format --check .
mypy src/ivrobust
pytest
```

## Notes

- Keep changes scoped and add tests for new behavior.
- Document assumptions and validity regions for any inference method.
- Avoid introducing new runtime dependencies unless clearly justified.
