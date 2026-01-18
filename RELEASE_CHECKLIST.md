# Release checklist (PyPI)

This checklist is designed to make the release process repeatable and auditable.

## Pre-release

- [ ] CI is green on `main` (lint, typecheck, tests, build).
- [ ] Notebooks execute in CI (`pytest --nbmake ...`) with artifacts written to
      `artifacts/`.
- [ ] `mkdocs build` succeeds locally and in CI.
- [ ] `CHANGELOG.md` has an entry for the release version.
- [ ] `CITATION.cff` is up to date.
- [ ] Version in `pyproject.toml` is correct and follows semantic versioning.

## Build and validate artifacts

From a clean environment:

```bash
python -m pip install --upgrade pip
python -m pip install build twine
python -m build
python -m twine check dist/*
```

Test install from the built wheel:

```bash
python -m venv /tmp/ivrobust-test
source /tmp/ivrobust-test/bin/activate
pip install dist/*.whl
python -c "import ivrobust; print(ivrobust.__version__)"
```

## Release steps (GitHub + PyPI)

- Tag the release: `git tag vX.Y.Z && git push --tags`
- Create a GitHub Release using the tag.
- Upload to PyPI:

```bash
python -m twine upload dist/*
```

## Post-release

- Verify PyPI install: `pip install ivrobust==X.Y.Z`
- Verify docs deployment (if enabled)
- Open a tracking issue for the next milestone (see `docs/roadmap.md`)
