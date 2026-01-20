# Replication

This repository includes a lightweight replication harness under `replication/`.

Contents

- `replication/data/weak_iv_fixture.csv`: deterministic fixture used by golden tests.
- `replication/outputs/golden.json`: committed reference outputs produced by ivrobust.
- `replication/stata/ivreg2.do`: placeholder Stata script (ivreg2).
- `replication/r/ivreg.R`: placeholder R script (AER::ivreg).

Workflow

1. Regenerate the golden outputs (Python):

```bash
python replication/generate_golden.py
```

2. Run Stata/R scripts to compare outputs against external packages.
3. Replace `replication/outputs/golden.json` with verified results and re-run tests.

Notes

- CI never runs Stata or R. The scripts are provided for reproducibility.
- The committed `golden.json` currently reflects ivrobust outputs on the fixture.
