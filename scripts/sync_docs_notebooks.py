from __future__ import annotations

import shutil
from pathlib import Path


NOTEBOOKS = [
    "00_quickstart",
    "01_practitioner_workflow_single_endog",
    "02_diagnostics_and_inference",
    "04_many_instruments_bias_tsls_liml_fuller",
    "04_real_data_example",
    "08_runtime_scaling",
]


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _copy_tree(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst, dirs_exist_ok=True)


def _rewrite_artifact_paths(path: Path, name: str) -> None:
    needle = f"artifacts/{name}/"
    replacement = f"../artifacts/{name}/"
    text = path.read_text()
    if needle in text:
        path.write_text(text.replace(needle, replacement))


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    notebooks_dir = root / "notebooks"
    docs_notebooks_dir = root / "docs-src" / "notebooks"
    docs_artifacts_dir = docs_notebooks_dir / "artifacts"

    for name in NOTEBOOKS:
        src_nb = notebooks_dir / f"{name}.ipynb"
        dst_nb = docs_notebooks_dir / f"{name}.ipynb"
        if not src_nb.exists():
            raise FileNotFoundError(f"Missing notebook: {src_nb}")
        _copy_file(src_nb, dst_nb)
        _rewrite_artifact_paths(dst_nb, name)

        src_artifacts = notebooks_dir / "artifacts" / name
        dst_artifacts = docs_artifacts_dir / name
        if not src_artifacts.exists():
            raise FileNotFoundError(f"Missing artifacts: {src_artifacts}")
        _copy_tree(src_artifacts, dst_artifacts)


if __name__ == "__main__":
    main()
