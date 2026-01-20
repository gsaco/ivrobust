from pathlib import Path

import numpy as np

import ivrobust as ivr
from ivrobust import ConfidenceSetResult, IntervalSet
from ivrobust.plot_style import style_context


def test_plot_ar_confidence_set_branches() -> None:
    grid = np.linspace(-1.0, 1.0, 5).reshape(-1, 1)

    cs_empty = ConfidenceSetResult(
        confidence_set=IntervalSet(intervals=[]),
        alpha=0.05,
        method="AR",
        grid_info={"grid": grid, "df": 1, "cov_type": "HC1"},
    )
    fig, ax = ivr.plot_ar_confidence_set(cs_empty)
    assert ax.axison is False

    cs_nonempty = ConfidenceSetResult(
        confidence_set=IntervalSet(intervals=[(-0.5, 0.5)]),
        alpha=0.05,
        method="AR",
        grid_info={"grid": grid, "df": 1, "cov_type": "HC1"},
    )
    fig2, ax2 = ivr.plot_ar_confidence_set(cs_nonempty)
    assert ax2.get_xlabel() == r"$\beta$"

    import matplotlib.pyplot as plt

    plt.close(fig)
    plt.close(fig2)


def test_savefig_writes_files(tmp_path: Path) -> None:
    ivr.set_style()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])

    out = tmp_path / "fig"
    paths = ivr.savefig(fig, out, formats=("png", "pdf"))

    for path in paths:
        assert path.exists()


def test_style_context_restores_rcparams() -> None:
    import matplotlib as mpl

    original = mpl.rcParams["axes.facecolor"]
    with style_context():
        assert mpl.rcParams["axes.facecolor"] == "white"
    assert mpl.rcParams["axes.facecolor"] == original
