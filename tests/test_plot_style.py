import numpy as np

import ivrobust as ivr


def test_set_style_sets_expected_rcparams() -> None:
    ivr.set_style()
    import matplotlib as mpl

    assert mpl.rcParams["figure.facecolor"] == "white"
    assert mpl.rcParams["axes.facecolor"] == "white"
    assert mpl.rcParams["patch.edgecolor"] == "black"
    assert mpl.rcParams["savefig.dpi"] == 300


def test_histogram_defaults_are_monochrome() -> None:
    ivr.set_style()
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    x = rng.standard_normal(1000)

    fig, ax = plt.subplots()
    _, _, patches = ax.hist(x, bins=10)

    # Patch facecolor should be grayscale (RGB channels equal)
    fc = patches[0].get_facecolor()
    assert abs(fc[0] - fc[1]) < 1e-12
    assert abs(fc[1] - fc[2]) < 1e-12

    # Edgecolor should be black-ish
    ec = patches[0].get_edgecolor()
    assert ec[0] <= 0.05 and ec[1] <= 0.05 and ec[2] <= 0.05

    plt.close(fig)
