from __future__ import annotations

import numpy as np

from .ar import ARConfidenceSetResult
from .plot_style import set_style


def plot_ar_confidence_set(cs: ARConfidenceSetResult, *, ax=None):
    """
    Plot an AR confidence set as horizontal intervals.

    Parameters
    ----------
    cs
        Output from `ar_confidence_set`.
    ax
        Optional matplotlib Axes.

    Returns
    -------
    (fig, ax)
    """
    set_style()
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.0, 1.6))
    else:
        fig = ax.figure

    intervals = cs.confidence_set.intervals
    if not intervals:
        ax.text(0.5, 0.5, "Empty confidence set", ha="center", va="center")
        ax.set_axis_off()
        return fig, ax

    y0 = 0.0
    for (lo, hi) in intervals:
        x1 = lo if np.isfinite(lo) else float(np.min(cs.grid)) - 0.5
        x2 = hi if np.isfinite(hi) else float(np.max(cs.grid)) + 0.5
        ax.plot([x1, x2], [y0, y0], solid_capstyle="butt")
        ax.scatter([x1, x2], [y0, y0], s=18)

    ax.set_yticks([])
    ax.set_xlabel(r"$\beta$")
    ax.set_title(
        f"AR {(1.0 - cs.alpha):.0%} confidence set (df={cs.df}, {cs.cov_type})"
    )
    return fig, ax
