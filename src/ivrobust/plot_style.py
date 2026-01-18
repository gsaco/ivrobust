from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterable


_STYLE_RC: dict[str, object] = {
    # Figure + axes
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "black",
    "axes.linewidth": 1.0,
    "axes.grid": False,
    "axes.axisbelow": True,
    # Typography
    "font.size": 11.0,
    "axes.labelsize": 11.0,
    "xtick.labelsize": 10.0,
    "ytick.labelsize": 10.0,
    "legend.fontsize": 10.0,
    # Ticks
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 4.0,
    "ytick.major.size": 4.0,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    # Lines / patches
    "lines.linewidth": 1.5,
    "patch.facecolor": "#4D4D4D",
    "patch.edgecolor": "black",
    "patch.linewidth": 0.9,
    # Saving
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    # Vector-friendly text in PDF/PS
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    # Math text (LaTeX-like without requiring LaTeX)
    "mathtext.fontset": "dejavusans",
}

def _apply_style(mpl) -> None:
    mpl.rcParams.update(_STYLE_RC)
    try:
        from cycler import cycler
    except ImportError:  # pragma: no cover
        return
    mpl.rcParams["axes.prop_cycle"] = cycler(color=["#4D4D4D"])


def set_style() -> None:
    """
    Apply the ivrobust plotting style globally (matplotlib rcParams).

    This is the single style entrypoint. All plotting functions in ivrobust call
    `set_style()` internally to prevent drift across the package.
    """
    try:
        import matplotlib as mpl
    except ImportError as e:  # pragma: no cover
        raise ImportError("Plotting requires matplotlib. Install ivrobust[plot].") from e

    _apply_style(mpl)


@contextmanager
def style_context():
    """
    Context manager that applies ivrobust style and restores previous rcParams.
    """
    try:
        import matplotlib as mpl
    except ImportError as e:  # pragma: no cover
        raise ImportError("Plotting requires matplotlib. Install ivrobust[plot].") from e

    old = mpl.rcParams.copy()
    _apply_style(mpl)
    try:
        yield
    finally:
        mpl.rcParams.update(old)


def savefig(
    fig,
    path: str | Path,
    *,
    formats: Iterable[str] = ("png", "pdf"),
    dpi: int | None = None,
) -> list[Path]:
    """
    Save a figure with ivrobust conventions.

    Parameters
    ----------
    fig
        Matplotlib Figure.
    path
        Output path without suffix (recommended) or with suffix.
    formats
        Iterable of formats, e.g. ("png", "pdf").
    dpi
        Override DPI for raster formats. Default uses rcParams (300).

    Returns
    -------
    list[Path]
        Paths written.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:  # pragma: no cover
        raise ImportError("Plotting requires matplotlib. Install ivrobust[plot].") from e

    set_style()

    out_base = Path(path)
    out_base.parent.mkdir(parents=True, exist_ok=True)

    # Tight layout without surprises
    fig.tight_layout()

    written: list[Path] = []
    for fmt in formats:
        fmt_clean = fmt.lower().lstrip(".")
        out = out_base.with_suffix(f".{fmt_clean}")
        kwargs: dict[str, object] = {}
        if dpi is not None and fmt_clean in {"png", "jpg", "jpeg", "tif", "tiff"}:
            kwargs["dpi"] = int(dpi)

        fig.savefig(out, **kwargs)
        written.append(out)

    plt.close(fig)
    return written
