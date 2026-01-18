from __future__ import annotations

from pathlib import Path

import numpy as np
import ivrobust as ivr


def main() -> None:
    ivr.set_style()

    outdir = Path("artifacts") / "plot_style_demo"
    outdir.mkdir(parents=True, exist_ok=True)

    # Histogram demo
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    x = rng.standard_normal(2000)

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.hist(x, bins=30)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel("count")
    ax.set_title("Histogram demo")
    ivr.savefig(fig, outdir / "histogram", formats=("png", "pdf"))

    # Line demo
    t = np.linspace(0, 2 * np.pi, 200)
    y = np.sin(t)

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(t, y)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\sin(t)$")
    ax.set_title("Line demo")
    ivr.savefig(fig, outdir / "line", formats=("png", "pdf"))

    # Scatter demo with math label
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.scatter(x[:200], x[200:400], s=18)
    ax.set_xlabel(r"$\hat{\nu}_1$")
    ax.set_ylabel(r"$\hat{\nu}_2$")
    ax.set_title("Scatter demo")
    ivr.savefig(fig, outdir / "scatter", formats=("png", "pdf"))


if __name__ == "__main__":
    main()
