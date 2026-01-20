from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .intervals import IntervalSet


@dataclass(frozen=True)
class TestResult:
    statistic: float
    pvalue: float
    df: int
    method: str
    cov_type: str
    warnings: tuple[str, ...] = ()
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ConfidenceSetResult:
    confidence_set: IntervalSet
    alpha: float
    method: str
    grid_info: dict[str, Any]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class WeakIVInferenceResult:
    tests: dict[str, TestResult]
    confidence_sets: dict[str, ConfidenceSetResult]
    recommended: str
    alpha: float
    cov_type: str
    diagnostics: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    def summary(self) -> str:
        lines: list[str] = []
        lines.append("Weak-IV robust inference")
        lines.append(f"alpha={self.alpha:.3f}, cov_type={self.cov_type}")
        if self.warnings:
            lines.append("warnings: " + "; ".join(self.warnings))

        if self.tests:
            lines.append("")
            lines.append("Tests")
            for name, res in self.tests.items():
                lines.append(
                    f"{name}: stat={res.statistic:.4f}, p={res.pvalue:.4f}, df={res.df}"
                )

        if self.confidence_sets:
            lines.append("")
            lines.append("Confidence sets")
            for name, cs in self.confidence_sets.items():
                lines.append(f"{name}: {cs.confidence_set.intervals}")

        if self.diagnostics:
            lines.append("")
            lines.append("Diagnostics")
            for name, val in self.diagnostics.items():
                lines.append(f"{name}: {val}")

        return "\n".join(lines)

    def plot(self, *, ax=None, methods: tuple[str, ...] | None = None):
        from .plot_style import set_style

        set_style()
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=(6.2, 3.4))
        else:
            fig = ax.figure

        methods_to_plot = methods or tuple(self.confidence_sets.keys())
        for name in methods_to_plot:
            cs = self.confidence_sets.get(name)
            if cs is None:
                continue
            grid = cs.grid_info.get("grid")
            pvals = cs.grid_info.get("pvalues")
            if grid is None or pvals is None:
                continue
            ax.plot(grid, pvals, label=name)

        ax.axhline(self.alpha, color="black", linestyle="--", linewidth=1.0)
        ax.set_xlabel(r"$\beta$")
        ax.set_ylabel("p-value")
        if methods_to_plot:
            ax.legend()
        return fig, ax
