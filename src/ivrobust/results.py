from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .intervals import IntervalSet


@dataclass(frozen=True)
class TestResult:
    statistic: float
    pvalue: float
    df: int | tuple[int, int] | None
    method: str
    cov_type: str
    alpha: float | None = None
    cov_config: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def metadata(self) -> dict[str, Any]:
        return self.details

    def as_dict(self) -> dict[str, Any]:
        return {
            "statistic": self.statistic,
            "pvalue": self.pvalue,
            "df": self.df,
            "method": self.method,
            "cov_type": self.cov_type,
            "alpha": self.alpha,
            "cov_config": dict(self.cov_config),
            "warnings": self.warnings,
            "details": dict(self.details),
        }

    def summary(self) -> str:
        lines = [f"{self.method} test"]
        lines.append(f"stat={self.statistic:.4f}, p={self.pvalue:.4f}, df={self.df}")
        lines.append(f"cov_type={self.cov_type}")
        if self.alpha is not None:
            lines.append(f"alpha={self.alpha:.3f}")
        if self.warnings:
            lines.append("warnings: " + "; ".join(self.warnings))
        return "\n".join(lines)

    def to_dataframe(self) -> Any:
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover
            raise ImportError("pandas is required for to_dataframe().") from exc
        return pd.DataFrame(
            [
                {
                    "statistic": self.statistic,
                    "pvalue": self.pvalue,
                    "df": self.df,
                    "method": self.method,
                    "cov_type": self.cov_type,
                    "alpha": self.alpha,
                }
            ]
        )

    def to_latex(self) -> str:
        df = self.to_dataframe()
        return str(df.to_latex(index=False))


@dataclass(frozen=True)
class ConfidenceSetResult:
    confidence_set: IntervalSet
    alpha: float
    method: str
    grid_info: dict[str, Any]
    warnings: tuple[str, ...] = ()

    @property
    def intervals(self) -> list[tuple[float, float]]:
        return self.confidence_set.intervals

    @property
    def is_empty(self) -> bool:
        return len(self.intervals) == 0

    @property
    def is_unbounded(self) -> bool:
        return any(
            (not (lo > float("-inf")) or not (hi < float("inf")))
            for lo, hi in self.intervals
        )

    @property
    def is_disjoint(self) -> bool:
        return len(self.intervals) > 1

    @property
    def grid_diagnostics(self) -> dict[str, Any]:
        return self.grid_info

    def as_dict(self) -> dict[str, Any]:
        return {
            "intervals": list(self.intervals),
            "alpha": self.alpha,
            "method": self.method,
            "grid_info": dict(self.grid_info),
            "warnings": self.warnings,
        }

    def summary(self) -> str:
        lines = [f"{self.method} confidence set"]
        lines.append(f"alpha={self.alpha:.3f}")
        lines.append(f"intervals={self.intervals}")
        if self.warnings:
            lines.append("warnings: " + "; ".join(self.warnings))
        return "\n".join(lines)

    def to_dataframe(self) -> Any:
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover
            raise ImportError("pandas is required for to_dataframe().") from exc
        return pd.DataFrame(self.intervals, columns=["lower", "upper"])

    def to_latex(self) -> str:
        df = self.to_dataframe()
        return str(df.to_latex(index=False))

    def plot(self, *, ax: Any | None = None) -> tuple[Any, Any]:
        from .plots import plot_ar_confidence_set

        if self.method.upper() == "AR":
            return plot_ar_confidence_set(self, ax=ax)

        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=(6.0, 1.8))
        else:
            fig = ax.figure

        for lo, hi in self.intervals:
            ax.plot([lo, hi], [0.0, 0.0], solid_capstyle="butt")
        ax.set_yticks([])
        ax.set_xlabel(r"$\beta$")
        ax.set_title(f"{self.method} confidence set")
        return fig, ax


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
                lines.append(f"{name}: {cs.intervals}")

        if self.diagnostics:
            lines.append("")
            lines.append("Diagnostics")
            for name, val in self.diagnostics.items():
                lines.append(f"{name}: {val}")

        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "tests": {k: v.as_dict() for k, v in self.tests.items()},
            "confidence_sets": {
                k: v.as_dict() for k, v in self.confidence_sets.items()
            },
            "recommended": self.recommended,
            "alpha": self.alpha,
            "cov_type": self.cov_type,
            "diagnostics": self.diagnostics,
            "warnings": self.warnings,
        }

    def plot(
        self, *, ax: Any | None = None, methods: tuple[str, ...] | None = None
    ) -> tuple[Any, Any]:
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
