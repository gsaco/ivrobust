"""Shared specs and diagnostics dataclasses."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class AssumptionSpec:
    """Assumptions underpinning an inference result."""

    id_regime: str
    error: str
    asymptotics: str
    notes: tuple[str, ...] = ()
    citations: tuple[str, ...] = ()


@dataclass(frozen=True)
class CovSpec:
    """Covariance regime specification."""

    kind: str
    df_adjust: bool = True
    cluster_ids: Optional[np.ndarray] = None


@dataclass
class NumericalDiagnostics:
    """Numerical diagnostics and fallback tracking."""

    condition_numbers: dict[str, float] = field(default_factory=dict)
    ranks: dict[str, int] = field(default_factory=dict)
    min_singular_values: dict[str, float] = field(default_factory=dict)
    used_pinv: bool = False
    pinv_rcond: Optional[float] = None
    notes: list[str] = field(default_factory=list)

    def merge(self, other: NumericalDiagnostics) -> None:
        self.condition_numbers.update(other.condition_numbers)
        self.ranks.update(other.ranks)
        self.min_singular_values.update(other.min_singular_values)
        self.used_pinv = self.used_pinv or other.used_pinv
        if other.pinv_rcond is not None:
            self.pinv_rcond = other.pinv_rcond
        self.notes.extend(other.notes)


@dataclass
class StrengthDiagnostics:
    """Instrument strength diagnostics."""

    first_stage_F: Optional[np.ndarray] = None
    partial_R2: Optional[np.ndarray] = None
    effective_F: Optional[float] = None
    k: Optional[int] = None
    n: Optional[int] = None
    k_over_n: Optional[float] = None
    rank_z: Optional[int] = None
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ReproSpec:
    """Reproducibility metadata."""

    seed: Optional[int] = None
    version: Optional[str] = None
    config: tuple[str, ...] = ()


@dataclass(frozen=True)
class ConfidenceSet:
    """Set-valued confidence region for a scalar parameter."""

    intervals: Sequence[tuple[float, float]]
    alpha: float
    method: str
    grid: np.ndarray
    diagnostics: dict[str, float] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    @property
    def is_empty(self) -> bool:
        return len(self.intervals) == 0

    @property
    def is_unbounded(self) -> bool:
        for lower, upper in self.intervals:
            if np.isneginf(lower) or np.isposinf(upper):
                return True
        return False
