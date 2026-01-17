"""Custom warning classes and helpers."""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from dataclasses import dataclass, field


class IVRobustWarning(UserWarning):
    """Base warning class for ivrobust."""


class NumericalWarning(IVRobustWarning):
    """Warning for numerical issues or fallbacks."""


class DataWarning(IVRobustWarning):
    """Warning for data validation or missingness issues."""


class CovarianceWarning(IVRobustWarning):
    """Warning for covariance regime issues."""


class ClusterWarning(IVRobustWarning):
    """Warning for clustered inference limitations."""


@dataclass
class WarningRecord:
    """Collect warnings emitted during a computation."""

    messages: list[str] = field(default_factory=list)

    def add(self, message: str) -> None:
        self.messages.append(message)

    def extend(self, messages: Iterable[str]) -> None:
        self.messages.extend(messages)


def warn_and_record(
    message: str,
    category: type[Warning] = IVRobustWarning,
    record: WarningRecord | None = None,
) -> None:
    """Emit a warning and optionally record it."""

    warnings.warn(message, category=category, stacklevel=2)
    if record is not None:
        record.add(message)
