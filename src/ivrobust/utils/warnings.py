from __future__ import annotations

import warnings
from enum import Enum


class IVRobustWarning(RuntimeWarning):
    pass


class WarningCategory(str, Enum):
    MANY_INSTRUMENTS = "many_instruments"
    WEAK_ID = "weak_id"
    RANK_DEFICIENT = "rank_deficient"


def warn(category: WarningCategory, message: str) -> None:
    warnings.warn(f"{category.value}: {message}", IVRobustWarning, stacklevel=2)
