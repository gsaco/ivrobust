from __future__ import annotations

from typing import Sequence, TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]
ArrayLike1D: TypeAlias = Sequence[float] | NDArray[np.floating[np.float64]]
