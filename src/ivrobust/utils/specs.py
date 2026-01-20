from __future__ import annotations

from collections.abc import Mapping

from .._typing import IntArray
from ..covariance import CovSpec, CovType, parse_cov_spec
from ..data.clusters import ClusterSpec


def as_cov_spec(
    cov: CovSpec | str | Mapping[str, object] | None,
    *,
    cov_type: CovType | None,
    clusters: IntArray | ClusterSpec | None,
    nobs: int,
) -> CovSpec:
    return parse_cov_spec(cov, cov_type=cov_type, clusters=clusters, nobs=nobs)


__all__ = ["as_cov_spec"]
