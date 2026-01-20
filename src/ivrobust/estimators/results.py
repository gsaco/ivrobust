from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .._typing import FloatArray
from ..data import IVData


@dataclass(frozen=True)
class IVResults:
    params: FloatArray
    vcov: FloatArray
    stderr: FloatArray
    cov_type: str
    cov_config: dict[str, Any]
    nobs: int
    df_resid: int
    k_endog: int
    k_instr: int
    k_exog: int
    method: str
    data: IVData | None = None
    first_stage: Any | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def beta(self) -> float:
        return float(self.params[-1, 0])

    def as_dict(self) -> dict[str, Any]:
        return {
            "params": self.params,
            "vcov": self.vcov,
            "stderr": self.stderr,
            "cov_type": self.cov_type,
            "cov_config": dict(self.cov_config),
            "nobs": self.nobs,
            "df_resid": self.df_resid,
            "k_endog": self.k_endog,
            "k_instr": self.k_instr,
            "k_exog": self.k_exog,
            "method": self.method,
            "diagnostics": self.diagnostics,
        }

    def summary(self) -> str:
        lines = [
            f"IVResults ({self.method})",
            f"nobs={self.nobs}, df_resid={self.df_resid}, cov_type={self.cov_type}",
        ]
        for i, (b, se) in enumerate(
            zip(self.params.ravel(), self.stderr.ravel(), strict=False)
        ):
            lines.append(f"beta[{i}]= {b:.4f} (se={se:.4f})")
        if self.diagnostics:
            lines.append("diagnostics:")
            for k, v in self.diagnostics.items():
                lines.append(f"  {k}: {v}")
        return "\n".join(lines)

    def to_dataframe(self) -> Any:
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover
            raise ImportError("pandas is required for to_dataframe().") from exc
        return pd.DataFrame(
            {
                "param": np.arange(self.params.size),
                "estimate": self.params.ravel(),
                "stderr": self.stderr.ravel(),
            }
        )

    def to_latex(self) -> str:
        df = self.to_dataframe()
        return str(df.to_latex(index=False))


TSLSResult = IVResults

__all__ = ["IVResults", "TSLSResult"]
