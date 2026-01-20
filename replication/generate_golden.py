import json
from pathlib import Path

import numpy as np

import ivrobust as ivr


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    data_dir = out_dir / "data"
    outputs_dir = out_dir / "outputs"
    data_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    data, beta_true = ivr.weak_iv_dgp(n=120, k=3, strength=0.6, beta=1.1, seed=123)

    # Save fixture CSV
    cols = ["y", "d"]
    cols += [f"z{i+1}" for i in range(data.z.shape[1])]
    cols += [f"x{i+1}" for i in range(data.x.shape[1])]
    mat = np.hstack([data.y, data.d, data.z, data.x])
    np.savetxt(data_dir / "weak_iv_fixture.csv", mat, delimiter=",", header=",".join(cols), comments="")

    res_tsls = ivr.tsls(data, cov_type="HC1")
    res_liml = ivr.liml(data, cov_type="HC1")
    res_fuller = ivr.fuller(data, alpha=1.0, cov_type="HC1")
    ar = ivr.ar_test(data, beta0=beta_true, cov_type="HC1")
    lm = ivr.lm_test(data, beta0=beta_true, cov_type="HC1")
    clr = ivr.clr_test(data, beta0=beta_true, cov_type="HC1")

    payload = {
        "beta_true": beta_true,
        "nobs": data.nobs,
        "k_instr": data.k_instr,
        "tsls": {"beta": float(res_tsls.beta), "se": float(res_tsls.stderr[-1, 0])},
        "liml": {"beta": float(res_liml.beta), "se": float(res_liml.stderr[-1, 0])},
        "fuller": {"beta": float(res_fuller.beta), "se": float(res_fuller.stderr[-1, 0])},
        "ar": {"stat": float(ar.statistic), "pvalue": float(ar.pvalue)},
        "lm": {"stat": float(lm.statistic), "pvalue": float(lm.pvalue)},
        "clr": {"stat": float(clr.statistic), "pvalue": float(clr.pvalue)},
    }

    with (outputs_dir / "golden.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
