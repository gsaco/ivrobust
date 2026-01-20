from __future__ import annotations

from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import ivrobust as ivr


OUTPUT_DIR = Path("docs/assets/figures")


def build_ar_confidence_set() -> None:
    data, beta_true = ivr.weak_iv_dgp(
        n=260,
        k=5,
        strength=0.45,
        beta=1.0,
        seed=2,
    )
    cs = ivr.ar_confidence_set(
        data,
        alpha=0.05,
        cov_type="HC1",
        beta_bounds=(beta_true - 2.0, beta_true + 2.0),
        n_grid=401,
    )
    fig, ax = ivr.plot_ar_confidence_set(cs)
    ax.set_title("AR 95% confidence set")
    ivr.savefig(fig, OUTPUT_DIR / "ar_confidence_set", dpi=500)


def build_pvalue_curve() -> None:
    data, beta_true = ivr.weak_iv_dgp(
        n=300,
        k=6,
        strength=0.5,
        beta=1.0,
        seed=3,
    )
    res = ivr.weakiv_inference(
        data,
        beta0=beta_true,
        alpha=0.05,
        methods=("AR", "LM", "CLR"),
        cov_type="HC1",
        grid=(beta_true - 2.0, beta_true + 2.0, 301),
        return_grid=True,
    )
    fig, ax = res.plot()
    ax.set_title("Weak-IV p-value curves")
    ivr.savefig(fig, OUTPUT_DIR / "pvalue_curve", dpi=500)


def build_simulation_illustration() -> None:
    rng = np.random.default_rng(7)
    strengths = np.array([0.2, 0.35, 0.5, 0.7, 0.9])
    n_rep = 60
    alpha = 0.05
    beta_true = 1.0
    beta0 = 0.0

    ivr.set_style()
    rates = []
    for strength in strengths:
        rejects = 0
        for _ in range(n_rep):
            seed = int(rng.integers(0, 1_000_000))
            data, _ = ivr.weak_iv_dgp(
                n=220,
                k=5,
                strength=float(strength),
                beta=beta_true,
                seed=seed,
            )
            pval = ivr.ar_test(data, beta0=beta0, cov_type="HC1").pvalue
            rejects += int(pval < alpha)
        rates.append(rejects / n_rep)

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot(strengths, rates, marker="o")
    ax.axhline(alpha, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Instrument strength")
    ax.set_ylabel("Rejection rate")
    ax.set_title("Simulation: AR rejection rates")
    ax.set_ylim(0.0, 1.0)
    ivr.savefig(fig, OUTPUT_DIR / "simulation_rejection", dpi=500)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    build_ar_confidence_set()
    build_pvalue_curve()
    build_simulation_illustration()


if __name__ == "__main__":
    main()
