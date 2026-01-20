from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from scipy.stats import norm

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import ivrobust as ivr


FIGURES_DIR = Path("docs-src/assets/figures")
DATA_DIR = Path("docs-src/assets/data")


def _save(fig: plt.Figure, name: str) -> None:
    ivr.savefig(fig, FIGURES_DIR / name, dpi=500)


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
    _save(fig, "ar_confidence_set")


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
    _save(fig, "pvalue_curve")


def build_rejection_vs_strength() -> None:
    rng = np.random.default_rng(11)
    strengths = np.array([0.15, 0.25, 0.35, 0.5, 0.7, 0.9])
    n_rep = 80
    alpha = 0.05
    beta_true = 1.0

    ar_rates = []
    tsls_rates = []
    for strength in strengths:
        ar_reject = 0
        tsls_reject = 0
        for _ in range(n_rep):
            seed = int(rng.integers(0, 1_000_000))
            data, _ = ivr.weak_iv_dgp(
                n=240,
                k=5,
                strength=float(strength),
                beta=beta_true,
                seed=seed,
            )
            ar_pval = ivr.ar_test(data, beta0=beta_true, cov_type="HC1").pvalue
            ar_reject += int(ar_pval < alpha)

            tsls_res = ivr.tsls(data, cov_type="HC1")
            t_stat = (tsls_res.beta - beta_true) / tsls_res.stderr[-1, 0]
            tsls_pval = 2.0 * norm.sf(abs(float(t_stat)))
            tsls_reject += int(tsls_pval < alpha)

        ar_rates.append(ar_reject / n_rep)
        tsls_rates.append(tsls_reject / n_rep)

    ar_rates = np.asarray(ar_rates)
    tsls_rates = np.asarray(tsls_rates)

    band = np.sqrt(np.clip(ar_rates * (1.0 - ar_rates) / n_rep, 0.0, 1.0))

    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    ax.plot(strengths, ar_rates, marker="o", label="AR (robust)")
    ax.plot(strengths, tsls_rates, marker="s", label="2SLS t-test")
    ax.fill_between(
        strengths,
        np.clip(ar_rates - 1.96 * band, 0.0, 1.0),
        np.clip(ar_rates + 1.96 * band, 0.0, 1.0),
        color="#9ca3af",
        alpha=0.25,
        linewidth=0.0,
        label="AR 95% MC band",
    )
    ax.axhline(alpha, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Instrument strength")
    ax.set_ylabel("Rejection rate at true beta")
    ax.set_title("Robust vs conventional rejection rates")
    ax.set_ylim(0.0, 1.0)
    ax.legend(frameon=False)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        DATA_DIR / "rejection_vs_strength.npz",
        strengths=strengths,
        ar_rates=ar_rates,
        tsls_rates=tsls_rates,
        n_rep=n_rep,
        alpha=alpha,
    )

    _save(fig, "rejection_vs_strength")


def build_runtime_scaling() -> None:
    rng = np.random.default_rng(31)
    configs = [
        (150, 3),
        (250, 5),
        (400, 7),
    ]
    timings = []
    for n, k in configs:
        data, beta_true = ivr.weak_iv_dgp(
            n=n,
            k=k,
            strength=0.4,
            beta=1.0,
            seed=int(rng.integers(0, 1_000_000)),
        )
        start = time.perf_counter()
        _ = ivr.weakiv_inference(
            data,
            beta0=beta_true,
            alpha=0.05,
            methods=("AR", "LM", "CLR"),
            cov_type="HC1",
            grid=(beta_true - 1.5, beta_true + 1.5, 301),
            return_grid=False,
        )
        timings.append(time.perf_counter() - start)

    timings = np.asarray(timings)

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    x_labels = [f"n={n}, k={k}" for n, k in configs]
    ax.plot(range(len(configs)), timings, marker="o")
    ax.set_xticks(range(len(configs)), x_labels, rotation=0)
    ax.set_ylabel("Runtime (seconds)")
    ax.set_title("Weak-IV inference runtime scaling")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        DATA_DIR / "runtime_scaling.npz",
        configs=np.array(configs),
        timings=timings,
    )

    _save(fig, "runtime_scaling")


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ivr.set_style()

    build_ar_confidence_set()
    build_pvalue_curve()
    build_rejection_vs_strength()
    build_runtime_scaling()


if __name__ == "__main__":
    main()
