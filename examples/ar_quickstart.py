"""Minimal Anderson-Rubin example."""

import ivrobust as ivr


def main() -> None:
    data, beta_true = ivr.weak_iv_dgp(n=300, k=5, strength=0.4, beta=1.0, seed=0)
    ar = ivr.ar_test(data, beta0=[beta_true])
    print("AR stat:", ar.statistic)
    print("AR p-value:", ar.pvalue)

    cs = ivr.ar_confidence_set(data, alpha=0.05)
    print("AR confidence set:", cs.confidence_set.intervals)


if __name__ == "__main__":
    main()
