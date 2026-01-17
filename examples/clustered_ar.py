"""Cluster-robust Anderson-Rubin example."""

import ivrobust as ivr


def main() -> None:
    data, beta_true = ivr.weak_iv_dgp(
        n=300, k=5, strength=0.4, beta=1.0, seed=1, n_clusters=30
    )
    ar = ivr.ar_test(data, beta0=[beta_true], cov="HC1")
    print("AR stat:", ar.statistic)
    print("AR p-value:", ar.pvalue)


if __name__ == "__main__":
    main()
