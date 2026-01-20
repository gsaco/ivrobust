from __future__ import annotations

from pathlib import Path

import ivrobust as ivr


def test_quickstart_workflow(tmp_path: Path) -> None:
    data, beta_true = ivr.weak_iv_dgp(
        n=220,
        k=4,
        strength=0.4,
        beta=1.0,
        seed=0,
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
    assert "AR" in res.tests
    assert "CLR" in res.confidence_sets

    fig, _ = res.plot()
    ivr.savefig(fig, tmp_path / "pvalue_curve", formats=("png",))


def test_ar_confidence_set_plot(tmp_path: Path) -> None:
    data, _ = ivr.weak_iv_dgp(
        n=200,
        k=3,
        strength=0.5,
        beta=1.0,
        seed=1,
    )
    cs = ivr.ar_confidence_set(data, alpha=0.05, cov_type="HC1")
    fig, _ = ivr.plot_ar_confidence_set(cs)
    ivr.savefig(fig, tmp_path / "ar_confidence_set", formats=("png",))
