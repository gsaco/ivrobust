# How to Cite

If you use ivrobust in academic research, please cite both the software and the 
methodological references for the specific tests you employ.

---

## Citing ivrobust

<div class="ivr-citation-block" markdown>
<span class="ivr-citation-block__label">BibTeX</span>

```bibtex
@software{ivrobust,
  title     = {ivrobust: Weak-IV Robust Inference in Python},
  author    = {Saco, Gabriel and contributors},
  year      = {2026},
  url       = {https://github.com/gsaco/ivrobust},
  version   = {0.2.0},
  note      = {Python package for Anderson-Rubin, LM, and CLR inference}
}
```

</div>

<div class="ivr-citation-block" markdown>
<span class="ivr-citation-block__label">APA Format</span>

```
Saco, G., & contributors. (2026). ivrobust: Weak-IV robust inference in Python 
(Version 0.2.0) [Computer software]. https://github.com/gsaco/ivrobust
```

</div>

---

## Methodological References

When reporting results from specific tests, please also cite the foundational papers:

### Anderson-Rubin (AR) Test

```bibtex
@article{anderson1949,
  title   = {Estimation of the Parameters of a Single Equation in a 
             Complete System of Stochastic Equations},
  author  = {Anderson, T. W. and Rubin, Herman},
  journal = {The Annals of Mathematical Statistics},
  year    = {1949},
  volume  = {20},
  number  = {1},
  pages   = {46--63}
}
```

### LM/K Test (Kleibergen)

```bibtex
@article{kleibergen2002,
  title   = {Pivotal Statistics for Testing Structural Parameters in 
             Instrumental Variables Regression},
  author  = {Kleibergen, Frank},
  journal = {Econometrica},
  year    = {2002},
  volume  = {70},
  number  = {5},
  pages   = {1781--1803}
}
```

### CLR Test (Moreira)

```bibtex
@article{moreira2003,
  title   = {A Conditional Likelihood Ratio Test for Structural Models},
  author  = {Moreira, Marcelo J.},
  journal = {Econometrica},
  year    = {2003},
  volume  = {71},
  number  = {4},
  pages   = {1027--1048}
}
```

---

## Example Acknowledgment

> We implement weak-IV robust inference using ivrobust (Saco et al., 2026), 
> which provides Anderson-Rubin (Anderson & Rubin, 1949), LM (Kleibergen, 2002), 
> and CLR (Moreira, 2003) tests with heteroskedasticity-robust covariance estimation.

---

## CITATION.cff

ivrobust includes a machine-readable `CITATION.cff` file in the repository root. 
GitHub and most reference managers can parse this format directly.

[View CITATION.cff on GitHub →](https://github.com/gsaco/ivrobust/blob/main/CITATION.cff){ .md-button }
