# References

<p class="ivr-section__lead">
The following references are foundational for weak-instrument robust inference 
and the methods implemented in ivrobust. We encourage users to cite the relevant 
methodological papers when reporting results.
</p>

---

## Core Methodological Papers

<div class="ivr-methods-grid" markdown>

<div class="ivr-method-card" markdown>
<div class="ivr-method-card__icon">📚</div>

### Anderson & Rubin (1949)

**Estimation of the Parameters of a Single Equation in a Complete System of Stochastic Equations**

*The Annals of Mathematical Statistics*, 20(1), 46–63.

The foundational paper introducing the Anderson-Rubin test for structural parameters in simultaneous equations models.

[DOI: 10.1214/aoms/1177730090](https://doi.org/10.1214/aoms/1177730090)
</div>

<div class="ivr-method-card" markdown>
<div class="ivr-method-card__icon">📚</div>

### Kleibergen (2002)

**Pivotal Statistics for Testing Structural Parameters in Instrumental Variables Regression**

*Econometrica*, 70(5), 1781–1803.

Introduces the LM (K) statistic based on the score function, providing weak-IV robust inference with improved power properties.

[DOI: 10.1111/1468-0262.00353](https://doi.org/10.1111/1468-0262.00353)
</div>

<div class="ivr-method-card" markdown>
<div class="ivr-method-card__icon">📚</div>

### Moreira (2003)

**A Conditional Likelihood Ratio Test for Structural Models**

*Econometrica*, 71(4), 1027–1048.

Develops the conditional likelihood ratio test that achieves near-optimal power while maintaining weak-IV robustness.

[DOI: 10.1111/1468-0262.00438](https://doi.org/10.1111/1468-0262.00438)
</div>

</div>

---

## Confidence Sets and Diagnostics

<div class="ivr-methods-grid" markdown>

<div class="ivr-method-card" markdown>
<div class="ivr-method-card__icon">📐</div>

### Mikusheva (2010)

**Robust Confidence Sets in the Presence of Weak Instruments**

*Journal of Econometrics*, 157(2), 236–247.

Theory for set-valued confidence sets under weak identification, including handling of disjoint and unbounded intervals.

[DOI: 10.1016/j.jeconom.2009.12.003](https://doi.org/10.1016/j.jeconom.2009.12.003)
</div>

<div class="ivr-method-card" markdown>
<div class="ivr-method-card__icon">📊</div>

### Montiel Olea & Pflueger (2013)

**A Robust Test for Weak Instruments**

*Journal of Business & Economic Statistics*, 31(3), 358–369.

The effective F-statistic for heteroskedasticity-robust weak instrument diagnostics.

[DOI: 10.1080/00401706.2013.806694](https://doi.org/10.1080/00401706.2013.806694)
</div>

<div class="ivr-method-card" markdown>
<div class="ivr-method-card__icon">🔧</div>

### Stock & Yogo (2005)

**Testing for Weak Instruments in Linear IV Regression**

*Identification and Inference for Econometric Models*, Cambridge University Press, 80–108.

Critical values for weak instrument testing and guidelines for first-stage diagnostics.
</div>

</div>

---

## Implementations and Reviews

<div class="ivr-methods-grid" markdown>

<div class="ivr-method-card" markdown>
<div class="ivr-method-card__icon">💻</div>

### Finlay & Magnusson (2009)

**Implementing Weak-Instrument Robust Tests for a General Class of Instrumental-Variables Models**

*The Stata Journal*, 9(3), 398–421.

Practical implementation guidance and Stata commands for weak-IV robust inference.

[DOI: 10.1177/1536867X0900900304](https://doi.org/10.1177/1536867X0900900304)
</div>

<div class="ivr-method-card" markdown>
<div class="ivr-method-card__icon">📖</div>

### Andrews, Stock & Sun (2019)

**Weak Instruments in Instrumental Variables Regression: Theory and Practice**

*Annual Review of Economics*, 11, 727–753.

Comprehensive modern review of weak instruments, recommended practices, and recent developments.

[DOI: 10.1146/annurev-economics-080218-025643](https://doi.org/10.1146/annurev-economics-080218-025643)
</div>

</div>

---

## See Also

- [How to Cite](how-to-cite.md) — BibTeX entries for citing ivrobust
- [Methods at a Glance](methods/index.md) — Overview of implemented methods
