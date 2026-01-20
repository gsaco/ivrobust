* ivrobust replication placeholder
* Requires ivreg2. Update paths as needed.

clear all
import delimited "../data/weak_iv_fixture.csv", varnames(1)

* Example IV regression
ivreg2 y (d = z1 z2 z3) x1, robust

* Save outputs manually for golden tables.
