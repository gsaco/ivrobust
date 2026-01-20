# ivrobust

ivrobust provides weak-instrument robust inference for linear instrumental
variable models, with a focus on routines that remain valid when instruments are
weak.

Current release scope:

- Anderson-Rubin (AR), LM/K, and CLR tests for a scalar structural parameter
- Set-valued (possibly disjoint) AR/LM/CLR confidence sets
- HC0/HC1/HC2/HC3 and one-way cluster robust covariance options
- Workflow estimators (2SLS/LIML/Fuller) and diagnostics (first-stage + effective F)

If you are new to weak-IV robust inference, start with the Quickstart and the
AR guide.
