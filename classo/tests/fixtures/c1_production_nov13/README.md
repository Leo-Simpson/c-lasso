# Classification-path regressions supplied 2026-11-13

Three numerical-only C1 inputs supplied by users, with intercepts and zero-sum
constraints. Cases 1 and 2 are synthetic; case 3 is an anonymized real trac design
with no sample or feature names. Each archive includes X, C, y, lambdas, intercept
and huber flags. The reported environment is NumPy 1.24.4 / SciPy 1.10.1.

The reported errors on maintenance commit d22da8d are outgoing-cone KKT,
zero-length segment and infeasible optimal-face pivots. The latter two reproduce
with NumPy 1.24.4 / SciPy 1.10.1 on Python 3.11; the first passes locally.
Some failures depend on numerical-library versions and platform. Tests check the original loss and constraints,
rather than merely accepting that the solver returns without an exception.
