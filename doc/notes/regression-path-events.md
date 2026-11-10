# R1/R2 scalar path corrections

R1/R2 follow their stated objectives without an implicit numerical ridge.
R1 retains its ordinary active linear-system update; incompatible zero-entry
signs or a failed returned-knot KKT check route to the shared cone/event solver.
R3's legacy least-squares updater is unchanged.

R2 uses the same scalar direction and event primitives through a loss adapter.
Writing r = X beta + intercept - y, squared loss is the sum of squared positive
parts of r and -r. Huber loss is the same pair with each quadratic continued
linearly beyond rho. Fixed offsets supply this identity to the shared solver;
coefficient and constraint dimensions do not increase. No new dependency or
multiclass/ridge formulation is introduced.

The adapter enforces equality constraints and compatible entering signs,
handles simultaneous sample and coefficient events, tracks actual residuals,
and uses outgoing one-sided curvature at the boundaries. It replaces the old
R2 update, including its e+s event-skip test, full-data intercept derivative,
single-sample switches, and scale-dependent zero-crossing suppression. Empty
quadratic sets continue through a Huber optimal-face LP instead of returning
an incomplete path. LPs are used only for failed curved segments or empty
quadratic sets. Returned-knot certificates retain tolerance 1e-7.

Reproduced failures corrected:

- For X=[0.01], y=[1] and relative lambda 0.5, R1 returns beta=50 instead
  of 4.5454545. The exact objective is (0.01 beta-1)^2+0.01|beta|.
- For X=[1,-1], y=[2,-2], rho=1 and relative lambda 0.1, R2 returns
  beta=1.9 instead of zero; its path now reaches the requested endpoint.
- R2's stored residuals agree with X beta+intercept-y as the quadratic
  sample set changes.

Twenty regression tests cover the analytic examples, zero lambda maximum,
and independent objective checks with/without intercepts, equality
constraints, dependent columns, and repeated samples. The tests pass with
NumPy 2.4.2 / SciPy 1.17.1 and NumPy 2.5.4 / SciPy 1.18.1.
The external 88-soils check uses an active-set cap of 30: R1 takes about 0.10 s
and R2 about 1.4 s (single development runs, one BLAS thread). Eight independent
convex objective comparisons have maximum relative difference 2.26e-10.
Private/intermediate audit data and optional reference solvers are not shipped.
