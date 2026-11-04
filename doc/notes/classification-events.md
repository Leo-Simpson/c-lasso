# C1/C2 degenerate path events

This patch targets the existing scalar classification formulations. It adds no
ridge penalty, multiclass formulation, optional dependency, or public API option.

A dual-tight zero coefficient is a candidate entering face. Equality constraints
can make the unconstrained active solve point into the opposite sign. A private
scalar critical-cone pivot selects compatible coefficient signs and outgoing
quadratic sample regions, followed by an analytic, dual-certified segment.
Returned knots retain the existing relative KKT tolerance of 1e-7.

C2 additionally uses the current quadratic sample mean for its intercept and
margin derivatives, handles simultaneous events at rho and one, and continues
flat loss cells using linear programs. Repeated-lambda optimal-face pivots are
handled explicitly during interpolation. The initialization refines the C2
intercept score equation after binary search and removes the implicit numerical
ridge from C2 active systems.

Validation uses synthetic dependent tree columns, actual-margin certificates,
and independent epigraph/slack QPs, with and without an intercept and equality
constraints. The six supplied private student datasets are not included in the
repository. They complete to relative lambda 0.001 for both C1 and C2 with
NumPy 2.4.2 / SciPy 1.17.1. C1 also passed 24 independent objective checks (maximum
relative difference 3.54e-11) with NumPy 2.5.4 / SciPy 1.18.1.

## Stable C2 rank and dual-event calculations

The follow-up removes two numerical causes of the draft failures:

- The curved C2 direction is solved from the reduced quadratic design by SVD,
  without forming its Gram matrix. This reveals weak curvature without
  squaring the condition number, while retaining a true nullspace and checking
  the resulting linear-system residual. No ridge is added.
- Active stationarity determines an affine equality-multiplier trajectory,
  plus any remaining multiplier nullspace. Only inactive coefficient bounds
  enter the dual-event problem. Active saturated bounds must not be imposed
  again as inequalities: tiny roundoff in these duplicates produced false
  negative event times and infeasible event LPs. Unique multipliers give
  analytic event times; nonunique multipliers use a smaller inequality LP.

The captured failing direction and segment both pass after this change.
Synthetic regressions cover weak curvature, true nullspaces, redundant active
bounds, exact inactive entry times, and nonunique equality multipliers.
Returned knots still pass the original relative KKT tolerance of 1e-7;
inconsistent directions and true nullspace inconsistencies still raise.

Final validation: 135 tests pass with NumPy 2.4.2 / SciPy 1.17.1. The 45 focused
C2 tests also pass with NumPy 2.5.4 / SciPy 1.18.1. Independent objective audits
check three path interiors and the endpoint on each of the six supplied cases
in both numerical environments (48 C2 comparisons total). The maximum relative
objective difference is 1.54e-10. These datasets remain private and
are not added to the repository. Production dependencies remain NumPy/SciPy;
CVXPY is used only for the external independent audit.
