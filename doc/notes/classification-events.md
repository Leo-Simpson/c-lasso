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

## Avoid redundant flat-face searches

A history of flat events no longer triggers LPs before every later segment.
C2 first attempts a certified curved segment; only a failed direction/segment
falls back to the optimal-face LP. A genuinely empty quadratic sample set still
uses the flat-face solver directly. Failed attempts restore the primal state
and rebuild derived state before the fallback. Certificate tolerances and
returned-knot checks are unchanged.

On the 143-by-353 supplied tree design, the instrumented long C2 path to relative
lambda 0.001 fell from about 48 seconds to 2.9 seconds, and flat-face calls fell
from 101 to zero. These are single-run development measurements with one BLAS
thread. Paths can choose different representatives at degenerate knots; both
paths were independently objective-checked. The active-set-cap-30 path takes
about 0.6 seconds and already required no flat-face LPs.

## Rounded optimal faces and tied inactive bounds (2026-11-13)

A tolerance-based flat-cell partition can include a sample whose current margin
is just inside the neighboring curved cell. Requiring an exact flat-cell bound
then excludes the certified starting point from its own pivot LP. Only these
rounded boundary bounds are anchored to the current score: the upper flat cell
uses `margin >= min(1, current_margin)` and the lower C2 cell uses
`margin <= max(rho, current_margin)`. All other bounds remain unchanged, and a
candidate still passes the existing original-loss KKT check at tolerance 1e-7.
Using an exact partition everywhere was tested and rejected because rounded C2
knots can then prevent valid continuation.

An inactive bound tied throughout a segment has mathematically zero slack and
zero rate. Floating-point residues in both could divide into a large negative
event time. The unique-multiplier analytic event solver now bounds this
uncertainty using active stationarity residuals and multiplier arithmetic.
It ignores only rates indistinguishable from roundoff, and clamps only negative
slack within that bound. Endpoint KKT checks remain unchanged. A separate test
retains small genuine inactive-entry rates.

Three supplied numerical-only paths are retained in the November 13 fixtures.
The zero-length error reproduces with NumPy 1.24.4 / SciPy 1.10.1, and the trac
pivot error also reproduces with NumPy 2.4.2 / SciPy 1.17.1. The reported outgoing
QP error passes locally in both environments; its full path remains a regression
without claiming an independently reproduced root cause. The older numerical
stack was tested on Python 3.11, rather than the reported Python 3.8.
