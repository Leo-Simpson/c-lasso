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

## Review limitation

C2 is not yet ready to merge: additional interleaved reference-solver audits
exposed remaining outgoing-face refinement / segment LP failures on nearly
singular tree designs, including with NumPy 2.4.2 / SciPy 1.17.1 and with
NumPy 2.5.4 / SciPy 1.18.1. The isolated six-case completion run passes, but this
does not establish robustness to other numerical execution contexts.
Certificates reject these cases rather than return an uncertified path.
The draft PR records this explicitly; cross-environment C2 validation must pass
before merging. The production implementation uses NumPy/SciPy only; CVXPY was
used outside the repository for independent numerical auditing.
