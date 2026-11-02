# Classification path continuation

C1 and C2 continue through empty quadratic sample sets without adding a ridge.
A KKT multiplier LP first determines how far lambda can decrease while the
coefficients remain fixed. If movement is needed at the endpoint, an optimal-face
LP selects the endpoint favored by decreasing lambda, at the same lambda.
Quadratic sample margins are preserved during a face pivot, covering flat
coefficient directions even when the quadratic sample set is nonempty.

C2 then uses an internal critical-cone direction solve with one-sided curvature
at both margin boundaries. This internal helper introduces no new public
multiclass formulation or interface. The normal curved path continues to use
active-set linear solves. Accepted classification knots must pass actual-loss
KKT checks; unresolved numerical states raise errors.

A face pivot can return distinct minimizers at the same lambda. Interpolation
skips zero-width intervals and uses the outgoing representative. Compare
objectives and KKT residuals, rather than expecting unique coefficients at these
knots. Zero-loss tails and zero lambda-max paths are handled explicitly.

The regression tests cover flat starts, constant-primal intervals, face pivots,
interpolation, intercepts, equality constraints, and zero lambda-max through the
public API. Independent slack-variable QP comparisons cover repeated C2 flat
cells. This is floating-point continuation with numerical certificates, not a
finite-termination guarantee for every degenerate input.

## Entry and zero-coefficient boundary corrections

For an inactive coordinate, the subgradient update is
`s_new = e + lambda/(lambda-delta) * (s_old-e)`. It remains constant when
`e == s_old`, not when `e == -s_old`; skipping the latter can miss an entry.
At an active coefficient equal to zero, a descending direction must satisfy
`sign * d_beta/d_delta >= 0`. If the equality-only active solve violates this
one-sided condition, classification continuation uses the existing critical-cone
direction/event helper and rebuilds its KKT state. The helper handles C1's
squared hinge directly and retains C2's lower margin boundary. An equivalent
rescaling of coefficients, design and lambda conditions the directional solve
without modifying the objective or adding a ridge.

The two-observation notebook examples, nearby correlation ties, column
permutations and feature-unit changes are regression tests, together with
independent squared-hinge QP comparisons for small constrained/unconstrained
problems with and without intercepts. Certificates remain enabled.
