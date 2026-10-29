# MC1: incremental implementation

MC1 is the proposed joint multiclass counterpart of C1, using unhalved
squared-hinge loss and the sum of row infinity norms:

    min_B sum_ik max(1 - T_ik (XB)_ik, 0)^2
          + lambda sum_j max_k |B_jk|,  subject to CB=0.

T is +1 for the observed class and -1 otherwise. This is an explicit extension
of binary C1; it is not a pairwise multiclass margin loss. The original memo's
one-hot least-squares objective remains a useful quadratic reference but is
not the classification loss assigned to MC1. Predictions will select the
largest class score. Coefficients need not sum to zero across class columns.

MC2 is reserved for a future Huberized-hinge counterpart of C2; it is not yet
implemented or selectable. Binary two-column MC1 has its own penalty scaling
and must not be advertised as numerically identical to C1 with the same lambda.
For two classes ordered (-1,+1), a symmetric optimum B=(-beta,beta) satisfies
`objective_MC1(B, 2*lambda) = 2*objective_C1(beta, lambda)`. Thus the first
numerical reference benchmark compares at this factor-of-two absolute lambda,
with and without equality constraints. Under unconstrained initialization the
relative lambda convention also agrees. Uniqueness is not assumed in general.
The benchmark solves small MC1 epigraph problems with SciPy SLSQP independently
of C1's path code; it is a future acceptance test for the MC1 path solver, not
evidence that the unfinished homotopy implementation already fits models.

## First increment

`solve_MC1.py` follows the existing `solve_R1.py` module organization and
`problem_R1(data, algo)` naming convention. `problem_MC1((X, C, y), 'Path-Alg')`
stores validated data, sorted class labels, signed targets, zero coefficients,
lambda_max, and a constraint multiplier certificate. Lambda initialization
uses the shared constrained dual-norm routine. `loss_gradient(B)` and
`active_hessians(B, active)` expose the local quadratic model. They do not
construct a full p-by-p Gram matrix.

This is an internal foundation, not a fitted estimator. There is no MC1 branch
in `Classo`, `pathlasso`, or `Formulation.name()` yet. Intercepts, weights and
one-hot input are deliberately deferred. Existing R1–R4 and C1/C2 dispatch is
unchanged. The generic least-squares face-direction helper remains a reference;
its common Hessian cannot directly handle class-specific hinge masks.

## Next increments

The first local path routines now live in `path_alg_MC1.py`:
`derivatives_MC1` uses class-specific Hessians on an already selected fixed
face, and `margin_events_MC1` analytically locates sample-margin crossings.
An identity-design three-class example verifies an exact affine segment and
its stationarity at interior points. These routines reject unresolved current
margin boundaries and singular directions. They do not yet choose initial
faces, release faces, propagate certificates, or traverse a full path.

1. Add a fixed-face KKT solve using class-specific Hessians and explicit
   face multipliers. Validate derivatives and KKT conditions on small problems.
2. Add all event candidates: sample margins, maximum-face ties/releases,
   row exits, and inactive-row dual boundaries. Define simultaneous-event,
   initial-face, and rank-deficiency handling before claiming an exact path.
3. Add `Classo_MC1` and `pathlasso_MC1` with the existing relative-lambda
   convention. Verify against independent convex solutions, including coupled
   tree constraints and tied entries; then measure runtime for large p.
4. Integrate compact dispatch and high-level formulation selection, followed
   by multiclass prediction and model selection with stratified folds.
5. Consider MC2 after MC1 is numerically and behaviorally stable.

Each increment should be separately reviewable. Retain original code idioms
where useful, with clearer validation and numerical checks for the new code.

## Reference-guided extension

The supplied references are Schmelzer and Hastie (2026),
https://arxiv.org/abs/2609.25704, and Hobson, Butcher and Lentati (2026),
https://arxiv.org/abs/2610.17024. Their PDFs are stored under `assets` in the
parent development workspace. The first supplies the bordered segment and
primal/dual ratio-test perspective; the second makes the directional release
decision and certificate trajectory explicit. We must not treat every tight
inequality as binding on the outgoing segment or arbitrarily recertify at
every kink and infer termination. The second paper's positive-definite-Hessian
assumption is not automatic for MC1. No ridge is added implicitly here.

For our epigraph extension, set z=(vec(B_A),t_A) and q=(0,1). On a fixed
sample region, the coefficient Hessian consists of the K blocks
`H_k=2 X[F_k,A].T X[F_k,A]`, with zero curvature on t. E stacks tree
equalities and selected maximum-face equalities `s*B[j,k]-t[j]=0`.
Differentiating stationarity gives

    [ H  E.T ] [ z_dot  ] = [ -q ]
    [ E   0  ] [ nu_dot ]   [  0 ]

This is our MC1 derivation, not a theorem claimed verbatim from either paper.
It is valid within a selected nondegenerate region; selecting that region is
the next problem. A full traversal additionally needs physical nonnegative
face multipliers, inactive-row dual L1 bounds, their piecewise-affine sign
changes, and a justified degeneracy/certificate rule. A positive number of
tested affine pieces does not establish a finite traversal theorem.

Binary reference tests use independent small epigraph solves. Their objectives
are scaled by sample count to improve SLSQP conditioning, without changing
the minimizer. The coefficient comparison to existing C1 uses 1e-4 tolerance
because its sample-margin events are snapped within 1e-4; objective comparisons
remain tighter. This baseline tolerance must not become the accuracy target
for the new MC1 homotopy algorithm.
