# MC1: joint multiclass squared-hinge path

MC1 extends C1 to signed multiclass targets T (observed class +1, others -1):

    min_B sum_ik max(1 - T_ik (XB)_ik, 0)^2
          + lambda sum_j w_j max_k |B_jk|, subject to CB=0.

The loss is unhalved. All weights must be finite and strictly positive. Classes
are sorted by `numpy.unique`; their order defines the columns of B. MC2 uses the same joint row penalty with the C2 Huberized squared-hinge loss.
Its lower margin is configured by `rho_classification` (default -1). This is a joint block-penalized
fit, not independent binary fits or a pairwise multiclass margin loss.

## Use

```python
import numpy as np
from classo import Classo, pathlasso
from classo.solve_MC1 import problem_MC1
from classo.mc1_path_engine import solve_path_MC1

# X: samples by features; y: one class label per sample; C: constraints by features
B = Classo((X, C, y), .1, typ="MC1", meth="Path-Alg", w=weights)
B_grid, absolute_lambdas = pathlasso(
    (X, C, y), [1., .5, .1], typ="MC1", w=weights)
pb = problem_MC1((X, C, y), w=weights)
B_knots, relative_knots = solve_path_MC1(pb, lamin=.1)
predicted = pb.classes[np.argmax(X_test @ B_knots[-1], axis=1)]
```

`lam` and the knot outputs are relative to lambda_max unless `true_lam=True`
is passed to the compact functions. Above-threshold absolute lambda returns
zero coefficients. Compact `Classo` requires `meth="Path-Alg"` explicitly;
its legacy default method DR is not an MC1 method. `pathlasso` computes knots
once and interpolates, rather than fitting each requested lambda independently.
The high-level `classo_problem`/CV/stability-selection interface is not yet
multiclass-aware. With `intercept=True`, the first coefficient row contains
unpenalized class biases; all remaining rows are feature coefficients. At the
initial knot the biases are `2*class_frequency-1`, with zero feature coefficients.
`n_active` limits selected feature rows and excludes the biases. It stops before
an outgoing segment would exceed the budget, without splitting simultaneous
entries. Compact pathlasso returns a shorter path including the stop knot, with
its actual lambda values, rather than copying the stopped fit to smaller lambdas.

## Event-driven algorithm

`solve_MC1.py` follows the problem_R1, Classo_R1, pathlasso_R1 naming pattern.
`mc1_path_engine.py` contains the traversal; `path_alg_MC1.py` retains the local
helpers. Each step:

1. Select working rows: current nonzero rows and inactive rows whose weighted
   dual L1 bounds are tight under the current certificate.
2. Solve the directional convex problem on the critical cone, including weak
   maximum faces and one-sided squared-hinge curvature at sample boundaries.
   Positive certificate mass fixes strong faces. Equality nullspace reduction
   removes redundant tangent rows. First try linear solves for the quadratic
   sample regions; accept only a cone-feasible point with zero reduced gradient.
   Otherwise use the constrained numerical QP.
3. Identify outgoing faces and sample regions, then refine the coefficient
   direction through a reduced KKT solve on the selected active rows.
4. Compute primal event caps analytically: sample margins reaching one,
   coordinates meeting row maxima, row maxima reaching zero, and the endpoint.
5. Try to certify the primal cap using a small least-squares multiplier solve.
   Check every inactive L1 bound and every active signed subgradient condition.
   If valid, convex interpolation of the starting and endpoint certificates
   proves feasibility along the whole segment. Otherwise find the largest
   certified step up to that cap by a sparse LP. The LP enforces
   stationarity, nonnegative signed face masses summing to lambda*w on nonzero
   rows, and weighted dual L1 bounds on inactive rows. Certificate failure ends
   the segment at a dual event (including entries and face releases).
6. Move along the affine segment, check endpoint stationarity and equality
   feasibility, record the knot, and select the next outgoing state.

The LP optimizes over all feasible certificates on the proposed primal segment;
this avoids spurious events caused by an arbitrary choice of nonunique dual
multipliers. It is our extension, not an implementation of the canonical
certificate rule in the CLASSO paper. Simultaneous primal events are retained;
the next directional problem handles their outgoing choice jointly.

`pb.path_info` records completion, event labels, working-row counts, and endpoint
residuals, plus direction_linear/direction_qp and
certificate_linear/certificate_lp counters. Exceeding max_events, an unresolved zero step, or a nonunique reduced
KKT direction raises an error. No implicit ridge or lambda-grid fallback is used.
Dense QP/KKT work depends on the working rows times class count. Sparse LPs scan
all p rows; assembly is vectorized and sparse. This has not yet been benchmarked
at p=10,000. No large-p performance claim is made.

## Binary baseline and numerical references

For two classes, B=(-beta,beta) gives

    objective_MC1(B, 2*lambda) = 2*objective_C1(beta, lambda).

Thus absolute lambda_MC1 must be twice lambda_C1. Independent epigraph solves
validate MC1 at multiple interior points of every tested affine segment, with
and without constraints. C1 now uses the unregularized Hessian and checks KKT
conditions at every returned knot. Its intercept margin derivative uses the
current quadratic sample mean, consistent with the intercept update. The old
full-sample mean caused tracked margin drift; the hidden direction ridge also
perturbed sample-event choices. Neither is used in the corrected C1 path.
Nonunique equality multipliers are checked with a residual-minimizing LP when
an active-set least-squares candidate fails. Singular tied directions use an
explicit minimum-norm policy, and the resulting knots must still pass the
certificate.
The independent epigraph optimizer remains the accuracy reference.

On the binary 88-soils path with a maximum of 30 features and target relative
lambda .001, corrected C1 and MC1 agree to 7.14e-14 in coefficients at 150 common
absolute lambda values. Supports and predictions are identical. Both have 22
features at the final common lambda. C1 has one extra initial zero-coefficient
segment because its lambda_max convention differs from MC1's constrained dual
threshold. Representative median runtimes including checks are .102 s (C1)
and 1.59 s (MC1); these are local timings, not a general scaling claim.

`examples/iris_MC1_path.py` reproduces small binary and three-class Iris runs
with a fixed stratified 80/20 split. Standardization uses training samples only.
The example reports path knots, runtime, endpoint residuals, reference objective
differences, C1 differences, and train/test accuracy at three prespecified
lambda values. It does not tune lambda on test data or claim a general accuracy
advantage. scikit-learn is an optional example dependency.

## trac

`prepare_trac_MC1((X,C,y), A, w)` accepts an already log/CLR-transformed leaf
design and a binary descendant-membership matrix A. With n_j descendants,
M=A/n_j maps node coefficients to leaves; the node problem uses X M, C M, and
w/n_j. C=None in this helper selects the leaf sum-to-zero log-contrast.
This matches the numerical aggregation and weight scaling in the inspected
local q2-classo add_taxa source. The helper does not infer a tree or process raw
counts. The four-leaf benchmark checks normalized versus unnormalized tree
parameterizations, prediction preservation, and log-contrast constraints.
Node coefficients may be nonunique; scores and objectives are the comparison.

## References and scope

The parent development workspace stores the supplied PDFs in `assets`:

- Schmelzer and Hastie, arXiv:2609.25704: bordered affine segments and primal/dual
  event tests. https://arxiv.org/abs/2609.25704
- Hobson, Butcher and Lentati, arXiv:2610.17024: directional face/release decisions
  and why finite primal paths do not alone guarantee finite certificate
  traversal. https://arxiv.org/abs/2610.17024
- Malioutov, Cetin and Willsky, *Homotopy Continuation for Sparse Signal
  Representation*, supplied PDF, Section 3: fixed-support affine continuation
  and scalar coefficient-zero/dual-boundary events. Added during implementation
  review; not used for the initial scaffold.

MC1 has class-specific piecewise-quadratic Hessians and zero-curvature epigraph
variables. The cited positive-definiteness/uniqueness assumptions do not transfer
automatically. This is an event-driven floating-point implementation of the
piecewise-affine construction, not a proof of finite termination or support for
all rank-deficient and degenerate inputs. Current tests cover a defined small
suite; arbitrary trees and high-dimensional paths need broader validation.

## Practical compositional example

`examples/soils_MC1_example.py` adapts the original 88-soils MATLAB log-abundance
matrix (88 samples, 116 OTUs). It uses ten pH-quantile classes, class intercepts,
and a shared sum-to-zero constraint. R1 predicts continuous pH on the same
centered log-abundance design. Feature supports are compared at budgets of
2, 5, 10, 15 and 20 using the first path segment attaining each size.
OTU/sample mappings are recovered from and verified against the original BIOM
counts; the MATLAB taxonomy table itself is opaque to SciPy.
This is an exploratory full-data support comparison, not predictive validation.
The example needs biom-format and pandas; neither is added as a core solver
dependency. Run it with `--output /absolute/path/to/results`.

### Numerical performance update

Certificate constraints are assembled directly as sparse COO/CSC matrices, with
constant stationarity blocks cached for each path. Fixed-face directions use
SVD to eliminate homogeneous equalities and solve the reduced positive-definite
system; singular directions are still rejected. The outgoing-direction step
also reuses a single SVD for its equality basis and nullspace. These changes
retain the directional QP, dual-certificate LP and endpoint residual checks.

On the 88-soils, 10-class example with support capped at 20, serial local runs
fell from 54.8 seconds to 9.2 seconds. All 130 knots were retained; coefficients
at 30 common lambda values differed by at most 2.85e-15 and the maximum endpoint
residual was 3.74e-13. These are illustrative single-run timings, not a general
scaling guarantee. The certificate LP is now the largest measured cost.

### Checked linear fast paths

The default engine now tries a cone-feasible stationary linear direction before
SLSQP, and a small multiplier solve plus full subgradient checks before the
segment LP. A failed trial requests the existing general solver. Diagnostic
`pb._mc1_fast=False` permits comparisons against that general traversal.

Three-repeat soil timing medians in the same run: binary 2.326 to 0.870 seconds;
ten classes 12.264 to 7.744 seconds. The paths retained 90 and 130 knots,
respectively, with zero coefficient difference at 100 common lambda values.
Binary used 85 linear direction selections and 5 QPs, and 65 linear certificates
and 24 LPs; ten-class used 121/9 and 86/43. This is an initial fast-path
implementation, not a complete incremental primal-dual pivot algorithm.

## Development scope

This branch adds MC1/MC2 without a ridge argument. Scalar R1–R4/C1–C2 solvers
come from the current maintenance baseline. Fixed ridge lives on `ridge`;
combining it with the multiclass interface is a separate integration step.
MC2 is selected with `typ="MC2"` and `meth="Path-Alg"`, with optional metadata,
weights, intercept and lower margin `rho_classification < 1`.
