# MC1: joint multiclass squared-hinge path

MC1 extends C1 to signed multiclass targets T (observed class +1, others -1):

    min_B sum_ik max(1 - T_ik (XB)_ik, 0)^2
          + lambda sum_j w_j max_k |B_jk|, subject to CB=0.

The loss is unhalved. All weights must be finite and strictly positive. Classes
are sorted by `numpy.unique`; their order defines the columns of B. MC2 remains
reserved for a future Huberized-hinge extension. This is a joint block-penalized
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
multiclass-aware. Intercepts and n_active early stopping are not implemented.

## Event-driven algorithm

`solve_MC1.py` follows the problem_R1, Classo_R1, pathlasso_R1 naming pattern.
`mc1_path_engine.py` contains the traversal; `path_alg_MC1.py` retains the local
helpers. Each step:

1. Select working rows: current nonzero rows and inactive rows whose weighted
   dual L1 bounds are tight under the current certificate.
2. Solve the directional convex problem on the critical cone, including weak
   maximum faces and one-sided squared-hinge curvature at sample boundaries.
   Positive certificate mass fixes strong faces. Equality nullspace reduction
   removes redundant tangent rows before the numerical QP.
3. Identify outgoing faces and sample regions, then refine the coefficient
   direction through a bordered KKT solve on the selected active rows.
4. Compute primal event caps analytically: sample margins reaching one,
   coordinates meeting row maxima, row maxima reaching zero, and the endpoint.
5. Find the largest certified step up to that cap by a sparse LP. It enforces
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
residuals. Exceeding max_events, an unresolved zero step, or a nonunique reduced
KKT direction raises an error. No implicit ridge or lambda-grid fallback is used.
Dense QP/KKT work depends on the working rows times class count. Sparse LPs scan
all p rows; their construction is currently Python-heavy and not yet benchmarked
at p=10,000. No large-p performance claim is made.

## Binary baseline and numerical references

For two classes, B=(-beta,beta) gives

    objective_MC1(B, 2*lambda) = 2*objective_C1(beta, lambda).

Thus absolute lambda_MC1 must be twice lambda_C1. Independent epigraph solves
validate MC1 at multiple interior points of every tested affine segment, with
and without constraints. C1 coefficients are a useful compatibility baseline,
but the existing implementation includes eps_L2 stabilization in path_alg.py
(default 1e-3) and 1e-4 sample-margin snapping. It therefore need not equal the
new unaugmented MC1 solution at machine precision. The independent optimizer
of the unaugmented objective is the accuracy reference.

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
