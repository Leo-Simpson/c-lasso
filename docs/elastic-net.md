# Fixed-ridge elastic net

`Classo` and `pathlasso` accept `ridge=0.0`. A positive value adds
`ridge / 2 * sum(beta**2)` to fixed-scale objectives, in the original
coefficient coordinates. For R3/R4 this term shares the concomitant scale:
`ridge / (2*sigma) * sum(beta**2)`. For MC1/MC2 it is the squared Frobenius norm.
Feature weights `w` multiply only the L1 or row-Linf penalty, not the ridge.
The intercept and explicitly unpenalized multiclass covariates receive neither
penalty. Equality constraints are unchanged. Losses remain sums, not averages;
the numerical scale of ridge therefore depends on training sample size.

```python
from classo import Classo, pathlasso

beta = Classo((X, C, y), 0.2, typ="R1", ridge=2.0, intercept=True)
betas, lambdas = pathlasso(
    (X, C, y), lambdas=[1., .5, .2, .1],
    typ="R1", ridge=2.0, intercept=True,
)
```

Lambda defaults to a fraction of the constrained lambda_max. `true_lam=True`
accepts absolute values. Ridge stays FIXED throughout a path. At a positive-scale, zero-feature
starting point its gradient is zero, so it does not change lambda_max. For R4
at scale zero, the perspective ridge changes the joint subgradient set; its
threshold is computed from that set rather than using the positive-scale rule.
`ridge=0` preserves the existing R1–R4/C1–C2/MC1 solver dispatch.

## Which paths are affine?

| Formulation | Positive-ridge computation |
| --- | --- |
| R1: squared residual | Event-driven, piecewise affine |
| R2: Huber residual | Event-driven, piecewise affine |
| C1: squared hinge | Event-driven, piecewise affine |
| C2: Huberized squared hinge | Event-driven, piecewise affine |
| MC1: joint row-Linf squared hinge | Event-driven, piecewise affine |
| MC2: joint row-Linf Huberized squared hinge | Event-driven, piecewise affine |
| R3: concomitant squared residual | One quadratic homotopy with exact scalar reparameterization |
| R4: concomitant Huber residual | Mean-shift reduction using the same homotopy solver |

MC2 uses signed one-vs-rest targets inside ONE joint optimization with a shared
row-Linf penalty. Its lower margin `rho_classification < 1` has C2's convention.
It is not Crammer–Singer classification. Class columns follow sorted unique
labels; numeric and string labels are accepted, as for MC1.

For fixed-scale losses, the ridge diagonal enters the working-set directional
QP, the reduced fixed-face Hessian, and the full gradient/certificate derivative.
The next sample, coefficient-face or dual event is still computed analytically.
Direction solves scale with the working set rather than a dense pK Hessian.
Certificate LPs still inspect all features. Returned points between exact knots
are affine interpolations; unresolved degeneracies still raise explicitly.
A positive ridge improves conditioning but is not a general cure for every
path-pivot or certificate failure.

R3 solves the concomitant-ridge objective

```
(||X beta + b - y||² + ridge/2 * ||beta||²) / sigma
+ e*sigma + lambda * sum(w*abs(beta)),
```

with default `e=n/2`. Write its quadratic numerator as Q(beta,b). The ordinary
quadratic lasso path solves Q + t*sum(w*abs(beta)). Its solutions map to R3 by
`sigma=sqrt(Q/e)` and `lambda=t/sigma`. Within each quadratic-path segment,
the requested lambda is located by a scalar norm equation, not linear
interpolation in concomitant lambda. The solver stops at the requested endpoint
or sparse budget, rather than tracing an unnecessary dense tail. Ridge enters
active Hessians directly; no p-by-p identity observations are constructed.

R4 solves `sigma*sum(huber_rho(residual/sigma)) + ridge/(2*sigma)*||beta||²
+ e*sigma + lambda*sum(w*abs(beta))`, with default `e=n`. The mean-shift identity
rewrites its loss as

```
(||residual-u||² + ridge/2*||beta||²)/sigma
+ e*sigma + 2*rho*||u||1.
```

For positive lambda it uses the same R3 path machinery with mean-shift weights
`2*rho/lambda`. Those auxiliary columns are sparse, have no ridge or tree
constraint, and do not count toward the microbial/feature budget. This reduction
depends on the requested lambda, as in the original R4 implementation: R4 is
not advertised as a single piecewise-affine lambda path. At lambda=0 a scalar
scale root uses ordinary constrained ridge-Huber solves. The closed sigma=0
case is handled without adding a scale floor.

No CVXPY, Clarabel or other conic package is needed for any production ridge
solver. Independent tests may use CVXPY as a numerical reference. Positive
ridge dispatches to this path machinery irrespective of a legacy `meth`
selection; the high-level interface reports `Path-Alg`.

The original constraint is always `C beta=0`. Intercepts and mean-shift columns
get zero constraint columns. Quadratic ridge directions remain in its null
space, so compositional and trac constraints carry through unchanged. In R4 the
ridge contribution stays quadratic; it is never passed through the Huber loss.

## Model selection

For the existing scalar high-level interface:

```python
problem.formulation.ridge = 2.0
problem.solve()
```

PATH, CV, fixed-lambda fitting and stability selection propagate the same fixed
ridge. The existing high-level interface is still scalar; joint MC1/MC2 use the
compact functions or their problem classes and externally grouped CV workflows.
Ridge-aware ALO is not implemented and raises instead of applying the old
ridge-free formula. R3/R4 support an intercept, but not an additional unpenalized
covariate matrix. Sparse-budget stops return only achieved path points.

To obtain every knot rather than sampled path values:

```python
from classo import problem_elastic_net, solve_path_elastic_net
pb = problem_elastic_net((X, C, y), typ="MC1", ridge=2., intercept=True)
betas, relative_knots = solve_path_elastic_net(pb, lamin=.1, n_active=30)
print(pb.path_info["stop_reason"])
```

In stability selection, after a sparse budget stop, the support-frequency plot
holds the stopped support fixed; coefficients are never extrapolated.

To tune ridge, compute separate lambda paths for a small set of fixed ridge
values, using identical validation folds. This does not implement the alternative
`lambda * (alpha*P(beta) + (1-alpha)/2*||beta||²)` parameterization: changing
lambda in that expression also changes curvature and loses affine segments.

## Validation

`classo/tests/test_elastic_net.py` compares all eight formulations with independent
convex references; checks segment interiors, custom scale parameters, closed
zero-scale cases, redundant constraints, collinear features, free covariates,
weighted penalties, absolute/relative lambdas, model-selection propagation and
zero-ridge compatibility. Binary MC1/MC2 equal C1/C2 at twice the absolute lambda
and the SAME ridge (the two class columns also double the squared norm).
