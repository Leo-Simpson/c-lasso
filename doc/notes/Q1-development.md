# Q1: constrained smoothed quantile regression

Branch `quantile` (renamed from `codex/q1-smoothed-quantile`), based on bio-datascience/c-lasso master
`d22da8d5ea54`. Q1 is fixed-smoothing constrained quantile regression with an
L1 penalty, analogous to R1/R2 and MC1/MC2. Nonconvex penalties are excluded.

## Public API

```python
from classo import Classo, pathlasso, Q1_path

beta = Classo((X, C, y), .1, typ='Q1', meth='DR', tau=.2,
              quantile_k=.8, intercept=True, true_lam=True)
B, absolute_lambdas = pathlasso(
    (X, C, y), lambdas=[1., .3, .1], typ='Q1', meth='Path-Alg',
    tau=.8, quantile_k=.8, intercept=True, n_active=8)
```

Compact functions use relative lambda unless `true_lam=True`.
`get_lambdamax=True` returns `(maximum, beta)` for `Classo`.
Positive penalty weights `w` are supported; returned coefficients are in
original units. Paths return descending absolute lambdas. The default grid has
100 points. As with the existing compact API, Classo defaults to DR and
pathlasso defaults to Path-Alg. P-PDS and PF-PDS are also supported. All four are actual Q1 algorithms. `SLSQP` is
retained as a small-problem validation option.

`Q1_path(X,y,lambdas,tau=.5,k=1.,C=None,intercept=True,divisor=None,
method='events',n_active=0)` exposes diagnostics with absolute lambdas. It
returns coefficients, KKT errors, event records, iterations, lambda_max,
evaluated_lambdas, and stop_reason. `method='DR'` selects DR. For this low-level
API only, C=None defaults to zero-sum; the compact API retains the repository's
constraint convention. Pass `np.ones((1,p))` explicitly for log contrasts.

## Objective and matching to R2

Q1 minimizes sum rho_tau,k(y-a-X beta)/D + lambda ||w*beta||_1, subject to
C beta=0. The intercept is unpenalized. D defaults to observations; use
`quantile_divisor` (compact API), or `divisor` (diagnostic API), for the paper's
subject normalization. This is a working-independence sum.

The loss derivative is clip(r/k,tau-1,tau), with thresholds
L=-(1-tau)k and U=tau*k. At tau=.5 the exact upstream R2 matching is
`rho_R2=k/2`, `lambda_R2=2*k*D*lambda_Q1`, with absolute lambdas.
The [initial research report](Q1-initial-research.md) documents the verified
paper and derivation. It is historical: its file paths, solver failure counts,
and timings describe the earlier research snapshot, superseded here.

## Path-following and sparse stopping

Start at the exact intercept-only optimum and constrained lambda_max, without
calling the dense warm optimizer. For a zero-sum row the maximum is half the
range of the slope gradient. General constraints use a small dual LP. A request
at or above lambda_max returns zero slopes without running either solver.

On a fixed residual cell and coefficient face, H=Z_I'Z_I/(kD), and the
descending-lambda direction solves H_AA d_A+C_A'nu_dot=sign(beta_A), C_A d_A=0,
with zero intercept sign. Small-support SVD solves avoid a p-by-p Hessian.
Active stationarity often uniquely determines the equality multiplier and its
derivative, so inactive entry times are computed directly. Remaining dual
freedom uses the shared upstream LP certificate. Events are coefficient
entry/exit and the two residual thresholds. The fast path excludes residual zero from its event list because the loss
has unchanged curvature there. The shared one-sided fallback can still
introduce extra zero crossings in ambiguous cases.

Ambiguous outgoing faces use the shared critical-cone solver with asymmetric
one-sided caps. Flat optimal faces use a Q1-specific LP: first descend lambda
with fixed coefficients when possible, otherwise move on the current exact
optimal face. LP feasibility tolerances start at 1e-10; a degenerate face may be
retried at 1e-9. Small-support KKT polishing removes simplex artifacts, and
accepted candidates must still pass the actual Q1 certificate. Unresolved cases raise;
there is no hidden ridge or substitution of DR for a failed event path.

`n_active` counts nonzero original slopes, excluding the intercept and any
loss-splitting variables. Path-Alg stops at the first knot reaching that count
and returns its actual lambda, which can differ from the requested grid. If a
tied outgoing entry would exceed the budget, it stops on the incoming side.
A flat-face pivot exceeding the budget raises explicitly. The splitting solvers (DR, P-PDS, PF-PDS) have no continuous
support-budget stopping and reject n_active. A small support budget cannot
bound the number of residual-threshold events; timings report both counts.

`store_segments=True` optionally stores affine segment starts and directions
for inspection. It is off by default to avoid storing p coefficients per knot.
Each event endpoint is certified; additional independent QP checks sample
segment interiors in `benchmark/validate_Q1.py`.

## Douglas–Rachford: reuse of R1 and R2 architecture

R2 already reduces Huber loss to a quadratic problem with auxiliary mean shifts
and calls `Classo_R1`. Q1 uses the same architecture. The exact Moreau identity
is

    rho_tau,k(r) = min_o (r-o)^2/(2k) + rho_tau(o).

Multiplying Q1 by 2kD gives the augmented constrained quadratic objective

    ||Zb+o-y||^2 + 2kD lambda ||b_slopes||_1
        + k ||o||_1 + 2k(tau-.5) sum(o),  Cb=0.

`_q1_dr.py` prepares this problem and calls the existing `Classo_R1` DR loop.
The only asymmetry change in that loop is shifting the auxiliary L1 proximal
input by gamma times the linear coefficient. The original coefficient prox,
constraint projection, relaxed updates, and quadratic resolvent are reused.
Legacy R1/R2 calls retain their default dense implementation and stopping rule.

For Q1, u=S*b scales slopes by max(1,||X||_F/sqrt(n)) and the intercept
by sqrt(n). The augmented design is [Z/S,I], the original penalty thresholds
are divided by the corresponding scales, and C u=0
is equivalent to C b=0. This is an invertible change of variables. The equality
projector is applied as v-C'Cv using orthonormal independent rows, without a
dense p-by-p matrix. The existing `QQ` helper has an optional matrix-free
Woodbury form, using a cached n-dimensional Cholesky factor instead of a dense
p-dimensional inverse. An unused dense `Q1 @ A` product was removed from the
shared DR setup. Quadratic resolvents and the existing three-vector DR state
are reused along the lambda grid. The R1 relaxation mu=1.95 is retained.

Q1 adds an actual objective KKT check to the existing iterate-change stopping
rule, with tolerance min(requested tolerance,1e-8), using physical coefficients
and residual y-Zb. Iteration failure raises explicitly. Tests check both reuse
of the R1 loop and equality of matrix-free and dense quadratic resolvents.

## Reproduction and scope

Run `python -m pytest classo/tests -q` with this checkout first on PYTHONPATH.
`benchmark/validate_Q1.py` uses optional CVXPY/Clarabel to compare eight designs,
three tau levels, 14 requested lambdas, three solvers, and sampled affine
interiors. `benchmark/benchmark_Q1_sparse.py` benchmarks 500, 2,000, and 5,000
predictors, tau=.2,.5,.8, at .9 lambda_max and at an eight-slope budget; three
repetitions (one for the 5,000-predictor stress cases) compare path and DR on
identical final lambdas. Timings include
initialization, projection construction, and certificates, and request one
BLAS thread. See `benchmark/q1-results/` for actual results.

The high-level classo_problem, CV, stability selection, and plotting do not yet
recognize Q1. No general finite-termination result is claimed. Broad performance
claims require more data than the small synthetic suite here; dense paths can
still accumulate many residual events.

## Current independent validation

Eight seeds at tau=.2,.5,.8 produce 24 paths per solver, with 14 requested
lambdas each. All 72 paths pass, with no failures. The event solver has maximum
objective difference 4.082e-11 against Clarabel, maximum requested-endpoint KKT
1.428e-14, and maximum sampled segment-interior objective difference 9.398e-11.
Reused R1 DR has maximum objective difference 2.080e-8 and maximum KKT 9.997e-9.
The small SLSQP comparator has maximum objective difference 4.082e-11.
The median-quantile path agrees with correctly scaled upstream R2 coefficients
to 8.882e-16. These are numerical tests, not a universal degeneracy guarantee.

Full regression suite: **336 passed in 218.77 seconds**, including 26 Q1 tests.

After the final Q1-only certificate and allocation optimizations, all 26 Q1
tests were rerun and passed in 4.88 seconds; shared R1 code was unchanged.

## Final sparse benchmark

All 18 benchmark cases passed. Against reused R1 DR, event paths are 8.90–84.14x
faster at .9 lambda_max (4–21 event steps, two slopes). At the eight-slope budget,
the speedup is 1.04–12.68x (33–61 event steps). A small support size does not
imply a uniformly small residual-event count. See
[full timing table](../../benchmark/q1-results/README.md) for all cases and
single-run versus three-repeat distinctions. These final timings supersede
all intermediate measurements from earlier DR adapters.

## PDS completion: P-PDS and PF-PDS

Both public compact functions now accept `meth='P-PDS'` and `meth='PF-PDS'`
with `typ='Q1'`; `Q1_path` accepts the same method strings. The existing
`Classo_R1` PDS iteration bodies are reused through the same Moreau mean-shift
reduction as DR. No alternative objective or extra penalty is introduced.
The auxiliary asymmetric linear term is handled by shifting the existing
soft-thresholding input by gamma times its linear coefficient.

For PDS, the augmented A'Av is applied as A'(Av), without storing A'A. The
largest eigenvalue of AA' supplies ||A||^2. Independent constraint rows are
orthonormal, so the adapter chooses

    gamma = .45/(2 ||A||^2), tau_dual = .45 if C has rows, otherwise 0.

Thus gamma times the quadratic-gradient Lipschitz constant is .45 and
tau_dual ||C||^2 is at most .45, leaving a strict margin in the shared PDS
step-size condition. P-PDS uses the existing projection as a matrix-free
operator. PF-PDS performs the existing projection-free updates.

Each scheme preserves its own existing warm state: (xbar,x,v) for P-PDS and
(x,v) for PF-PDS. The state, operator, and step sizes are reused across a
requested lambda grid. Iteration-change stopping must also pass the physical
Q1 KKT certificate with tolerance min(requested tolerance,1e-8). The diagnostic
API's `pds_max_iter` defaults to 500000 iterations per requested point; exceeding
it raises an explicit nonconvergence error. Support-budget stopping remains a
Path-Alg feature. DR and path defaults are unchanged.

Examples:

```python
B, absolute_lambdas = pathlasso(
    (X, C, y), lambdas=[1., .3, .1], typ='Q1', meth='P-PDS',
    tau=.2, quantile_k=.8, intercept=True)
beta = Classo((X, C, y), .1, typ='Q1', meth='PF-PDS',
              tau=.8, quantile_k=.8, intercept=True)
```

Independent validation can now select methods and its result filename:

    python benchmark/validate_Q1.py --methods P-PDS PF-PDS --output pds-validation.json

The original three-method validation and sparse benchmark files remain intact;
PDS results and test logs are stored separately in benchmark/q1-results/.

### PDS validation results

Both methods passed all 24 independent paths (tau=.2,.5,.8, eight data seeds),
for 48 paths and 672 requested fits. P-PDS maximum absolute objective difference
from Clarabel is 5.045e-11; PF-PDS is 5.287e-11. Maximum endpoint KKT errors are
9.9995e-9 and 9.99993e-9 respectively. The tests include redundant constraints,
dependent columns, rounded responses, both intercept choices, p>n, and a
subject divisor different from the observation count.

At the initial 100000 per-point limit, 46 paths completed. Both methods hit
that limit on seed 7, tau=.5. Retrying these two paths at the standard R1
500000-iteration budget passed at the unchanged tolerance, with total path
iteration counts 273380 (P-PDS) and 273450 (PF-PDS). The final PDS default is
500000 per point. The combined result records this provenance rather than
hiding the initial limit cases. These methods are completeness backends;
no large-design speed advantage is claimed over the event solver.

Results: `benchmark/q1-results/pds-validation.json`; initial limits:
`pds-validation-100k.*`; independent retry: `pds-retry-validation.*`.
The full 48-case experiment is reproducible with the command above. The retry
can be reproduced with `--seeds 7 --taus .5` added to that command.

The added tests check general constraints, weights and compact interfaces,
lambda zero, matrix-free R1 reuse, warm states, empty constraints, explicit
nonconvergence, and rejection of support budgets for splitting schemes.

PDS completion full regression suite: **352 passed in 408.82 seconds**,
including 42 Q1 tests (16 added for the PDS schemes). The run overlapped
independent validation, so its elapsed time is not a solver benchmark.

## Controlled four-method speed comparison

The new `benchmark/benchmark_Q1_methods.py` times all four production backends
on matched single fits and complete 14-point paths. Eleven cases cover
40x9, 80x500, and 100x2000 designs. It verifies one actual OpenMP/OpenBLAS
thread through threadpoolctl; the old OPENBLAS_NUM_THREADS-only request did
not reliably control this OpenMP build. Earlier timing records are historical;
use [the controlled comparison](../../benchmark/q1-results/METHOD-SPEED.md).

In this experiment P-PDS takes 3.67–11.95x the DR runtime; PF-PDS takes
5.41–15.89x. P-PDS is consistently faster than PF-PDS. Event paths lead at
large lambda; DR can match the event solver at some eight-slope endpoints.
All timed fits pass the same KKT threshold and paired objective check.
These are case-specific observations, not a universal performance guarantee.
