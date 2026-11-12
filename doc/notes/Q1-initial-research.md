# Archived initial experiment

This report records the earlier research snapshot. Current Q1 code, algorithms,
validation, and scope are documented in Q1-development.md. Nonconvex penalties
are excluded from this project.

# Constrained smoothed quantile research

This task implements and validates a separate research API, `classo.quantile.quantile_path`, in a frozen upstream snapshot. It does not integrate quantile loss into `classo_problem`, cross-validation, stability selection, or the existing compact `Classo` interface. No previous pilot results are used. MCP/SCAD are excluded.

## Sources and provenance

Core source: https://github.com/bio-datascience/c-lasso at **d22da8d5ea54** (downloaded master tarball, 2026-11-12; tar root confirms d22da8d). `results/upstream-commits.json` records 15 recent commits. The old local `../c-lasso` clone points to Leo-Simpson and was not used or edited.

Paper verified from arXiv v1, submitted November 5, 2026: Changwei Du, Linhui Li, Huijuan Ma, Qi Zheng, *A Unified Coordinate Descent Framework for Penalized Longitudinal Log-Contrast Quantile Regression*, https://arxiv.org/html/2611.07379v1 . Equations (1)–(3) define residual sign, longitudinal normalization, and the smoothing. The implementation adopts the convex L1 objective; it does not reproduce the paper's entire coordinate MM/augmented Lagrangian procedure or longitudinal statistical inference.

Recent upstream changes inspected: 61f84721 corrects R1/R2 curvature, actual residual tracking, and degenerate events; e7553fda improves weak-curvature SVD solves and dual-event elimination; e5ba6cc6 tries curved segments before flat-face LPs; 934be97b repairs R3/R4; ca9cd065 fixes response-scale-sensitive exits; 4745fe9d/61c8edad repair dependent C1 faces and exact exiting zeros; 1c162e49 unifies constraints and preserves boundary objectives. This work uses `_classification_path.direction/segment`, not the old Huber updater. The symmetric R2 flat-face adapter is not directly reusable for asymmetric flat cells.

## Objective and exact normalization

Let r=y-a-Xb, 0<tau<1, k>0. Minimize

    sum_i rho(r_i)/D + lambda * ||b||_1, subject to C b=0.

The intercept a is optional and unpenalized, with a zero constraint column. C defaults to a row of ones. The default D is the observation count. For the paper's repeated observations, D must instead be the number of subjects, explicitly passed via `divisor`; rows are summed with working independence. No subject correlation adjustment is performed.

The loss is quadratic r²/(2k) between L=-(1-tau)k and U=tau*k, with linear tails joined continuously. Its derivative is clip(r/k,tau-1,tau), and curvature is 1/k in the interior. The precise definition is implemented in `loss`.

At tau=.5, the conventional Huber H_delta(r), quadratic r²/2 inside delta, satisfies rho=H_(k/2)/k. Upstream R2 uses h_delta=2 H_delta, sums observations, and adds lambda_R2 ||b||_1. Therefore

    rho_.5,k = h_(k/2)/(2k)
    lambda_R2 = 2*k*D*lambda_quantile; rho_R2 = k/2.

This matching requires absolute lambda (`true_lam=True`). Relative lambda/lambda_max values must not be substituted without computing the corresponding maximum. Multiplying the entire quantile objective by 2kD gives the R2 objective exactly, without an additive constant.

## Warm-started lambda path

`method='warm'` solves a decreasing requested lambda grid using the previous coefficient vector. SLSQP minimizes the differentiable convex loss with explicit epigraph variables t>=b and t>=-b and exact homogeneous linear constraints. No ridge is added. Equality rows are reduced by SVD to remove redundant constraints. An intercept-only score root initializes the first fit. Each endpoint must pass a separate LP-based KKT certificate, independent of the optimizer's success flag. The certificate minimizes the infinity stationarity violation over equality multipliers, treating slopes smaller than 1e-7 as numerically zero; default acceptance is 2e-6 absolute. It also checks primal equality feasibility. This is a numerical certificate, not a rigorous interval bound or duality-gap certificate.

KKT: g=-Z'psi(r)/D, Ctilde=[0,C], and

    g_intercept=0,
    g_slopes+C'nu+lambda*s=0,
    s_j=sign(b_j) if b_j !=0; |s_j|<=1 otherwise,
    C b=0.

Lambda zero is accepted. Uniqueness is not assumed; fitted values can differ on flat minimizer sets. No lambda_max API or automatic grid is implemented. A zero-slope threshold for a single zero-sum constraint, after optimizing the intercept, is (max(g)-min(g))/2. More general constraints require minimizing ||g+C'nu||_infinity over nu.

## Event homotopy derivation and implementation

On fixed coefficient signs and residual cells, set Z=[1,X] if fitting an intercept. For interior observations I, the Hessian is H=Z_I' Z_I/(kD). For decreasing lambda parameter t=lambda0-lambda, the affine direction obeys

    H_AA d_A + C_A' nu_dot = s_A,
    C_A d_A = 0,

with zero intercept sign. Inactive adjusted gradients remain in [-lambda,lambda]. Candidate events are coefficient exit b_j+t*d_j=0, inactive adjusted-gradient boundary entry, and r_i(t)=L or U. Since r_dot=-Z d, threshold times are (r_i-threshold)/(Z_i d), retaining positive times. Tied events require outgoing critical-cone selection, not independent coordinate toggles. Singular H can admit multiple directions or an optimal-face jump, so a naive inverse is inadequate.

The loss is exactly the sum of two capped one-sided quadratics, each with coefficient 1/(2kD): r_+ capped at U and (-r)_+ capped at -L. The implementation scales paired designs by sqrt(2kD), using upstream hinge primitives with observation-specific cap widths. Only one upstream primitive line changes: threshold values are broadcast and indexed when computing residual crossing roots. The paired representation gives the exact loss gradient and Hessian without a linear-tilt approximation; unit tests check it for all three tau values.

`method='events'` initializes its first requested point with the convex warm solver, then traces affine segments, using upstream SVD/critical-cone selection and dual interval certificates. It records segment endpoint lambdas; requested points may also truncate segments. It snaps analytically exiting coefficients and checks every requested endpoint. It raises on unresolved singular/flat outgoing faces; it does not silently switch methods or return an uncertified result. It is a numerical event solver, not a symbolic exact-arithmetic solver. Seven of 24 tested paths fail explicitly, so this is exploratory, not a generally robust completed homotopy implementation. The warm path is the usable initial implementation.

## Reproduction and actual results

From the project root:

    OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR="$PWD/.cache/matplotlib" .venv/bin/python quantile-research/validate.py
    OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR="$PWD/.cache/matplotlib" .venv/bin/python -m pytest quantile-research/test_quantile.py quantile-research/upstream/classo/tests -q

Validation adds the existing `.comparison-deps` directory only to load CVXPY/Clarabel reference packages. Production uses NumPy/SciPy. This local environment is not a portable dependency lock. Versions and all per-case results/failures are in `results/validation.json`; tests are recorded in `results/tests.txt`.

Eight deterministic data seeds, tau=.2,.5,.8, 14 absolute lambdas from .5 to .01, k=.8: six designs have n=40,p=9 and two have n=24,p=30. Both intercept choices, duplicated columns, redundant constraints, rounded outcomes, and a subject divisor distinct from row count are covered. The independent QP uses the Moreau envelope: r=v+u_plus-u_minus, with nonnegative u_plus/u_minus and objective ||v||²/(2kD)+(tau sum u_plus+(1-tau) sum u_minus)/D+lambda||b||_1. Clarabel tolerances are 1e-10. This modeling does not call the implementation's loss or event functions.

- Warm: 24/24 paths, 336 requested fits; maximum absolute objective difference 4.082e-11; maximum endpoint KKT 2.359e-7.
- Events: 17/24 complete paths, 238 validated requested fits; maximum objective difference 4.082e-11; maximum endpoint KKT 3.331e-16. Seven explicit outgoing-cone singular failures are preserved in the JSON. Failed paths are excluded from event timing summaries and are not counted as successes.
- A separate n=50,p=10, 20-lambda tau=.5 intercept comparison against current upstream R2 gives maximum coefficient difference 1.554e-15 with the above scaling. R2 takes .1894 seconds in that run.
- Warm maximum prediction difference against the QP is .1342, despite objective agreement: piecewise-linear tails allow nonunique minimizers. Successful event paths have maximum prediction difference 1.522e-5. Objective and KKT, rather than coefficient identity, are the primary checks.

Timings include path solving and certificates; reference times include QP construction and repeated solves. Single invocation per case, single BLAS thread requested, Python 3.12.13, NumPy 2.4.2, SciPy 1.17.1; full installed versions are in JSON. These are small synthetic timings, not statistically stable speed claims. Median times over eight cases at each tau:

| tau | warm path (seconds) | independent QP path (seconds) |
| --- | ---: | ---: |
| .2 | .2074 | .0694 |
| .5 | .2570 | .0829 |
| .8 | .2529 | .0804 |

Across all successful paths, warm median .2490 seconds; events median .4143 seconds on the 17 completed cases. Those medians describe different case sets and must not be treated as a paired speedup. There is no evidence here of a performance advantage over the independent solver.

## Changed files and next questions

Added `upstream/classo/quantile.py`, `validate.py`, `test_quantile.py`, and this report; changed one line in `upstream/classo/_classification_path.py` to support vector thresholds. The rest of `upstream` is the untouched downloaded core snapshot. `implementation.patch` isolates these solver changes against that snapshot for review.

Remaining work: asymmetric optimal-face LP pivots for zero-curvature cells; robust cone direction selection under singular curvature and nonunique minimizers; scale-aware thresholds throughout shared primitives; event provenance and interior-of-segment validation across randomized larger stress cases; optional weighted penalties/unpenalized covariates; high-level integration and automatic lambda grids; repeated paired benchmarks on larger compositional designs. No general finite-termination theorem is claimed. MCP/SCAD are excluded.

Final regression run: **321 passed in 187.60 seconds** (11 new quantile tests plus 310 upstream tests); process exit code 0. The focused quantile run independently reports 11 passed in 3.25 seconds.
