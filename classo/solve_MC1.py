from .misc_functions import _constraint_rows
import numpy as np

from .multiclass_path import initialize_path


r"""
Problem : min sum_ik max(1 - T_ik * (X B)_ik, 0)^2
              + lambda sum_j max_k |B_jk|, subject to C B = 0.

T_ik is +1 for the observed class and -1 otherwise. The class scores
are coupled by the row penalty: this is a joint problem, not K binary fits.
The squared loss is unhalved, consistently with C1 and R1.

Classo_MC1 and pathlasso_MC1 use an event-driven path engine, with explicit
failure on unresolved singular/zero-step states. Explicit fixed ridge is
supported; no implicit ridge is added.
"""

tol = 1e-8


def fit_unpenalized_hinge(Z, targets, tol=1e-9):
    """Fit the metadata-only starting point and certify free stationarity.

    Classwise convex squared hinge, with analytic gradient and local quadratic
    refinement. Full-column-rank Z is required; flat loss directions use the
    minimum correction to the numerical optimizer's representative.
    """
    from scipy.optimize import minimize
    coefficients = np.zeros((Z.shape[1], targets.shape[1]))
    for k, target in enumerate(targets.T):
        def objective(gamma):
            r = np.minimum(target*Z.dot(gamma)-1., 0.)
            return r.dot(r), 2*Z.T.dot(target*r)
        initial = np.linalg.lstsq(Z, target, rcond=None)[0]
        fit = minimize(objective, initial, jac=True, method="L-BFGS-B",
                       options=dict(gtol=1e-11, ftol=1e-15, maxiter=3000,
                                    maxls=50))
        gamma = fit.x
        for _ in range(20):
            _, gradient = objective(gamma)
            if np.max(np.abs(gradient), initial=0.) <= tol:
                break
            active = target*Z.dot(gamma) < 1.
            correction = np.linalg.lstsq(Z[active], target[active]-Z[active].dot(gamma), rcond=None)[0]
            old = objective(gamma)[0]
            for step in 2.**(-np.arange(30)):
                candidate = gamma+step*correction
                if objective(candidate)[0] <= old+1e-13:
                    gamma = candidate
                    break
            else:
                break
        if np.max(np.abs(objective(gamma)[1]), initial=0.) > 2e-8:
            raise RuntimeError("Metadata-only squared-hinge initialization failed its stationarity check")
        coefficients[:, k] = gamma
    return coefficients


class problem_MC1:
    """MC1 data and initialization, following the problem_R1 convention.

    data : (X, C, y)
        X has shape (m, d), C has shape (k, d), y contains m class labels.
        C=None means no equality constraints. At least two observed classes
        are required. One-hot matrices are intentionally not accepted here.
    algo : str
        Only 'Path-Alg' is reserved for this incremental implementation.
        Constructing this object does not run a path solver.

    Positive feature weights are supported through w. With intercept=True,
    the returned coefficient matrix starts with an unpenalized class-bias row.
    classes defines the column order of B; targets contains signed indicators.
    init has zero feature coefficients and fitted constant class biases when
    requested. active is empty at lambdamax;
    candidates contains dual-tight rows, not a selected entering face.
    covariates is an optional full-rank unpenalized n-by-q design. Returned
    rows are [intercept (if requested), covariates, penalized features]. Only
    the last p rows enter the block penalty and feature budget. Covariate
    encodings and transformations must be fit on training samples only.
    """

    def __init__(self, data, algo="Path-Alg", w=None, intercept=False, covariates=None, ridge=0.0):
        if algo != "Path-Alg":
            raise ValueError("MC1 currently reserves only the Path-Alg method")
        from .elastic_net import validate_ridge
        self.ridge = validate_ridge(ridge)
        X, C, y = data
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        if X.ndim != 2 or not all(X.shape) or not np.isfinite(X).all():
            raise ValueError("X must be a finite, nonempty matrix")
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError("y must contain one class label per sample")
        if y.dtype.kind not in "biufUS":
            raise ValueError("Use numeric or string class labels")
        if y.dtype.kind in "f" and not np.isfinite(y).all():
            raise ValueError("Class labels must be finite")
        C = np.empty((0, X.shape[1])) if C is None else np.asarray(C, dtype=float)
        if C.ndim != 2 or C.shape[1] != X.shape[1] or not np.isfinite(C).all():
            raise ValueError("C must be finite with one column per feature")
        self.original_constraints = C.copy()
        C = _constraint_rows(C, X.shape[1])

        self.classes, encoded = np.unique(y, return_inverse=True)
        self.n_classes = len(self.classes)
        if self.n_classes < 2:
            raise ValueError("MC1 requires at least two observed classes")
        # Own the data: caller mutations must not invalidate initialization.
        self.matrix = X.copy(), C.copy(), y.copy()
        self.dim = X.shape[0], X.shape[1], C.shape[0]
        self.targets = -np.ones((len(X), self.n_classes))
        self.targets[np.arange(len(X)), encoded] = 1.0
        self.type, self.name = algo, algo + " MC1"
        self.tol, self.regpath = tol, False
        # The directional cone QP supplies a verified null-space reference;
        # dependent tree columns need not define a unique coefficient path.
        self._mc1_allow_tree_nullspace = True
        self.weights = np.ones(X.shape[1]) if w is None else np.asarray(w, float).copy()
        if (self.weights.shape != (X.shape[1],) or not np.isfinite(self.weights).all()
                or np.any(self.weights <= 0)):
            raise ValueError("w must contain one finite positive weight per feature")

        self.intercept = bool(intercept)
        metadata = np.empty((len(X), 0)) if covariates is None else np.asarray(covariates, float)
        if metadata.ndim != 2 or len(metadata) != len(X) or not np.isfinite(metadata).all():
            raise ValueError("covariates must be a finite sample-by-covariate matrix")
        self.n_covariates = metadata.shape[1]
        free = np.c_[np.ones(len(X)), metadata] if self.intercept else metadata.copy()
        if free.shape[1] and np.linalg.matrix_rank(free) < free.shape[1]:
            raise ValueError("Unpenalized covariates (including the intercept) must have full column rank; remove redundant encodings")
        self.feature_offset = free.shape[1]
        self.unpenalized = np.r_[np.ones(self.feature_offset, bool), np.zeros(X.shape[1], bool)]
        self.matrix = np.c_[free, X], np.c_[np.zeros((len(C), self.feature_offset)), C], y.copy()
        self.weights = np.r_[np.zeros(self.feature_offset), self.weights]
        self.ridge_diagonal = np.where(self.unpenalized, 0., self.ridge)
        self.dim = len(X), self.matrix[0].shape[1], len(C)
        base = np.zeros((self.dim[1], self.n_classes))
        if self.feature_offset:
            if not self.n_covariates and self.intercept:
                base[0] = self.targets.mean(axis=0)
            else:
                base[:self.feature_offset] = fit_unpenalized_hinge(free, self.targets)
        X, C, _ = self.matrix
        penalized = ~self.unpenalized
        margins = self.targets*X.dot(base)
        negative_gradient_response = 2*self.targets*np.maximum(1-margins, 0.)
        state = initialize_path(X[:, penalized]/self.weights[penalized],
                                negative_gradient_response,
                                C[:, penalized]/self.weights[penalized], tol)
        self.lambdamax = state["lambda_max"]
        self.init = base
        self.active = np.empty(0, dtype=int)
        self.candidates = np.flatnonzero(penalized)[state["candidates"]]
        self.dual = state["dual"]
        self.adjusted_correlations = np.zeros_like(base)
        self.adjusted_correlations[penalized] = (state["adjusted_correlations"]
                                                *self.weights[penalized, None])

    def _coefficient_matrix(self, B):
        B = np.asarray(B, dtype=float)
        if B.shape != (self.dim[1], self.n_classes) or not np.isfinite(B).all():
            raise ValueError("B must be finite with shape (features, classes)")
        return B

    def loss_gradient(self, B):
        """Return squared-hinge loss plus explicit ridge and its p-by-K gradient.

        No feasibility requirement is imposed; this is also useful for KKT
        checks. The loss is differentiable even at margin=1.
        """
        B = self._coefficient_matrix(B)
        X = self.matrix[0]
        residual = np.minimum(self.targets * X.dot(B) - 1.0, 0.0)
        return (np.sum(residual ** 2)+.5*np.sum(self.ridge_diagonal[:, None]*B*B),
                2 * X.T.dot(self.targets * residual)+self.ridge_diagonal[:, None]*B)

    def active_hessians(self, B, active):
        """Return K class-specific active-row Hessians and quadratic masks.

        The result has shape (K, a, a), with a=len(active), rather than
        (pK, pK). A strict margin<1 selects the quadratic region. At exactly
        margin=1 the Hessian is not unique; the future path pivot must choose
        the appropriate one-sided region rather than using this convention
        to decide the path direction.
        """
        B = self._coefficient_matrix(B)
        active = np.asarray(active)
        if active.ndim != 1 or (active.size and active.dtype.kind not in "iu"):
            raise ValueError("active must be a vector of feature indices")
        active = active.astype(int)
        if (len(np.unique(active)) != len(active) or np.any(active < 0)
                or np.any(active >= self.dim[1])):
            raise ValueError("active indices must be distinct and in bounds")
        X = self.matrix[0]
        quadratic = self.targets * X.dot(B) < 1.0
        XA = X[:, active]
        H = np.empty((self.n_classes, len(active), len(active)))
        for k in range(self.n_classes):
            Xk = XA[quadratic[:, k]]
            H[k] = 2 * Xk.T.dot(Xk)+np.diag(self.ridge_diagonal[active])
        return H, quadratic


def Classo_MC1(pb, lam):
    """Fit at relative lambda in [0,1] by tracing and interpolating the path."""
    return pathlasso_MC1(pb, [lam])[0]


def pathlasso_MC1(pb, path, n_active=0):
    """Evaluate the knot path at descending relative lambda values.

    n_active limits nonzero feature rows, excluding the intercept. The path
    stops before an outgoing segment would exceed this limit; simultaneous
    entries are not split. On a sparse stop, return only achieved points,
    including the stop knot. Their relative lambdas are available through
    pb.path_info['evaluated_lambdas']; never extrapolate a stopped solution.
    """
    from .mc1_path_engine import solve_path_MC1

    path = np.asarray(path, float)
    if (path.ndim != 1 or not len(path) or not np.isfinite(path).all()
            or np.any(path < 0) or np.any(path > 1) or np.any(np.diff(path) > 0)):
        raise ValueError("path must be a nonempty descending sequence in [0,1]")
    BETA, knots = solve_path_MC1(pb, path[-1], n_active=n_active)
    if pb.path_info["stop_reason"] == "n_active":
        path = path[path >= knots[-1]]
        if not len(path) or path[-1] > knots[-1]:
            path = np.r_[path, knots[-1]]
    pb.path_info["evaluated_lambdas"] = path.copy()
    if len(knots) == 1:
        return [BETA[0].copy() for lam in path]
    out = []
    for lam in path:
        i = min(max(np.searchsorted(-np.asarray(knots), -lam, side="right")-1, 0), len(knots)-2)
        fraction = (knots[i]-lam)/(knots[i]-knots[i+1])
        out.append((1-fraction)*BETA[i]+fraction*BETA[i+1])
    return out
