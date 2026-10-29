import numpy as np

from .multiclass_path import initialize_path


r"""
Problem : min sum_ik max(1 - T_ik * (X B)_ik, 0)^2
              + lambda sum_j max_k |B_jk|, subject to C B = 0.

T_ik is +1 for the observed class and -1 otherwise. The class scores
are coupled by the row penalty: this is a joint problem, not K binary fits.
The squared loss is unhalved, consistently with C1 and R1.

Classo_MC1 and pathlasso_MC1 use an event-driven path engine, with explicit
failure on unresolved singular/zero-step states. No implicit ridge is added.
"""

tol = 1e-8


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
    """

    def __init__(self, data, algo="Path-Alg", w=None, intercept=False):
        if algo != "Path-Alg":
            raise ValueError("MC1 currently reserves only the Path-Alg method")
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
        self.weights = np.ones(X.shape[1]) if w is None else np.asarray(w, float).copy()
        if (self.weights.shape != (X.shape[1],) or not np.isfinite(self.weights).all()
                or np.any(self.weights <= 0)):
            raise ValueError("w must contain one finite positive weight per feature")

        self.intercept = bool(intercept)
        self.unpenalized = np.zeros(X.shape[1], dtype=bool)
        base = np.zeros((X.shape[1], self.n_classes))
        if self.intercept:
            X, C, y = self.matrix
            self.matrix = np.c_[np.ones(len(X)), X], np.c_[np.zeros(len(C)), C], y
            self.weights = np.r_[0., self.weights]
            self.unpenalized = np.r_[True, self.unpenalized]
            self.dim = self.dim[0], self.dim[1]+1, self.dim[2]
            base = np.zeros((self.dim[1], self.n_classes))
            base[0] = self.targets.mean(axis=0)
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
        """Return the unpenalized squared-hinge loss and its p-by-K gradient.

        No feasibility requirement is imposed; this is also useful for KKT
        checks. The loss is differentiable even at margin=1.
        """
        B = self._coefficient_matrix(B)
        X = self.matrix[0]
        residual = np.minimum(self.targets * X.dot(B) - 1.0, 0.0)
        return np.sum(residual ** 2), 2 * X.T.dot(self.targets * residual)

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
            H[k] = 2 * Xk.T.dot(Xk)
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
