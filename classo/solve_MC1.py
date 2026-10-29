import numpy as np

from .multiclass_path import initialize_path


r"""
Problem : min sum_ik max(1 - T_ik * (X B)_ik, 0)^2
              + lambda sum_j max_k |B_jk|, subject to C B = 0.

T_ik is +1 for the observed class and -1 otherwise. The class scores
are coupled by the row penalty: this is a joint problem, not K binary fits.
The squared loss is unhalved, consistently with C1 and R1.

This module currently provides the problem data and local quadratic pieces.
It does not yet provide Classo_MC1 or pathlasso_MC1.
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

    Intercepts and feature weights are not supported by this first increment.
    classes defines the column order of B; targets contains signed indicators.
    init is the zero coefficient matrix. active is empty at lambdamax;
    candidates contains dual-tight rows, not a selected entering face.
    """

    def __init__(self, data, algo="Path-Alg"):
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

        # At B=0, -grad(loss)=2 X.T T. Reuse the general constrained
        # dual-norm calculation with Y=2T; no full feature Gram is needed.
        state = initialize_path(self.matrix[0], 2 * self.targets, self.matrix[1], tol)
        self.lambdamax = state["lambda_max"]
        self.init = state["B"]
        self.active = state["active"]
        self.candidates = state["candidates"]
        self.dual = state["dual"]
        self.adjusted_correlations = state["adjusted_correlations"]

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
