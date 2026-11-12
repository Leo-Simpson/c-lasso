"""Joint MC2 loss, independent of the ridge extension."""
import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from .misc_functions import _constraint_rows
from .multiclass_path import initialize_path
from .solve_MC1 import Classo_MC1, pathlasso_MC1

class problem_MC2:
    """Joint row-Linf Huberized squared-hinge homotopy problem.

    MC2 uses the signed, classwise Huberized squared-hinge loss with the
    same lower margin rho_classification as C2 and a JOINT row-Linf penalty.
    It is not a Crammer-Singer loss. Free rows precede feature rows.
    """
    def __init__(self, data, algo="Path-Alg", w=None, intercept=False,
                 covariates=None, rho_classification=-1.):
        if algo != "Path-Alg":
            raise ValueError("MC2 supports only the Path-Alg method")
        typ = "MC2"
        self.ridge = 0.0
        self.type, self.name = algo, algo + " MC2"
        self.regpath = False
        X, C, y = data
        X = sparse.csr_matrix(X, dtype=float) if sparse.issparse(X) else np.asarray(X, float)
        y = np.asarray(y)
        if X.ndim != 2 or not all(X.shape) or not np.isfinite(X.data if sparse.issparse(X) else X).all():
            raise ValueError("X must be a finite nonempty matrix")
        n, p = X.shape
        if y.shape != (n,):
            raise ValueError("y must contain one outcome per sample")
        C = np.empty((0,p)) if C is None else np.asarray(C, float)
        if C.ndim != 2 or C.shape[1] != p or not np.isfinite(C).all():
            raise ValueError("C must be finite with one column per feature")
        self.original_constraints = C.copy()
        C = _constraint_rows(C, X.shape[1])
        w = np.ones(p) if w is None else np.asarray(w, float)
        if w.shape != (p,) or not np.isfinite(w).all() or np.any(w <= 0):
            raise ValueError("w must contain one finite positive weight per feature")
        Z = np.empty((n,0)) if covariates is None else np.asarray(covariates, float)
        if Z.ndim != 2 or len(Z) != n or not np.isfinite(Z).all():
            raise ValueError("covariates must be a finite sample-by-covariate matrix")
        free = np.c_[np.ones(n), Z] if intercept else Z
        if free.shape[1] and np.linalg.matrix_rank(free) < free.shape[1]:
            raise ValueError("Remove redundant unpenalized covariates")
        self.typ, self.tol = typ, 1e-8
        self.intercept, self.n_covariates = bool(intercept), Z.shape[1]
        self.feature_offset = q = free.shape[1]
        design = sparse.hstack([sparse.csr_matrix(free), X],format="csr") if sparse.issparse(X) else np.c_[free,X]
        self.matrix = design, np.c_[np.zeros((len(C),q)), C], y.copy()
        self.weights = np.r_[np.zeros(q),w]
        self.ridge_diagonal = np.zeros(p+q)
        self.unpenalized = self.weights == 0
        if y.dtype.kind not in 'biufUS' or (y.dtype.kind == 'f' and not np.isfinite(y).all()):
            raise ValueError("Class labels must be finite numbers or strings")
        self.classes, encoded = np.unique(y, return_inverse=True)
        if len(self.classes) < 2:
            raise ValueError("Multiclass problems require at least two classes")
        self.targets = 2*np.eye(len(self.classes))[encoded]-1
        self.n_classes = self.targets.shape[1]
        self.dim = n, p+q, len(C)
        self._margin_offset = 0.
        self._margin_upper = 1.
        self._hinge_lower = rho_classification
        if not np.isfinite(rho_classification) or rho_classification >= 1:
            raise ValueError("rho_classification must be finite and smaller than 1")
        self.init = np.zeros((p+q,self.n_classes))
        if q:
            def objective(v):
                B = self.init.copy(); B[:q] = v.reshape(q,self.n_classes)
                value, gradient = self.loss_gradient(B)
                return value, gradient[:q].ravel()
            fit = minimize(objective, np.zeros(q*self.n_classes), jac=True,
                           method='L-BFGS-B', options=dict(gtol=1e-11,ftol=1e-15,maxiter=5000,maxls=50))
            # Local quadratic refinement, preserving the current representative
            # when a free loss direction is flat.
            v = fit.x
            for _ in range(30):
                value,g = objective(v)
                if np.max(abs(g),initial=0.) < 1e-11: break
                B = self.init.copy(); B[:q] = v.reshape(q,self.n_classes)
                margin = self.targets*self.matrix[0].dot(B)+self._margin_offset
                mask = margin < self._margin_upper
                if self._hinge_lower is not None: mask &= margin > self._hinge_lower
                correction = np.zeros((q,self.n_classes))
                for k in range(self.n_classes):
                    H = 2*free[mask[:,k]].T.dot(free[mask[:,k]])
                    correction[:,k] = np.linalg.lstsq(H,-g.reshape(q,-1)[:,k],rcond=1e-12)[0]
                for step in 2.**(-np.arange(25)):
                    trial = v+step*correction.ravel()
                    if objective(trial)[0] <= value+1e-13: v=trial; break
                else: break
            if np.max(abs(objective(v)[1]),initial=0.) > 1e-10:
                raise RuntimeError("Unpenalized initialization failed stationarity")
            self.init[:q] = v.reshape(q,self.n_classes)
        # Gradient = X.T @ response. Ridge vanishes at the feature-zero start.
        _, response = self._response(self.matrix[0].dot(self.init))
        weighted_X = X.multiply(1/w).tocsr() if sparse.issparse(X) else X/w
        state = initialize_path(weighted_X, -response, C/w, self.tol)
        self.lambdamax, self.dual = state['lambda_max'], state['dual']
        self.candidates = q+state['candidates']
        self.active = np.empty(0,dtype=int)
        self._mc1_allow_tree_nullspace = True

    def _response(self, scores):
        m = self.targets*scores+self._margin_offset
        upper = np.minimum(m-1.,0.)
        lower = self._hinge_lower
        if lower is None:
            return np.sum(upper*upper), 2*self.targets*upper
        h = 1.-lower
        loss = np.where(m < lower, h*h+2*h*(lower-m), upper*upper)
        return np.sum(loss), 2*self.targets*np.clip(m-1.,-h,0.)

    def loss_gradient(self, B):
        B = np.asarray(B,float)
        if B.shape != self.init.shape or not np.isfinite(B).all():
            raise ValueError("Invalid coefficient matrix")
        loss, response = self._response(self.matrix[0].dot(B))
        return (loss+.5*np.sum(self.ridge_diagonal[:,None]*B*B),
                self.matrix[0].T.dot(response)+self.ridge_diagonal[:,None]*B)


Classo_MC2 = Classo_MC1
pathlasso_MC2 = pathlasso_MC1
