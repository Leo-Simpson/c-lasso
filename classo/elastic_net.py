from .misc_functions import _constraint_rows
"""Fixed-ridge elastic net: exact affine paths for piecewise quadratic losses.

The objective adds ridge/2 * ||beta||^2 in ORIGINAL feature coordinates.
Intercepts and explicitly unpenalized covariates receive neither penalty.
R3/R4 put the ridge INSIDE the scale perspective. R3 reparameterizes a
quadratic homotopy; R4 uses its mean-shift reduction to that same solver.
"""
import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from .multiclass_path import initialize_path


def validate_ridge(ridge):
    try:
        if np.ndim(ridge): raise ValueError
        value = float(ridge)
        if not np.isfinite(value) or value < 0: raise ValueError
    except (TypeError, ValueError):
        raise ValueError("ridge must be a finite nonnegative scalar") from None
    return value


class problem_elastic_net:
    """Working-set homotopy problem for R1, R2, C1, C2, MC1 and MC2.

    MC2 uses the signed, classwise Huberized squared-hinge loss with the
    same lower margin rho_classification as C2 and a JOINT row-Linf penalty.
    It is not a Crammer-Singer loss. Free rows precede feature rows.
    """
    def __init__(self, data, typ="R1", ridge=0., w=None, intercept=False,
                 rho=1.345, rho_classification=-1., covariates=None):
        self.ridge = validate_ridge(ridge)
        if typ not in ("R1", "R2", "C1", "C2", "MC1", "MC2"):
            raise ValueError("This homotopy requires a piecewise quadratic loss")
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
        self.ridge_diagonal = np.r_[np.zeros(q), np.full(p,self.ridge)]
        self.unpenalized = self.weights == 0
        if typ.startswith("MC"):
            if y.dtype.kind not in 'biufUS' or (y.dtype.kind == 'f' and not np.isfinite(y).all()):
                raise ValueError("Class labels must be finite numbers or strings")
            self.classes, encoded = np.unique(y, return_inverse=True)
            if len(self.classes) < 2:
                raise ValueError("Multiclass problems require at least two classes")
            self.targets = 2*np.eye(len(self.classes))[encoded]-1
        elif typ.startswith("C"):
            if not set(y).issubset({-1,1}):
                raise ValueError("Binary classification requires labels -1 and +1")
            self.classes = np.array([-1,1])
            self.targets = y.astype(float)[:,None]
        else:
            y = y.astype(float)
            if not np.isfinite(y).all():
                raise ValueError("Outcomes must be finite")
            self.classes = None
            self.targets = np.ones((n,1))
        self.n_classes = self.targets.shape[1]
        self.dim = n, p+q, len(C)
        self._margin_offset = -y[:,None] if typ.startswith('R') else 0.
        self._margin_upper = np.inf if typ == 'R1' else (rho if typ == 'R2' else 1.)
        self._hinge_lower = (-rho if typ == 'R2' else
                             rho_classification if typ in ('C2','MC2') else None)
        if typ == 'R2' and (not np.isfinite(rho) or rho <= 0):
            raise ValueError("rho must be finite and positive")
        if typ in ('C2','MC2') and (not np.isfinite(rho_classification) or rho_classification >= 1):
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
        if self.typ == 'R1':
            return np.sum(m*m), 2*m
        if self.typ == 'R2':
            r = self._margin_upper
            a = abs(m)
            return np.sum(np.where(a <= r,m*m,2*r*a-r*r)), 2*np.clip(m,-r,r)
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


def _concomitant_path(matrix, values, typ, ridge, w, intercept, e, rho,
                      true_lam, n_active):
    from .concomitant_ridge import concomitant_path
    return concomitant_path(matrix,values,typ,ridge,w,intercept,e,rho,true_lam,n_active)


def ridge_path(matrix, values, typ, ridge, w=None, intercept=False, rho=1.345,
               rho_classification=-1., covariates=None, true_lam=False,
               e=None, n_active=0):
    """Internal common dispatch; returns coefficients, actual lambdas, sigma, max."""
    ridge = validate_ridge(ridge)
    values = np.asarray(values,float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or np.any(values<0):
        raise ValueError("lambdas must be a finite nonnegative vector")
    if not isinstance(n_active,(int,np.integer)) or n_active<0: raise ValueError("n_active must be a nonnegative integer")
    values = np.sort(values)[::-1]
    if not true_lam and np.any(values>1): raise ValueError("Relative lambdas must lie in [0,1]")
    if typ in ('R3','R4'):
        if covariates is not None: raise ValueError("Unpenalized covariates are not supported for ridge R3/R4; intercept is supported")
        return _concomitant_path(matrix,values,typ,ridge,w,intercept,e,rho,true_lam,n_active)
    from .solve_MC1 import pathlasso_MC1
    pb = problem_elastic_net(matrix,typ,ridge,w,intercept,rho,rho_classification,covariates)
    relative = np.minimum(values/pb.lambdamax,1.) if true_lam and pb.lambdamax else (np.ones_like(values) if true_lam else values)
    coefficients = np.asarray(pathlasso_MC1(pb,relative,n_active=n_active))
    real = values if true_lam else values*pb.lambdamax
    if pb.path_info['stop_reason'] == 'n_active': real = pb.path_info['evaluated_lambdas']*pb.lambdamax
    if not typ.startswith('MC'): coefficients = coefficients[:,:,0]
    return coefficients,np.asarray(real),None,pb.lambdamax


def solve_path_elastic_net(pb, lamin=.01, max_events=2000, tol=1e-8, n_active=0):
    """Return every affine knot and relative lambda for a fixed-scale problem.

    Diagnostics are in pb.path_info. Concomitant R3/R4 deliberately cannot
    construct problem_elastic_net: use Classo/pathlasso for their concomitant fits.
    """
    from .mc1_path_engine import solve_path_MC1
    return solve_path_MC1(pb,lamin,max_events,tol,n_active)
