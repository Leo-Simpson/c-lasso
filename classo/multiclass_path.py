"""Fixed-face building blocks for constrained multiclass least squares.

Objective: 0.5 * ||X B - Y||_F**2 + lam * sum_j ||B[j]||_inf,
subject to C B = 0. No intercept; callers must handle centering explicitly.
This is infrastructure, not a complete event/pivot homotopy solver.
"""
import numpy as np


def initialize_path(X, Y, C=None, tol=1e-8):
    """Return lambda_max, zero B, empty support, boundary candidates and dual.

    With constraints, lambda_max = min_U max_j ||X.T Y - C.T U||_1.
    The constrained case uses a sparse SciPy LP, with O(p K + m K)
    variables; it never forms a pK-by-pK Hessian.
    Boundary candidates are NOT necessarily the support immediately below
    lambda_max. Degenerate starts require a joint feasible-face pivot.
    """
    X, Y = np.asarray(X, float), np.asarray(Y, float)
    if X.ndim != 2 or Y.ndim != 2 or X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must be matrices with matching sample counts")
    p, K = X.shape[1], Y.shape[1]
    if not p or not K or not np.isfinite(X).all() or not np.isfinite(Y).all():
        raise ValueError("Require nonempty feature/class axes and finite inputs")
    C = np.empty((0, p)) if C is None else np.asarray(C, float)
    if C.ndim != 2 or C.shape[1] != p or not np.isfinite(C).all():
        raise ValueError("C must be finite with shape (m, p)")
    R = X.T @ Y
    U = np.zeros((C.shape[0], K))
    if C.shape[0]:
        from scipy.optimize import linprog
        from scipy import sparse as sp

        # Column-major ordering. Variables are vec(U), vec(T), lam.
        D = sp.kron(sp.eye(K), sp.csc_matrix(C.T), format="csc")
        I = sp.eye(p * K, format="csc")
        Z = sp.csc_matrix((p * K, 1))
        S = sp.kron(np.ones((1, K)), sp.eye(p), format="csc")
        A = sp.vstack([
            sp.hstack([-D, -I, Z]),
            sp.hstack([D, -I, Z]),
            sp.hstack([sp.csc_matrix((p, U.size)), S, -np.ones((p, 1))]),
        ], format="csc")
        r = R.ravel(order="F")
        objective = np.zeros(U.size + p * K + 1)
        objective[-1] = 1
        fit = linprog(objective, A_ub=A, b_ub=np.r_[-r, r, np.zeros(p)],
                      bounds=[(None, None)] * U.size + [(0, None)] * (p*K+1),
                      method="highs")
        if not fit.success:
            raise RuntimeError(f"lambda_max LP failed: {fit.message}")
        U = fit.x[:U.size].reshape(U.shape, order="F")
    adjusted = R - C.T @ U
    norms = np.abs(adjusted).sum(axis=1)
    lam = float(norms.max())
    candidates = (np.flatnonzero(np.abs(norms - lam) <= tol * max(1., lam))
                  if lam > tol else np.empty(0, dtype=int))
    return dict(lambda_max=lam, B=np.zeros((p, K)),
                active=np.empty(0, dtype=int), candidates=candidates,
                dual=U, adjusted_correlations=adjusted)


def face_direction(X, C, active, faces, n_classes, tol=1e-9, hessians=None, weights=None):
    """Compute dB_A/dlambda and dt_A/dlambda on a valid fixed face.

    faces[a] is a sequence of (class_index, sign) pairs binding on active
    row a: sign * B[a, k] = t[a], t[a] > 0. Include every binding face.
    Other active-row coordinates remain free (they need not be zero).
    Caller must supply a primal/dual-valid face and select all next events.
    The direction is with respect to INCREASING lambda; descending updates
    use B_new = B + (lambda_new - lambda_old) * dB.

    Dense reference implementation: factors only active variables, with
    aK+a primal variables. It does not promise an a-by-a system.
    Singular systems are rejected: silently choosing a pseudoinverse can
    hide nonunique paths and requires an explicit degeneracy policy.
    Optional hessians has shape (K, a, a) and supplies class-specific loss
    Hessians, for example the local unhalved squared-hinge pieces in MC1.
    """
    X = np.asarray(X, float)
    active = np.asarray(active, dtype=int)
    K, a = int(n_classes), len(active)
    weights = np.ones(a) if weights is None else np.asarray(weights, float)
    if weights.shape != (a,) or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("weights must be finite and positive, one per active row")
    if K < 1 or len(np.unique(active)) != a or np.any(active < 0) or np.any(active >= X.shape[1]):
        raise ValueError("Invalid active rows or class count")
    C = np.empty((0, X.shape[1])) if C is None else np.asarray(C, float)
    if len(faces) != a or C.ndim != 2 or C.shape[1] != X.shape[1]:
        raise ValueError("Face/constraint dimensions do not match")
    if not a:
        return np.empty((0, K)), np.empty(0)
    if hessians is None:
        G = X[:, active].T @ X[:, active]
        hessians = np.repeat(G[None], K, axis=0)
    hessians = np.asarray(hessians, float)
    if hessians.shape != (K, a, a) or not np.isfinite(hessians).all():
        raise ValueError("hessians must be finite with shape (classes, active, active)")
    if not np.allclose(hessians, hessians.transpose(0, 2, 1)):
        raise ValueError("Loss Hessians must be symmetric")
    # Column-major B_A: one a-by-a Gram block for each class.
    H = np.zeros((a*K+a, a*K+a))
    for k in range(K):
        H[k*a:(k+1)*a, k*a:(k+1)*a] = hessians[k]
    rows = []
    for k in range(K):
        for c in C[:, active]:
            row = np.zeros(a*K+a)
            row[k*a:(k+1)*a] = c
            rows.append(row)
    for j, bindings in enumerate(faces):
        if not bindings:
            raise ValueError("Each active row must have at least one binding face")
        for k, sign in bindings:
            if k not in range(K) or sign not in (-1, 1):
                raise ValueError("Faces require a class index and sign +/-1")
            row = np.zeros(a*K+a)
            row[k*a+j], row[a*K+j] = sign, -1
            rows.append(row)
    E = np.asarray(rows)
    # Remove redundant equality rows (e.g. dependent tree constraints).
    _, singular, vh = np.linalg.svd(E, full_matrices=False)
    rank = np.sum(singular > tol * singular[0])
    E = vh[:rank]
    M = np.block([[H, E.T], [E, np.zeros((rank, rank))]])
    rhs = np.r_[np.zeros(a*K), -weights, np.zeros(rank)]
    if np.linalg.matrix_rank(M, tol=tol * np.linalg.norm(M, 2)) < len(M):
        raise np.linalg.LinAlgError("Nonunique fixed-face direction; pivot or reduce the face")
    direction = np.linalg.solve(M, rhs)
    if not np.allclose(M @ direction, rhs, atol=tol, rtol=tol):
        raise np.linalg.LinAlgError("KKT residual exceeds tolerance")
    return direction[:a*K].reshape((a, K), order="F"), direction[a*K:a*K+a]
