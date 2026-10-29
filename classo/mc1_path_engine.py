"""Event-driven MC1 homotopy, with directional face selection.

Small working-set dense direction solves and sparse full-feature certificate
LPs. No lambda grid, finite-difference direction, or implicit ridge is used.
This floating-point implementation rejects unresolved singular/zero-step states;
it does not claim a general finite-termination theorem for degenerate MC1.
"""
import numpy as np
from scipy import sparse
from scipy.optimize import linprog, minimize

from .multiclass_path import face_direction


def _outgoing_direction(pb, B, lam, certificate, tol):
    X, C, _ = pb.matrix
    p, K = B.shape
    weights = getattr(pb, "weights", np.ones(p))
    t = np.max(np.abs(B), axis=1)
    tight = np.abs(certificate).sum(axis=1) >= lam*weights - 20*tol*max(1., lam)
    working = np.flatnonzero((t > tol) | tight)
    a = len(working)
    if not a:
        raise RuntimeError("No outgoing MC1 working set")
    XA, BA, ta = X[:, working], B[working], t[working]
    wa = weights[working]
    _, gradient = pb.loss_gradient(B)
    # Critical cone: tree tangent and equality for faces carrying strictly
    # positive certificate mass. This is equivalent to the scalar critical
    # equality but avoids its poorly conditioned nearly-dependent form.
    E = np.c_[np.kron(C[:, working], np.eye(K)), np.zeros((len(C)*K, a))]
    inequalities = []
    for j in range(a):
        for k in range(K):
            for sign in (-1, 1):
                if ta[j] - sign*BA[j, k] <= 20*tol*max(1., ta[j]):
                    row = np.zeros(a*K+a)
                    row[a*K+j], row[j*K+k] = 1., -sign
                    if sign*certificate[working[j], k] > 20*tol*max(1., lam*wa[j]):
                        E = np.vstack([E, row])
                    else:
                        inequalities.append(row)
    if len(E):
        _, singular, vh = np.linalg.svd(E, full_matrices=False)
        rank = np.sum(singular > tol*max(1., singular[0]))
        E = vh[:rank]
    else:
        E = np.empty((0, a*K+a))
    J = np.asarray(inequalities).reshape(-1, a*K+a)
    # Eliminate equalities before the numerical QP. On constrained starts,
    # tangent inequalities can become identically zero in this subspace;
    # leaving them in SLSQP can falsely certify the zero direction.
    _, se, ve = np.linalg.svd(E, full_matrices=True)
    erank = np.sum(se > tol)
    N = ve[erank:].T
    if not N.shape[1]:
        raise RuntimeError("MC1 critical cone has no free directions")
    JN = J.dot(N)
    JN = JN[np.linalg.norm(JN, axis=1) > 10*tol]
    margins = pb.targets * X.dot(B)
    interior = margins < 1. - 20*tol
    boundary = np.abs(margins-1.) <= 20*tol

    def objective(z):
        D = z[:a*K].reshape(a, K)
        scores = XA.dot(D)
        # One-sided second-order loss at sample-margin boundaries.
        r = np.where(interior, scores, 0.)
        r += np.where(boundary, pb.targets*np.minimum(pb.targets*scores, 0.), 0.)
        return (np.sum(r*r) - wa.dot(z[a*K:]),
                np.r_[2*XA.T.dot(r).ravel(), -wa])

    def reduced_objective(v):
        value, gradient_v = objective(N.dot(v))
        return value, N.T.dot(gradient_v)

    constraints = []
    if len(JN):
        constraints.append(dict(type="ineq", fun=lambda v: JN.dot(v), jac=lambda v: JN))
    fit = minimize(reduced_objective, np.zeros(N.shape[1]), jac=True, method="SLSQP",
                   constraints=constraints, options=dict(ftol=1e-12, maxiter=2000))
    if not fit.success:
        raise RuntimeError("MC1 directional face selection failed: " + fit.message)
    z = N.dot(fit.x)
    if (np.max(np.abs(E.dot(z)), initial=0.) > 1e-7
            or np.min(J.dot(z), initial=0.) < -1e-7):
        raise RuntimeError("MC1 directional cone residual exceeds tolerance")
    D = np.zeros_like(B)
    D[working] = z[:a*K].reshape(a, K)
    dt = np.zeros(p)
    dt[working] = z[a*K:]
    active = np.flatnonzero((t > tol) | (np.max(np.abs(D), axis=1) > tol))
    if not len(active):
        raise RuntimeError("MC1 directional program returned no moving rows")
    faces = []
    for j in active:
        face = []
        for k in range(K):
            for sign in (-1, 1):
                if (t[j]-sign*B[j, k] <= 20*tol*max(1., t[j])
                        and abs(dt[j]-sign*D[j, k]) <= 2e-6*max(1., abs(dt[j]))):
                    face.append((k, sign))
        if not face:
            raise RuntimeError("Cannot identify an outgoing MC1 maximum face")
        faces.append(face)
    margin_rate = pb.targets * X.dot(D)
    quadratic = interior | (boundary & (margin_rate < 0))
    H = np.asarray([2*X[quadratic[:, k]][:, active].T.dot(
        X[quadratic[:, k]][:, active]) for k in range(K)])
    # As in c-lasso, refine the selected state by linear algebra before moving.
    beta_dot, t_dot = face_direction(X, C, active, faces, K, tol=tol,
                                    hessians=H, weights=weights[active])
    refined = np.zeros_like(B)
    refined[active] = -beta_dot  # derivative w.r.t. decrease in lambda
    refined_t = np.zeros(p)
    refined_t[active] = -t_dot
    # SLSQP selects the face; its direction need not have linear-solve
    # accuracy. Validate the refined direction itself, not proximity to the
    # approximate QP iterate. The subsequent LP also requires a valid KKT
    # certificate for an interval of positive length.
    local = np.r_[refined[working].ravel(), refined_t[working]]
    if (np.min(J.dot(local), initial=0.) < -1e-7
            or np.max(np.abs(E.dot(local)), initial=0.) > 1e-7):
        raise RuntimeError("MC1 refined direction violates the outgoing cone")
    refined_rate = pb.targets*X.dot(refined)
    if np.any(boundary & np.where(quadratic, refined_rate > 1e-7, refined_rate < -1e-7)):
        raise RuntimeError("MC1 refined direction changes its selected sample region")
    critical = np.sum(gradient*refined) + lam*weights.dot(refined_t)
    if abs(critical) > 2e-6*max(1., abs(lam*weights.dot(refined_t))):
        raise RuntimeError("MC1 outgoing direction is outside the critical cone")
    return refined, refined_t, active, faces, quadratic, len(working)


def _segment_end(pb, B, lam, D, dt, active, faces, quadratic, cap, tol):
    """Maximize the certified affine interval, including all dual events.

    Instead of selecting arbitrary nonunique multipliers and stopping when
    one fails, the sparse LP asks whether ANY valid certificate exists along
    this primal segment. This projection is an MC1 implementation choice, not
    the canonical-certificate traversal from the CLASSO paper.
    """
    X, C, _ = pb.matrix
    p, K = B.shape
    m = len(C)
    weights = getattr(pb, "weights", np.ones(p))
    t = np.max(np.abs(B), axis=1)
    margins, rate = pb.targets*X.dot(B), pb.targets*X.dot(D)
    candidates = [(float(cap), "endpoint")]
    for i, k in np.ndindex(margins.shape):
        r = rate[i, k]
        if ((quadratic[i, k] and r > 0) or (not quadratic[i, k] and r < 0)):
            root = (1.-margins[i, k])/r
            if root > 20*tol:
                candidates.append((float(root), "sample_margin"))
    for j in active:
        if dt[j] < 0 and t[j] > tol:
            candidates.append((float(-t[j]/dt[j]), "row_exit"))
        binding = set(faces[list(active).index(j)])
        for k in range(K):
            for sign in (-1, 1):
                slack, speed = t[j]-sign*B[j, k], dt[j]-sign*D[j, k]
                if (k, sign) not in binding and speed < -tol:
                    root = slack/(-speed)
                    if root > 20*tol:
                        candidates.append((float(root), "maximum_face"))
    upper = min(value for value, _ in candidates)
    _, grad = pb.loss_gradient(B)
    gdot = 2*X.T.dot(quadratic*X.dot(D))
    # Variables: delta, vec(U), vec(S), vec(abs(S)) (row-major).
    nu, ns = m*K, p*K
    start_s, start_t = 1+nu, 1+nu+ns
    total = 1+nu+2*ns
    stationarity = sparse.hstack([
        sparse.csc_matrix(gdot.reshape(-1, 1)),
        sparse.kron(sparse.csc_matrix(C.T), sparse.eye(K)),
        sparse.eye(ns), sparse.csc_matrix((ns, ns))], format="csc")
    eq = [stationarity]
    eq_rhs = [-grad.ravel()]
    bounds = [(0., upper)] + [(None, None)]*nu + [(None, None)]*ns + [(0., None)]*ns
    active_set = set(active)
    rows, rhs = [], []
    for j in range(p):
        if j in active_set:
            allowed = dict(faces[list(active).index(j)])
            row = sparse.lil_matrix((1, total))
            row[0, 0] = weights[j]
            for k in range(K):
                pos = start_s+j*K+k
                if k in allowed:
                    sign = allowed[k]
                    bounds[pos] = (0., None) if sign == 1 else (None, 0.)
                    row[0, pos] = sign
                else:
                    bounds[pos] = (0., 0.)
                bounds[start_t+j*K+k] = (0., 0.)
            eq.append(row.tocsc())
            eq_rhs.append(np.array([lam*weights[j]]))
        else:
            for k in range(K):
                for sign in (-1, 1):
                    row = sparse.lil_matrix((1, total))
                    row[0, start_s+j*K+k] = sign
                    row[0, start_t+j*K+k] = -1.
                    rows.append(row.tocsc())
                    rhs.append(0.)
            row = sparse.lil_matrix((1, total))
            row[0, 0] = weights[j]
            row[0, start_t+j*K:start_t+(j+1)*K] = 1.
            rows.append(row.tocsc())
            rhs.append(lam*weights[j])
    objective = np.zeros(total)
    objective[0] = -1.
    fit = linprog(objective, A_ub=sparse.vstack(rows) if rows else None,
                  b_ub=rhs if rows else None, A_eq=sparse.vstack(eq),
                  b_eq=np.concatenate(eq_rhs), bounds=bounds, method="highs",
                  options=dict(primal_feasibility_tolerance=1e-9,
                               dual_feasibility_tolerance=1e-9))
    if not fit.success:
        raise RuntimeError("MC1 segment certificate LP failed: " + fit.message)
    step = float(fit.x[0])
    if step <= tol*max(1., lam):
        raise RuntimeError("MC1 zero-length segment; unresolved face or degeneracy")
    S = fit.x[start_s:start_s+ns].reshape(p, K)
    U = fit.x[1:1+nu].reshape(m, K)
    new_B = B+step*D
    _, new_grad = pb.loss_gradient(new_B)
    residual = max(np.max(np.abs(new_grad+C.T.dot(U)+S)),
                   np.max(np.abs(C.dot(new_B)), initial=0.))
    if residual > 5e-7*max(1., np.max(np.abs(new_grad))):
        raise RuntimeError("MC1 endpoint KKT residual exceeds tolerance")
    reasons = [reason for value, reason in candidates
               if abs(value-step) <= 50*tol*max(1., lam)]
    if not reasons:
        reasons = ["dual_boundary"]
    return step, S, reasons, residual


def solve_path_MC1(pb, lamin=.01, max_events=2000, tol=1e-8):
    """Trace MC1 knots down to lamin*lambda_max, without refitting a grid.

    Returns BETA and relative LAMBDAS as in c-lasso's solve_path. Diagnostics
    (events, working-set sizes and endpoint residuals) are stored on pb.path_info.
    A rank-deficient outgoing face or an unresolved zero step raises, rather
    than silently adding ridge or returning an incomplete path as successful.
    Direction QPs use working rows; certificate LPs scan all feature rows.
    """
    if not np.isfinite(lamin) or not 0 <= lamin <= 1:
        raise ValueError("lamin must lie in [0,1]")
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be finite and positive")
    if max_events < 1:
        raise ValueError("max_events must be positive")
    B, lam = pb.init.copy(), pb.lambdamax
    BETA, LAMBDAS = [B.copy()], [1.]
    info = dict(events=[], working_sizes=[], residuals=[], completed=False)
    pb.path_info = info
    if lam == 0 or lamin == 1:
        info["completed"] = True
        if lamin < 1:
            BETA.append(B.copy())
            LAMBDAS.append(float(lamin))
        return BETA, LAMBDAS
    _, grad = pb.loss_gradient(B)
    certificate = -grad-pb.matrix[1].T.dot(pb.dual)
    target = lamin*pb.lambdamax
    for _ in range(max_events):
        D, dt, active, faces, quadratic, size = _outgoing_direction(pb, B, lam, certificate, tol)
        step, certificate, reasons, residual = _segment_end(
            pb, B, lam, D, dt, active, faces, quadratic, lam-target, tol)
        B, lam = B+step*D, max(target, lam-step)
        BETA.append(B.copy())
        LAMBDAS.append(lam/pb.lambdamax)
        info["events"].append(reasons)
        info["working_sizes"].append(size)
        info["residuals"].append(residual)
        if lam <= target + tol*max(1., pb.lambdamax):
            info["completed"] = True
            return BETA, LAMBDAS
    raise RuntimeError("MC1 exceeded max_events before reaching the requested endpoint")
