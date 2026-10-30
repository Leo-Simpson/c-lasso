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


def _linear_direction(N, JN, XA, targets, interior, boundary, weights, objective, tol):
    """Try equality-only piecewise-quadratic solves; certify global optimality.

    A feasible point with zero reduced gradient minimizes the convex directional
    objective even in the presence of the remaining cone inequalities. Failure
    to find one merely requests the general constrained QP.
    """
    a, K = len(weights), targets.shape[1]
    blocks = N[:a*K].reshape(a, K, -1)
    scores = np.einsum('ij,jkr->ikr', XA, blocks)
    region = interior | boundary
    seen = set()
    for _ in range(8):
        key = region.tobytes()
        if key in seen:
            return None
        seen.add(key)
        selected_scores = scores[region]
        H = 2*selected_scores.T.dot(selected_scores)
        rhs = N[a*K:].T.dot(weights)
        try:
            v = np.linalg.solve(H, rhs)
        except np.linalg.LinAlgError:
            return None
        if not np.isfinite(v).all():
            return None
        z = N.dot(v)
        _, gradient = objective(z)
        if (np.max(np.abs(N.T.dot(gradient)), initial=0.) <= 1e-8
                and np.min(JN.dot(v), initial=0.) >= -1e-9):
            return z
        rate = targets*np.einsum('ikr,r->ik', scores, v)
        region = interior | (boundary & (rate < 0))
    return None


def _endpoint_certificate(pb, grad, lam, active, faces, tol):
    """Try a small multiplier solve, then check every row's subgradient.

    Feasibility at both endpoints certifies the full affine interval by convex
    interpolation. A failed candidate is inconclusive and requests the LP.
    """
    C = pb.matrix[1]
    p, K = grad.shape
    m = len(C)
    rows, rhs = [], []
    for j, binding in zip(active, faces):
        binding = set(binding)
        classes = {k for k, sign in binding}
        for k in range(K):
            if pb.weights[j] == 0 or k not in classes:
                row = np.zeros((m, K)); row[:, k] = C[:, j]
                rows.append(row.ravel()); rhs.append(-grad[j, k])
        if pb.weights[j] > 0:
            row = np.zeros((m, K))
            for k, sign in binding:
                row[:, k] += sign*C[:, j]
            rows.append(row.ravel())
            rhs.append(-sum(sign*grad[j, k] for k, sign in binding)-lam*pb.weights[j])
    if m:
        if not rows:
            return None
        U = np.linalg.lstsq(np.asarray(rows), np.asarray(rhs), rcond=None)[0].reshape(m, K)
    else:
        U = np.empty((0, K))
    S = -grad-C.T.dot(U)
    tolerance = 5e-9*max(1., np.max(np.abs(grad)), lam)
    selected = np.zeros(p, dtype=bool); selected[active] = True
    if np.any(np.abs(S[~selected]).sum(axis=1) > lam*pb.weights[~selected]+tolerance):
        return None
    for j, binding in zip(active, faces):
        if pb.weights[j] == 0:
            if np.max(np.abs(S[j]), initial=0.) > tolerance:
                return None
            continue
        allowed = np.zeros(K, dtype=bool)
        for k, sign in binding:
            allowed[k] = True
            if sign*S[j, k] < -tolerance:
                return None
        if (np.max(np.abs(S[j, ~allowed]), initial=0.) > tolerance
                or abs(sum(sign*S[j, k] for k, sign in binding)-lam*pb.weights[j]) > tolerance):
            return None
    return U, S


def _outgoing_direction(pb, B, lam, certificate, tol):
    X, C, _ = pb.matrix
    p, K = B.shape
    weights = getattr(pb, "weights", np.ones(p))
    t = np.max(np.abs(B), axis=1)
    t[weights == 0] = 0.
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
        if wa[j] == 0:
            row = np.zeros(a*K+a)
            row[a*K+j] = 1.
            E = np.vstack([E, row])
            continue
        for k in range(K):
            for sign in (-1, 1):
                if ta[j] - sign*BA[j, k] <= 20*tol*max(1., ta[j]):
                    row = np.zeros(a*K+a)
                    row[a*K+j], row[j*K+k] = 1., -sign
                    if sign*certificate[working[j], k] > 20*tol*max(1., lam*wa[j]):
                        E = np.vstack([E, row])
                    else:
                        inequalities.append(row)
    J = np.asarray(inequalities).reshape(-1, a*K+a)
    # Eliminate equalities before the numerical QP. On constrained starts,
    # tangent inequalities can become identically zero in this subspace;
    # leaving them in SLSQP can falsely certify the zero direction.
    if len(E):
        _, singular, vh = np.linalg.svd(E, full_matrices=True)
        rank = np.sum(singular > tol*max(1., singular[0]))
        E, N = vh[:rank], vh[rank:].T
    else:
        E, N = np.empty((0, a*K+a)), np.eye(a*K+a)
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
    z = (_linear_direction(N, JN, XA, pb.targets, interior, boundary, wa, objective, tol)
         if getattr(pb, "_mc1_fast", True) else None)
    if z is None:
        fit = minimize(reduced_objective, np.zeros(N.shape[1]), jac=True, method="SLSQP",
                       constraints=constraints, options=dict(ftol=1e-12, maxiter=2000))
        if not fit.success:
            raise RuntimeError("MC1 directional face selection failed: " + fit.message)
        z = N.dot(fit.x)
        pb.path_info["direction_qp"] += 1
    else:
        pb.path_info["direction_linear"] += 1
    if (np.max(np.abs(E.dot(z)), initial=0.) > 1e-7
            or np.min(J.dot(z), initial=0.) < -1e-7):
        raise RuntimeError("MC1 directional cone residual exceeds tolerance")
    D = np.zeros_like(B)
    D[working] = z[:a*K].reshape(a, K)
    dt = np.zeros(p)
    dt[working] = z[a*K:]
    active = np.flatnonzero((t > tol) | (np.max(np.abs(D), axis=1) > tol) | (weights == 0))
    if not len(active):
        raise RuntimeError("MC1 directional program returned no moving rows")
    faces = []
    for j in active:
        if weights[j] == 0:
            faces.append([])
            continue
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


def _certificate_system(pb, grad, gdot, lam, upper, active, faces):
    """Assemble the certificate LP with COO triplets, not per-entry LIL rows.

    Variable order and inequality row order match the original formulation.
    The fixed stationarity block is cached only for the current path run.
    No pK-by-pK dense matrix is constructed.
    """
    C = pb.matrix[1]
    p, K = grad.shape
    weights = pb.weights
    nu, ns = len(C)*K, p*K
    start_s, start_t = 1+nu, 1+nu+ns
    total = 1+nu+2*ns
    cache = getattr(pb, '_mc1_certificate_cache', None)
    if cache is None:
        cache = pb._mc1_certificate_cache = {}
    if 'stationarity' not in cache:
        cache['stationarity'] = sparse.hstack([
            sparse.kron(sparse.csc_matrix(C.T), sparse.eye(K), format='csc'),
            sparse.eye(ns, format='csc'), sparse.csc_matrix((ns, ns))], format='csc')
    stationarity = sparse.hstack([
        sparse.csc_matrix(gdot.reshape(-1, 1)), cache['stationarity']], format='csc')

    active_mask = np.zeros(p, dtype=bool)
    active_mask[active] = True
    free = np.flatnonzero(weights == 0)
    selected = np.flatnonzero(active_mask & (weights > 0))
    inactive = np.flatnonzero(~active_mask & (weights > 0))
    bounds = np.empty((total, 2))
    bounds[:, 0], bounds[:, 1] = -np.inf, np.inf
    bounds[0] = 0., upper
    bounds[start_t:, 0] = 0.
    pinned = np.r_[selected, free]
    coordinates = (pinned[:, None]*K+np.arange(K)).ravel()
    bounds[start_s+coordinates] = 0.
    bounds[start_t+coordinates] = 0.

    # Only the small active face list is traversed in Python.
    entries = [(j, k, sign) for j, binding in zip(active, faces)
               if weights[j] > 0 for k, sign in binding]
    if entries:
        feature, klass, sign = np.asarray(entries, dtype=int).T
        position = start_s+feature*K+klass
        positive = sign > 0
        bounds[position[positive], 1] = np.inf
        bounds[position[~positive], 0] = -np.inf
    else:
        feature = klass = sign = np.empty(0, dtype=int)
    row_number = np.full(p, -1, dtype=int)
    row_number[selected] = np.arange(len(selected))
    face_matrix = sparse.coo_matrix((
        np.r_[weights[selected], sign],
        (np.r_[np.arange(len(selected)), row_number[feature]],
         np.r_[np.zeros(len(selected), dtype=int), start_s+feature*K+klass])),
        shape=(len(selected), total)).tocsc()
    A_eq = sparse.vstack([stationarity, face_matrix], format='csc')
    b_eq = np.r_[-grad.ravel(), lam*weights[selected]]

    if len(inactive):
        # For each inactive row: -S-T<=0, S-T<=0, sum(T)+delta*w<=lambda*w.
        q = len(inactive)
        coordinate = (inactive[:, None]*K+np.arange(K)).ravel()
        row = (np.arange(q)[:, None]*(2*K+1)+2*np.arange(K)).ravel()
        sum_row = np.arange(q)*(2*K+1)+2*K
        rows = np.r_[row, row, row+1, row+1, sum_row, np.repeat(sum_row, K)]
        columns = np.r_[start_s+coordinate, start_t+coordinate,
                        start_s+coordinate, start_t+coordinate,
                        np.zeros(q, dtype=int), start_t+coordinate]
        values = np.r_[-np.ones(q*K), -np.ones(q*K), np.ones(q*K),
                       -np.ones(q*K), weights[inactive], np.ones(q*K)]
        A_ub = sparse.coo_matrix((values, (rows, columns)),
                                 shape=(q*(2*K+1), total)).tocsc()
        b_ub = np.zeros(q*(2*K+1))
        b_ub[sum_row] = lam*weights[inactive]
    else:
        A_ub = b_ub = None
    objective = np.zeros(total)
    objective[0] = -1.
    return objective, A_eq, b_eq, A_ub, b_ub, bounds


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
    t[weights == 0] = 0.
    margins, rate = pb.targets*X.dot(B), pb.targets*X.dot(D)
    candidates = [(float(cap), "endpoint")]
    for i, k in np.ndindex(margins.shape):
        r = rate[i, k]
        if ((quadratic[i, k] and r > 0) or (not quadratic[i, k] and r < 0)):
            root = (1.-margins[i, k])/r
            if root > 20*tol:
                candidates.append((float(root), "sample_margin"))
    for j in active:
        if weights[j] == 0:
            continue
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
    # The primal event gives an upper bound. If its endpoint admits a dual
    # certificate, that bound is attained: no optimization LP is necessary.
    endpoint = (_endpoint_certificate(pb, grad+upper*gdot, lam-upper, active, faces, tol)
                if getattr(pb, "_mc1_fast", True) else None)
    if endpoint is not None:
        U, S = endpoint
        _, actual_grad = pb.loss_gradient(B+upper*D)
        residual = max(np.max(np.abs(actual_grad+C.T.dot(U)+S)),
                       np.max(np.abs(C.dot(B+upper*D)), initial=0.))
        if residual <= 5e-9*max(1., np.max(np.abs(actual_grad))):
            pb.path_info["certificate_linear"] += 1
            reasons = [reason for value, reason in candidates
                       if abs(value-upper) <= 50*tol*max(1., lam)]
            return upper, S, reasons, residual
    pb.path_info["certificate_lp"] += 1
    # Variables: delta, vec(U), vec(S), vec(abs(S)) (row-major).
    nu, ns = m*K, p*K
    start_s, start_t = 1+nu, 1+nu+ns
    objective, A_eq, b_eq, A_ub, b_ub, bounds = _certificate_system(
        pb, grad, gdot, lam, upper, active, faces)
    fit = linprog(objective, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq,
                  b_eq=b_eq, bounds=bounds, method="highs",
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


def solve_path_MC1(pb, lamin=.01, max_events=2000, tol=1e-8, n_active=0):
    """Trace MC1 knots down to lamin*lambda_max, without refitting a grid.

    Returns BETA and relative LAMBDAS as in c-lasso's solve_path. Diagnostics
    (events, working-set sizes and endpoint residuals) are stored on pb.path_info.
    A rank-deficient outgoing face or an unresolved zero step raises, rather
    than silently adding ridge or returning an incomplete path as successful.
    Linear-solve fast paths are checked before directional QPs and certificate
    LPs. Direction solves use working rows; certificates scan all feature rows.
    n_active=0 is unlimited. A positive value stops at the current knot before
    the next segment would exceed the feature-row budget. Free class biases
    do not count. path_info distinguishes this stop from reaching lamin.
    """
    if not np.isfinite(lamin) or not 0 <= lamin <= 1:
        raise ValueError("lamin must lie in [0,1]")
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be finite and positive")
    if not isinstance(n_active, (int, np.integer)) or n_active < 0:
        raise ValueError("n_active must be a nonnegative integer")
    if max_events < 1:
        raise ValueError("max_events must be positive")
    B, lam = pb.init.copy(), pb.lambdamax
    BETA, LAMBDAS = [B.copy()], [1.]
    info = dict(events=[], working_sizes=[], residuals=[], completed=False, stop_reason=None,
                direction_linear=0, direction_qp=0, certificate_linear=0, certificate_lp=0)
    pb.path_info = info
    pb._mc1_certificate_cache = {}
    if lam == 0 or lamin == 1:
        info["completed"] = True
        info["stop_reason"] = "lambda_target"
        if lamin < 1:
            BETA.append(B.copy())
            LAMBDAS.append(float(lamin))
        return BETA, LAMBDAS
    _, grad = pb.loss_gradient(B)
    certificate = -grad-pb.matrix[1].T.dot(pb.dual)
    target = lamin*pb.lambdamax
    for _ in range(max_events):
        D, dt, active, faces, quadratic, size = _outgoing_direction(pb, B, lam, certificate, tol)
        if n_active and np.count_nonzero(pb.weights[active] > 0) > n_active:
            info["stop_reason"] = "n_active"
            return BETA, LAMBDAS
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
            info["stop_reason"] = "lambda_target"
            return BETA, LAMBDAS
    raise RuntimeError("MC1 exceeded max_events before reaching the requested endpoint")
