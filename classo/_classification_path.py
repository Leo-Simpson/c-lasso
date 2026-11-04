"""Private scalar C1/C2 pivots for degenerate coefficient and sample events.

The usual active linear solve remains in path_alg. This module selects a
feasible outgoing direction only when that solve gives an incompatible sign
at zero or coupled hinge-boundary choices cycle. NumPy/SciPy only; no added
penalty and no change to the public formulation or its certificate tolerance.
"""
import numpy as np
from scipy.optimize import linprog, minimize, nnls


def _nullspace(C, tol=1e-10):
    if not len(C):
        return np.eye(C.shape[1])
    _, singular, vh = np.linalg.svd(C, full_matrices=True)
    rank = np.sum(singular > tol*max(1., singular[0]))
    return vh[rank:].T


def _solve(H, rhs, reference=None):
    """Solve a PSD reduced system, preserving a certified null-space choice."""
    H = (H+H.T)/2
    values, vectors = np.linalg.eigh(H)
    cutoff = 1e-10*max(np.finfo(float).tiny, values[-1])
    flat = values <= cutoff
    if np.any(values < -cutoff) or np.max(abs(vectors[:, flat].T@rhs), initial=0.) > 1e-8:
        raise np.linalg.LinAlgError('Classification direction requires another outgoing face')
    keep = ~flat
    result = vectors[:, keep]@((vectors[:, keep].T@rhs)/values[keep])
    if reference is not None:
        result += vectors[:, flat]@(vectors[:, flat].T@reference)
    scale = max(1., np.linalg.norm(H,np.inf)*np.linalg.norm(result,np.inf),
                np.linalg.norm(rhs,np.inf))
    allowance = max(1e-10*(1+np.linalg.norm(rhs,np.inf)),32*np.finfo(float).eps*scale)
    if not np.isfinite(result).all() or np.max(abs(H@result-rhs),initial=0.) > allowance:
        raise np.linalg.LinAlgError('Classification reduced direction failed its backward-error check')
    return result


def _gradient(X,y,beta,lower=None):
    hinge = np.maximum(1-y*(X@beta),0.)
    if lower is not None: hinge = np.minimum(hinge,1-lower)
    return -2*X.T@(y*hinge)


def direction(X,C,y,beta,lam,certificate,free=0,lower=None):
    """Descending-lambda direction in the scalar L1 critical cone."""
    tol = 1e-9
    nonzero = abs(beta)>tol
    tight = abs(certificate)>=lam-20*tol*max(1.,lam)
    working = np.flatnonzero(nonzero|tight|(np.arange(len(beta))<free))
    A = X[:,working]
    signs = np.where(nonzero[working],np.sign(beta[working]),np.sign(certificate[working]))
    signs[working<free] = 0.
    zero = ~nonzero[working] & (working>=free)
    N = _nullspace(C[:,working])
    if not N.shape[1]:
        return np.zeros_like(beta),np.flatnonzero(nonzero|(np.arange(len(beta))<free)),y*(X@beta)<1.
    J = signs[zero,None]*N[zero]
    J = J[np.linalg.norm(J,axis=1)>1e-10]
    margin = y*(X@beta)
    interior = margin < 1.-2e-11
    if lower is not None: interior &= margin > lower+2e-11
    lower_boundary = abs(margin-lower)<=2e-11 if lower is not None else np.zeros(len(y),bool)
    boundary = abs(margin-1.)<=2e-11

    def objective(v):
        D = N@v
        scores = A@D
        r = np.where(interior,scores,0.)+np.where(boundary,y*np.minimum(y*scores,0.),0.)
        r += np.where(lower_boundary,y*np.maximum(y*scores,0.),0.)
        return r@r-signs@D, N.T@(2*A.T@r-signs)

    region = interior|boundary|lower_boundary
    seen = set()
    candidate = None
    for _ in range(8):
        if region.tobytes() in seen: break
        seen.add(region.tobytes())
        Z = A[region]@N
        try: v = _solve(2*Z.T@Z,N.T@signs)
        except np.linalg.LinAlgError: break
        if (np.max(abs(objective(v)[1]),initial=0.)<=1e-8
                and np.min(J@v,initial=0.)>=-1e-9):
            candidate = N@v
            break
        rate = y*(A@(N@v))
        region = interior|(boundary&(rate<0))|(lower_boundary&(rate>0))
    if candidate is None:
        constraints = [dict(type='ineq',fun=lambda v:J@v,jac=lambda v:J)] if len(J) else []
        fit = minimize(objective,np.zeros(N.shape[1]),jac=True,method='SLSQP',
                       constraints=constraints,options=dict(ftol=1e-12,maxiter=2000))
        gradient = objective(fit.x)[1]
        slack = J@fit.x
        if len(J):
            multipliers,_ = nnls(J.T,gradient,maxiter=10000)
            residual = gradient-J.T@multipliers
            complementarity = np.max(abs(multipliers*slack),initial=0.)
        else: residual,complementarity = gradient,0.
        if (np.min(slack,initial=0.) < -1e-8 or
                (not fit.success and (np.max(abs(residual),initial=0.)>1e-7 or complementarity>1e-7))):
            raise RuntimeError('Classification outgoing cone QP failed its KKT check: '+fit.message)
        candidate = N@fit.x

    D = np.zeros_like(beta);D[working] = candidate
    active = np.flatnonzero(nonzero|(abs(D)>tol)|(np.arange(len(beta))<free))
    rate = y*(X@D)
    quadratic = interior|(boundary&(rate<0))|(lower_boundary&(rate>0))
    seen = set()
    for _ in range(len(active)+12):
        key=(quadratic.tobytes(),active.tobytes())
        if key in seen:
            raise RuntimeError('Classification outgoing sample regions cycled')
        seen.add(key)
        Z = X[:,active]
        tangent = _nullspace(C[:,active])
        active_signs = np.where(nonzero[active],np.sign(beta[active]),np.sign(certificate[active]))
        active_signs[active<free] = 0.
        if tangent.shape[1]:
            H = 2*(Z[quadratic]@tangent).T@(Z[quadratic]@tangent)
            v = _solve(H,tangent.T@active_signs,tangent.T@D[active])
            refined = np.zeros_like(beta);refined[active] = tangent@v
        else: refined = np.zeros_like(beta)
        # A numerically moving tight zero row can refine to a direction
        # outside its permitted sign. Release that face and solve again;
        # never convert the small QP error into an opposite-sign entry.
        incompatible=(~nonzero[active]) & (active>=free) & (active_signs*refined[active] < -tol)
        if np.any(incompatible):
            active=active[~incompatible]
            continue
        rate = y*(X@refined)
        arithmetic = max(1.,np.linalg.norm(X,np.inf)*np.max(abs(refined),initial=0.))
        threshold = max(1e-12,64*np.finfo(float).eps*arithmetic)
        bad = boundary&np.where(quadratic,rate>threshold,rate<-threshold)
        bad |= lower_boundary&np.where(quadratic,rate<-threshold,rate>threshold)
        if not np.any(bad): break
        quadratic[bad] = (boundary[bad]&(rate[bad]<0))|(lower_boundary[bad]&(rate[bad]>0))
    else: raise RuntimeError('Classification outgoing sample regions did not converge')
    zero_working = ~nonzero[working]&(working>=free)
    if (np.min(signs[zero_working]*refined[working[zero_working]],initial=0.)<-1e-7
            or np.max(abs(C@refined),initial=0.)>1e-7):
        raise RuntimeError('Classification direction violates its critical cone')
    active = np.flatnonzero(nonzero|(abs(refined)>tol)|(np.arange(len(beta))<free))
    return refined,active,quadratic


def _certificate(C,gradient,lam,active,signs,free):
    selected = np.r_[np.arange(free),active[active>=free]]
    rhs = -gradient[selected]-lam*signs[selected]
    dual = np.linalg.lstsq(C[:,selected].T,rhs,rcond=None)[0] if len(C) else np.empty(0)
    adjusted = gradient+C.T@dual
    residual = max(np.max(abs(adjusted[selected]+lam*signs[selected]),initial=0.),
                   np.max(np.maximum(abs(adjusted[free:])-lam,0.),initial=0.))
    return dual,residual


def segment(X,C,y,beta,lam,D,active,quadratic,cap,free=0,lower=None):
    """Largest dual-certified interval, including nearby and simultaneous events."""
    eps = 64*np.finfo(float).eps*max(1.,lam)
    margin,rate = y*(X@beta),y*(X@D)
    candidates = [cap]
    crossing = ((quadratic&(rate>0))|(~quadratic&(rate<0)))&(rate!=0)
    roots = (1-margin[crossing])/rate[crossing]
    candidates.extend(roots[roots>eps])
    if lower is not None:
        roots = (lower-margin[rate!=0])/rate[rate!=0]
        candidates.extend(roots[roots>eps])
    exits = (np.arange(len(beta))>=free)&(abs(beta)>1e-9)&(beta*D<0)
    candidates.extend((-beta[exits]/D[exits]).tolist())
    upper = float(min(candidates))
    if upper<=eps: raise RuntimeError('Classification has no positive certified event interval')
    signs = np.sign(beta)
    zero = abs(beta)<=1e-9
    signs[zero] = np.sign(D[zero]);signs[:free] = 0.
    gradient = _gradient(X,y,beta,lower)
    gdot = 2*X.T@(quadratic*(X@D))
    dual,residual = _certificate(C,gradient+upper*gdot,lam-upper,active,signs,free)
    allowance = 5e-9*max(1.,lam,np.max(abs(gradient),initial=0.))
    if residual<=allowance:
        step = upper
    else:
        # Variables: unit interval fraction, equality multipliers. Eliminating
        # the scalar L1 subgradient leaves only m+1 LP variables.
        penalized = np.arange(free,len(beta))
        A = np.vstack([np.c_[upper*(gdot[penalized]+1),C[:,penalized].T],
                       np.c_[upper*(1-gdot[penalized]),-C[:,penalized].T]])
        b = np.r_[lam-gradient[penalized],lam+gradient[penalized]]
        selected = np.r_[np.arange(free),active[active>=free]]
        E = np.c_[upper*(gdot[selected]-signs[selected]),C[:,selected].T]
        rhs = -gradient[selected]-lam*signs[selected]
        fit = linprog(np.r_[-1.,np.zeros(len(C))],A_ub=A,b_ub=b,A_eq=E,b_eq=rhs,
                      bounds=[(0.,1.)]+[(None,None)]*len(C),method='highs',
                      options=dict(primal_feasibility_tolerance=1e-9,dual_feasibility_tolerance=1e-9))
        if not fit.success: raise RuntimeError('Classification segment certificate LP failed: '+fit.message)
        step = upper*fit.x[0];dual = fit.x[1:]
    if step<=eps: raise RuntimeError('Classification zero-length segment; unresolved outgoing face')
    new_beta = beta+step*D
    actual = _gradient(X,y,new_beta,lower)+C.T@dual
    residual = max(np.max(np.maximum(abs(actual[free:])-(lam-step),0.),initial=0.),
                   np.max(abs(actual[active]+(lam-step)*signs[active]),initial=0.),
                   np.max(abs(C@new_beta),initial=0.))
    if residual>5e-7*max(1.,np.max(abs(gradient),initial=0.)):
        raise RuntimeError('Classification endpoint KKT residual exceeds tolerance')
    return step


def step(param):
    """Take one scalar classification cone pivot, then restore the legacy path state."""
    from .path_alg import check_C1_certificate,_C1_inverse,next_idr1
    A,C,y = param.A,param.C,param.y
    lower = param.rho if param.formulation == "C2" else None
    free = int(param.intercept)
    X = np.c_[np.ones(len(A)),A] if free else A
    constraints = np.c_[np.zeros((len(C),free)),C]
    beta = np.r_[param.beta0,param.beta] if free else param.beta.copy()
    certificate = check_C1_certificate(param)
    gradient = _gradient(X,y,beta,lower)
    signed = -gradient-constraints.T@certificate['dual']
    absolute = param.lam*param.lambdamax
    D,active,quadratic = direction(X,constraints,y,beta,absolute,signed,free,lower)
    delta = segment(X,constraints,y,beta,absolute,D,active,quadratic,
                    absolute-param.lamin*param.lambdamax,free,lower)
    beta += delta*D
    param.beta = beta[free:].copy()
    if free: param.beta0 = float(beta[0])
    target = param.lamin*param.lambdamax
    remaining = absolute-delta
    param.lam = param.lamin if remaining-target<=1e-10*max(1.,param.lambdamax) else remaining/param.lambdamax
    param.r = y*(A@param.beta+param.beta0)
    param.F = param.r<=1.+1e-10
    if lower is not None: param.F &= param.r>=lower-1e-10
    P = A[param.F]
    if free and len(P): P=P-P.mean(axis=0)
    H = 2*P.T@P
    param.M = np.block([[H,C.T],[C,np.zeros((len(C),len(C)))]])
    cert = check_C1_certificate(param)
    adjusted = _gradient(X,y,beta,lower)[free:]+C.T@cert['dual']
    absolute = param.lam*param.lambdamax
    param.s = -adjusted/absolute if absolute else np.sign(param.beta)
    activity = abs(param.beta)>1e-9
    if absolute: activity |= abs(param.s)>=1.-1e-9
    param.activity = activity.tolist();param.number_act = int(activity.sum())
    param.idr = [False]*len(C)
    for _ in range(len(C)):
        row = next_idr1(param.idr,C[:,activity])
        if type(row) is not int and not isinstance(row,np.integer): break
        param.idr[int(row)] = True
    selection = param.activity+param.idr
    param.Xt = _C1_inverse(param.M[selection,:][:,selection])
