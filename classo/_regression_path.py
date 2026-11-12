"""R1/R2 adapters to the shared scalar polyhedral path primitives."""
import numpy as np
from scipy.optimize import linprog
from ._classification_path import direction, segment


def _flat_step(param):
    """Huber optimal-face pivot, used only when a curved segment cannot move."""
    from .path_alg import check_C1_certificate, _refresh_classification_state
    A, C, y = param.A, param.C, param.y
    rho = param.rho
    n, d = A.shape
    k, free = len(C), int(param.intercept)
    Z = np.c_[np.ones(n), A] if free else A
    beta = np.r_[param.beta0, param.beta] if free else param.beta.copy()
    r = Z@beta-y
    gradient = 2*A.T@np.clip(r, -rho, rho)
    absolute, target = param.lam*param.lambdamax, param.lamin*param.lambdamax
    active = abs(param.beta)>1e-9
    inequalities = np.vstack([np.c_[C.T,-np.ones(d)], np.c_[-C.T,-np.ones(d)]])
    equalities = np.c_[C[:,active].T, np.sign(param.beta[active])]
    fit = linprog(np.r_[np.zeros(k),1.], A_ub=inequalities,
                  b_ub=np.r_[-gradient,gradient],
                  A_eq=equalities if active.any() else None,
                  b_eq=-gradient[active] if active.any() else None,
                  bounds=[(None,None)]*k+[(0.,absolute)], method='highs')
    if not fit.success:
        raise RuntimeError('R2 flat-state LP failed: '+fit.message)
    lower = max(target,float(fit.x[-1]))
    if lower < absolute-1e-10*max(1.,absolute) or absolute == 0:
        param.lam = param.lamin if lower == target else lower/param.lambdamax
        _refresh_classification_state(param)
        return
    low, high = r <= -rho+1e-10, r >= rho-1e-10
    curved = ~(low|high)
    linear = 2*rho*(Z[high].sum(0)-Z[low].sum(0))
    constant = (2*rho*(y[low].sum()-y[high].sum())
                -rho*rho*(low.sum()+high.sum())+(r[curved]**2).sum())
    loss = np.where(abs(r)<=rho,r*r,2*rho*abs(r)-rho*rho).sum()
    select = np.c_[np.zeros((d,free)),np.eye(d)]
    rows = [np.c_[select,-np.eye(d)],np.c_[-select,-np.eye(d)]]
    rhs = [np.zeros(d),np.zeros(d)]
    if low.any():
        rows.append(np.c_[Z[low],np.zeros((low.sum(),d))]);rhs.append(y[low]-rho)
    if high.any():
        rows.append(np.c_[-Z[high],np.zeros((high.sum(),d))]);rhs.append(-y[high]-rho)
    tree = np.c_[np.zeros((k,free)),C,np.zeros((k,d))]
    fixed = np.c_[Z[curved],np.zeros((curved.sum(),d))]
    optimality = np.r_[linear,np.full(d,absolute)]
    value = loss+absolute*abs(param.beta).sum()
    pivot = linprog(np.r_[np.zeros(Z.shape[1]),-np.ones(d)],
                    A_ub=np.vstack(rows),b_ub=np.concatenate(rhs),
                    A_eq=np.vstack([tree,fixed,optimality]),
                    b_eq=np.r_[np.zeros(k),y[curved]+r[curved],value-constant],
                    bounds=[(None,None)]*Z.shape[1]+[(0.,None)]*d,method='highs')
    if not pivot.success:
        raise RuntimeError('R2 optimal-face LP failed: '+pivot.message)
    candidate = pivot.x[:Z.shape[1]]
    if np.max(abs(candidate-beta),initial=0.)<=1e-9:
        raise RuntimeError('R2 flat-face pivot made no progress')
    param.beta = candidate[free:].copy()
    if free: param.beta0 = float(candidate[0])
    _refresh_classification_state(param)
    check_C1_certificate(param)


def step(param):
    """Take a certified scalar regression segment, without numerical ridge.

    Squared loss is (r_+)^2+((-r)_+)^2. Huber loss is the same pair
    with each one-sided quadratic capped into its linear tail at rho.
    Offsets therefore reuse the classification cone/event primitives exactly;
    coefficients and equality constraints remain in the original dimension.
    """
    from .path_alg import check_C1_certificate, _refresh_classification_state
    A, C, y = param.A, param.C, param.y
    free = int(param.intercept)
    Z = np.c_[np.ones(len(y)),A] if free else A
    beta = np.r_[param.beta0,param.beta] if free else param.beta.copy()
    constraints = np.c_[np.zeros((len(C),free)),C]
    clipped = np.clip(Z@beta-y,-param.rho,param.rho) if param.formulation=='R2' else Z@beta-y
    gradient = 2*Z.T@clipped
    cert = check_C1_certificate(param)
    signed = -gradient-constraints.T@cert['dual']
    X = np.vstack([Z,-Z])
    targets = np.ones(2*len(y))
    offset = np.r_[1-y,1+y]
    lower = 1-param.rho if param.formulation=='R2' else None
    absolute = param.lam*param.lambdamax
    if param.formulation == 'R2' and not np.any(param.F):
        return _flat_step(param)
    try:
        D,active,quadratic = direction(X,constraints,targets,beta,absolute,signed,
                                       free,lower,offset)
        delta = segment(X,constraints,targets,beta,absolute,D,active,quadratic,
                        absolute-param.lamin*param.lambdamax,free,lower,offset)
    except (np.linalg.LinAlgError,RuntimeError):
        if param.formulation != 'R2': raise
        return _flat_step(param)
    beta += delta*D
    param.beta = beta[free:].copy()
    if free: param.beta0 = float(beta[0])
    target = param.lamin*param.lambdamax
    remaining = absolute-delta
    param.lam = param.lamin if remaining-target<=1e-10*max(1.,param.lambdamax) else remaining/param.lambdamax
    _refresh_classification_state(param)
