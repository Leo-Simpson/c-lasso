"""Q1: constrained smoothed-quantile paths (absolute lambda).

Production dependencies: NumPy/SciPy. No implicit ridge. Intercept is unpenalized.
"""
import numpy as np
from scipy.optimize import minimize, linprog, brentq
from ._classification_path import direction, segment


def loss(r, tau, k):
    return np.where(r < -(1-tau)*k, (tau-1)*r-k*(1-tau)**2/2,
                    np.where(r > tau*k, tau*r-k*tau**2/2, r*r/(2*k)))


def certificate(Z, C, y, b, lam, tau, k, divisor, free, shortcut_tol=1e-9):
    g = -Z.T@np.clip((y-Z@b)/k, tau-1, tau)/divisor
    nz = np.flatnonzero(abs(b[free:]) > 1e-7)+free
    selected = np.r_[np.arange(free), nz]
    E=C[:,selected].T
    rhs_active=-g[selected]-lam*np.where(selected<free,0,np.sign(b[selected]))
    dual=np.linalg.lstsq(E,rhs_active,rcond=1e-12)[0] if len(C) else np.empty(0)
    adjusted=g+C.T@dual
    error=max(np.max(abs(adjusted[selected]+lam*np.where(selected<free,0,np.sign(b[selected]))),initial=0.),
              np.max(np.maximum(abs(adjusted[free:])-lam,0.),initial=0.),
              np.max(abs(C@b),initial=0.))
    if error<=shortcut_tol:return float(error),-adjusted
    if not len(C):return float(error),-adjusted
    if len(C)==1 and np.allclose(C[0,free:],C[0,free],rtol=1e-12,atol=0.) and C[0,free]!=0:
        # For zero-sum constraints the multiplier is a scalar shift. Intersect
        # inactive intervals and active singleton targets, then expand them by
        # the smallest common infinity error. No LP is needed, even off optimum.
        lower=float(np.max(-lam-g[free:]))
        upper=float(np.min(lam-g[free:]))
        if len(nz):
            targets=-g[nz]-lam*np.sign(b[nz])
            lower=max(lower,float(targets.max()))
            upper=min(upper,float(targets.min()))
        shift=(lower+upper)/2 if lower>upper else float(np.clip(0.,lower,upper))
        adjusted=g.copy();adjusted[free:]+=shift
        error=max(0.,(lower-upper)/2,np.max(abs(g[:free]),initial=0.),np.max(abs(C@b),initial=0.))
        return float(error),-adjusted
    # Minimize the maximum stationarity violation over equality multipliers.
    M = np.c_[C.T, -np.ones(len(b))]
    rows = [M[free:], np.c_[-C.T, -np.ones(len(b))][free:]]
    rhs = [lam-g[free:], lam+g[free:]]
    if len(selected):
        E = C[:,selected].T
        target = -g[selected]-lam*np.where(selected<free,0,np.sign(b[selected]))
        rows += [np.c_[E,-np.ones(len(selected))],np.c_[-E,-np.ones(len(selected))]]
        rhs += [target,-target]
    fit = linprog(np.r_[np.zeros(len(C)),1.],A_ub=np.vstack(rows),b_ub=np.concatenate(rhs),
                  bounds=[(None,None)]*len(C)+[(0,None)],method='highs',options={'primal_feasibility_tolerance':1e-10,'dual_feasibility_tolerance':1e-10})
    if not fit.success: raise RuntimeError(fit.message)
    error = max(float(fit.x[-1]), np.max(abs(C@b),initial=0.))
    return error, -g-C.T@fit.x[:-1]


def _lambda_max(g,C):
    """Small dual LP; closed form for absent or constant zero-sum rows."""
    if not len(C):return float(np.max(abs(g),initial=0.))
    if len(C)==1 and np.allclose(C[0],C[0,0],rtol=1e-12,atol=0.) and C[0,0]!=0:
        return float((g.max()-g.min())/2)
    M=np.vstack([np.c_[C.T,-np.ones(len(g))],np.c_[-C.T,-np.ones(len(g))]])
    fit=linprog(np.r_[np.zeros(len(C)),1.],A_ub=M,b_ub=np.r_[-g,g],
                bounds=[(None,None)]*len(C)+[(0,None)],method='highs',options={'primal_feasibility_tolerance':1e-10,'dual_feasibility_tolerance':1e-10})
    if not fit.success:raise RuntimeError('Q1 lambda_max LP: '+fit.message)
    return float(fit.x[-1])


def _flat_pivot(Z,C,y,b,lam,target,tau,k,divisor,free):
    """Descend with fixed b, or pivot on the current exact optimal face."""
    q=len(b);p=q-free;e=Z@b-y
    g=Z.T@np.clip(e/k,-tau,1-tau)/divisor
    active=abs(b[free:])>1e-9
    selected=np.r_[np.arange(free),np.flatnonzero(active)+free]
    rows=np.vstack([np.c_[C[:,free:].T,-np.ones(p)],np.c_[-C[:,free:].T,-np.ones(p)]])
    E=np.c_[C[:,selected].T,np.where(selected<free,0,np.sign(b[selected]))]
    fit=linprog(np.r_[np.zeros(len(C)),1.],A_ub=rows,b_ub=np.r_[-g[free:],g[free:]],
                A_eq=E if len(selected) else None,b_eq=-g[selected] if len(selected) else None,
                bounds=[(None,None)]*len(C)+[(target,lam)],method='highs',options={'primal_feasibility_tolerance':1e-10,'dual_feasibility_tolerance':1e-10})
    if not fit.success:raise RuntimeError('Q1 fixed-face lambda LP: '+fit.message)
    if lam-fit.x[-1]>1e-11*max(1.,lam):return b.copy(),float(fit.x[-1]),'flat-lambda'
    low=e<=-tau*k+1e-10;high=e>=(1-tau)*k-1e-10;curved=~(low|high)
    linear=(-tau*Z[low].sum(0)+(1-tau)*Z[high].sum(0))/divisor
    constant=(tau*y[low].sum()-low.sum()*k*tau**2/2-(1-tau)*y[high].sum()
              -high.sum()*k*(1-tau)**2/2+(e[curved]**2).sum()/(2*k))/divisor
    S=np.c_[np.zeros((p,free)),np.eye(p)]
    rows=[np.c_[S,-np.eye(p)],np.c_[-S,-np.eye(p)]];rhs=[np.zeros(p),np.zeros(p)]
    if low.any():rows.append(np.c_[Z[low],np.zeros((low.sum(),p))]);rhs.append(y[low]-tau*k)
    if high.any():rows.append(np.c_[-Z[high],np.zeros((high.sum(),p))]);rhs.append(-y[high]-(1-tau)*k)
    objective=np.r_[linear,np.full(p,lam)]
    value=loss(-e,tau,k).sum()/divisor+lam*abs(b[free:]).sum()
    eq=np.vstack([np.c_[C,np.zeros((len(C),p))],np.c_[Z[curved],np.zeros((curved.sum(),p))],objective])
    pivot=linprog(np.r_[np.zeros(q),-np.ones(p)],A_ub=np.vstack(rows),b_ub=np.concatenate(rhs),
                  A_eq=eq,b_eq=np.r_[np.zeros(len(C)),y[curved]+e[curved],value-constant],
                  bounds=[(None,None)]*q+[(0,None)]*p,method='highs',options={'primal_feasibility_tolerance':1e-10,'dual_feasibility_tolerance':1e-10})
    if not pivot.success:
        # Degenerate optimal faces can be reported infeasible by dual simplex
        # even when the incoming point satisfies all rows to machine precision.
        # A second numerical solve still must pass the actual Q1 certificate.
        pivot=linprog(np.r_[np.zeros(q),-np.ones(p)],A_ub=np.vstack(rows),b_ub=np.concatenate(rhs),
                      A_eq=eq,b_eq=np.r_[np.zeros(len(C)),y[curved]+e[curved],value-constant],
                      bounds=[(None,None)]*q+[(0,None)]*p,method='highs',
                      options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
    if not pivot.success:raise RuntimeError('Q1 optimal-face LP: '+pivot.message)
    candidate=pivot.x[:q]
    # Polish the LP face in original Q1 units. Nearly zero basic variables
    # otherwise inherit simplex feasibility tolerances and can select the
    # wrong outgoing coefficient sign at the next event.
    from ._classification_path import _nullspace, _solve_design
    keep=(abs(candidate)>1e-7)|(np.arange(q)<free)
    A=Z[:,keep];CC=C[:,keep]
    polished=np.zeros_like(candidate);polished[keep]=candidate[keep]
    if len(C):
        polished[keep]-=np.linalg.lstsq(CC,C@polished,rcond=1e-12)[0]
    N=_nullspace(CC)
    if N.shape[1]:
        rr=y-Z@polished
        gradient=-A.T@np.clip(rr/k,tau-1,tau)/divisor
        signs=np.sign(polished[keep]);signs[np.flatnonzero(keep)<free]=0.
        try:
            correction=_solve_design(A[curved]@N/np.sqrt(2*k*divisor),-N.T@(gradient+lam*signs))
            polished[keep]+=N@correction
        except np.linalg.LinAlgError:pass
    error,_=certificate(Z,C,y,polished,lam,tau,k,divisor,free)
    if error<=1e-8:candidate=polished
    if np.max(abs(candidate-b),initial=0.)<1e-9:raise RuntimeError('Q1 optimal-face pivot made no progress')
    return candidate,lam,'flat-pivot'


def quantile_path(X, y, lambdas, tau=.5, k=1., C=None, intercept=True,
                  divisor=None, method='events', tol=2e-6, max_events=10000,
                  n_active=0, dr_max_iter=100000, store_segments=False, pds_max_iter=500000):
    """Return coefficients [intercept, slopes], KKT errors, and diagnostics.

    Minimize sum rho(y-intercept-X beta)/divisor + lambda ||beta||_1,
    subject to C beta=0. C defaults to the zero-sum row. Divisor defaults n.
    Methods: 'events', 'DR', 'P-PDS', 'PF-PDS', or 'warm' (SLSQP).
    Unresolved event faces and splitting iteration limits raise explicitly.
    """
    X,y,lams = np.asarray(X,float),np.asarray(y,float),np.asarray(lambdas,float)
    if X.ndim!=2 or y.shape!=(len(X),) or not len(y) or not X.shape[1]: raise ValueError('invalid data shape')
    if not np.isfinite(X).all() or not np.isfinite(y).all(): raise ValueError('nonfinite data')
    if not np.isfinite([tau,k]).all() or not 0<tau<1 or k<=0: raise ValueError('require 0<tau<1 and k>0')
    if lams.ndim!=1 or not len(lams) or not np.isfinite(lams).all() or np.any(lams<0) or np.any(np.diff(lams)>0): raise ValueError('descending nonnegative lambdas required')
    if method not in ('warm','events','DR','P-PDS','PF-PDS'): raise ValueError('unknown method')
    if isinstance(n_active,bool): n_active=int(n_active)
    if not isinstance(n_active,(int,np.integer)) or n_active<0: raise ValueError('n_active must be a nonnegative integer')
    if n_active and method!='events': raise ValueError('support stopping requires Path-Alg')
    divisor = len(y) if divisor is None else float(divisor)
    if not np.isfinite(divisor) or divisor<=0: raise ValueError('positive divisor required')
    p,free = X.shape[1],int(intercept)
    C = np.ones((1,p)) if C is None else np.asarray(C,float)
    if C.ndim!=2 or C.shape[1]!=p or not np.isfinite(C).all(): raise ValueError('invalid constraints')
    # Retain the same feasible subspace, removing redundant rows.
    if len(C):
        _,s,V = np.linalg.svd(C,full_matrices=False)
        C = V[s>1e-12*max(1.,s.max())]
    Z = np.c_[np.ones(len(y)),X] if free else X
    C = np.c_[np.zeros((len(C),free)),C]
    q = Z.shape[1]; b = np.zeros(q)
    if free:
        b[0] = brentq(lambda a:np.clip((y-a)/k,tau-1,tau).sum(),y.min()-k,y.max()+k,xtol=1e-14)
    def solve(lam,b):
        # Epigraph matrices are allocated only for the reference warm solver.
        selector = np.c_[np.zeros((p,free)),np.eye(p)]
        ineq = np.block([[selector,-np.eye(p)],[-selector,-np.eye(p)]])
        eq = np.c_[C,np.zeros((len(C),p))]
        cons = [dict(type='ineq',fun=lambda v:-ineq@v,jac=lambda v:-ineq)]
        if len(C): cons += [dict(type='eq',fun=lambda v:eq@v,jac=lambda v:eq)]
        def obj(v):
            r = y-Z@v[:q]
            return loss(r,tau,k).sum()/divisor+lam*v[q:].sum(), np.r_[-Z.T@np.clip(r/k,tau-1,tau)/divisor,np.full(p,lam)]
        fit = minimize(obj,np.r_[b,abs(b[free:])],jac=True,method='SLSQP',constraints=cons,
                       options=dict(ftol=1e-13,maxiter=3000))
        candidate = fit.x[:q]
        err,_ = certificate(Z,C,y,candidate,lam,tau,k,divisor,free)
        if err>tol: raise RuntimeError(f'warm solve failed KKT: {err:g}; {fit.message}')
        return candidate,fit.nit
    output=[]; errors=[]; events=[]; iterations=0; evaluated=[]; stop_reason='endpoint'
    score=-Z.T@np.clip((y-Z@b)/k,tau-1,tau)/divisor
    maximum=_lambda_max(score[free:],C[:,free:])
    current=maximum
    certify=lambda b,l:certificate(Z,C,y,b,l,tau,k,divisor,free, min(tol,1e-8) if method in ('DR','P-PDS','PF-PDS') else 1e-9)
    dr=None
    paired=None
    def record(value):
        err,_=certify(b,value)
        if err>tol:raise RuntimeError(f'endpoint KKT {err:g}')
        output.append(b.copy());errors.append(err);evaluated.append(float(value))
    for target in lams:
        if target>=maximum:
            record(target)
            continue
        if method=='warm':
            b,it=solve(target,b);iterations+=it
        elif method in ('DR','P-PDS','PF-PDS'):
            if dr is None:
                from ._q1_dr import Q1DR, Q1PDS
                dr=(Q1DR(Z,C,y,tau,k,divisor,free) if method=='DR'
                    else Q1PDS(Z,C,y,tau,k,divisor,free,method))
            limit=dr_max_iter if method=='DR' else pds_max_iter
            b,it=dr.solve(target,b,certify,min(tol,1e-8),limit);iterations+=it
        else:
            while current-target>1e-12*max(1.,current):
                err,signed=certify(b,current)
                if err>tol:raise RuntimeError('uncertified event start')
                start_lambda=current;start_b=b.copy()
                try:
                    from ._q1_path import curved_direction, next_event
                    try:
                        D,active,region=curved_direction(Z,C,y,b,current,signed,tau,k,divisor,free)
                        delta=next_event(Z,C,y,b,current,D,active,region,current-target,tau,k,divisor,free)
                    except (RuntimeError,np.linalg.LinAlgError):
                        if paired is None:
                            scale=np.sqrt(2*k*divisor)
                            XX=np.vstack([-Z,Z])/scale
                            offsets=np.r_[1+y/scale,1-y/scale]
                            lower=1-np.r_[np.full(len(y),(1-tau)*k),np.full(len(y),tau*k)]/scale
                            paired=(XX,lower,offsets)
                        XX,lower,offsets=paired
                        D,active,quad=direction(XX,C,np.ones(2*len(y)),b,current,signed,free,lower,offsets)
                        delta=segment(XX,C,np.ones(2*len(y)),b,current,D,active,quad,current-target,free,lower,offsets)
                    # Tied entries may exceed the budget at a single knot.
                    # Stop on the incoming side instead of violating the bound.
                    outgoing=np.count_nonzero((abs(b[free:])>1e-9)|(abs(D[free:])>1e-9))
                    if n_active and outgoing>n_active:
                        stop_reason='n_active';break
                    old=b.copy();b=b+delta*D
                    exits=(np.arange(q)>=free)&(old*D<0)&(abs(old+delta*D)<1e-10)
                    b[exits]=0.;current-=delta;kind='segment'
                except (RuntimeError,np.linalg.LinAlgError):
                    b,new,kind=_flat_pivot(Z,C,y,b,current,target,tau,k,divisor,free)
                    delta=current-new;current=new
                    if n_active and np.count_nonzero(abs(b[free:])>1e-9)>n_active:
                        raise RuntimeError('flat optimal-face pivot exceeds support budget')
                event=dict(lam=float(current),delta=float(delta),kind=kind)
                if store_segments:
                    event.update(start_lambda=start_lambda,coefficients=start_b,
                                 direction=D.copy() if kind=='segment' else np.zeros_like(b))
                events.append(event)
                if len(events)>max_events:raise RuntimeError('event limit')
                if n_active and np.count_nonzero(abs(b[free:])>1e-9)>=n_active:
                    stop_reason='n_active';break
            if stop_reason=='n_active':
                record(current);break
        current=float(target)
        record(target)
    return dict(coefficients=np.array(output),kkt=np.array(errors),events=events,
                iterations=iterations,lambdamax=maximum,evaluated_lambdas=np.array(evaluated),
                stop_reason=stop_reason)


# Public formulation name, retaining the descriptive low-level function name.
Q1_path = quantile_path


class problem_Q1:
    """Q1 adapter with the same absolute/relative lambda convention as MC1.

    Path-Alg traces knots; DR and PDS reuse R1 splitting iterations.
    SLSQP remains available as a small-problem validation solver.
    """
    def __init__(self, matrix, algo='SLSQP', tau=.5, k=1., divisor=None,
                 w=None, intercept=False):
        from .misc_functions import _constraint_rows
        if algo not in ('SLSQP', 'Path-Alg', 'DR', 'P-PDS', 'PF-PDS'):
            raise ValueError("Q1 supports Path-Alg, DR, P-PDS, PF-PDS, or SLSQP")
        self.method = {'Path-Alg':'events','SLSQP':'warm','DR':'DR','P-PDS':'P-PDS','PF-PDS':'PF-PDS'}[algo]
        X,C,y=matrix
        X,y=np.asarray(X,float),np.asarray(y,float)
        if X.ndim!=2 or y.shape!=(len(X),) or not len(y) or not X.shape[1]:
            raise ValueError('invalid data shape')
        if not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError('nonfinite data')
        if not np.isfinite([tau,k]).all() or not 0<tau<1 or k<=0:
            raise ValueError('require 0<tau<1 and k>0')
        self.divisor=len(y) if divisor is None else float(divisor)
        if not np.isfinite(self.divisor) or self.divisor<=0:
            raise ValueError('positive divisor required')
        C=_constraint_rows(C,X.shape[1])
        self.w=np.ones(X.shape[1]) if w is None else np.asarray(w,float)
        if self.w.shape!=(X.shape[1],) or not np.isfinite(self.w).all() or np.any(self.w<=0):
            raise ValueError('Q1 weights must be finite and positive')
        self.X,self.C,self.y=X/self.w,C/self.w,y
        self.tau,self.k,self.intercept=tau,k,bool(intercept)
        a=brentq(lambda a:np.clip((y-a)/k,tau-1,tau).sum(),y.min()-k,y.max()+k) if intercept else 0.
        g=-self.X.T@np.clip((y-a)/k,tau-1,tau)/self.divisor
        self.lambdamax=_lambda_max(g,self.C)


def pathlasso_Q1(pb, path, n_active=False):
    """Solve a relative-lambda Q1 grid; coefficients in original units."""
    result=Q1_path(pb.X,pb.y,np.asarray(path,float)*pb.lambdamax,
                   pb.tau,pb.k,pb.C,pb.intercept,pb.divisor,pb.method,n_active=n_active)
    weights=np.r_[1.,pb.w] if pb.intercept else pb.w
    pb.path_info=result
    return result['coefficients']/weights


def Classo_Q1(pb, lam):
    return pathlasso_Q1(pb,[lam])[0]
