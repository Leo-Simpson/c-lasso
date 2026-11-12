"""Concomitant ridge through quadratic paths and the Huber mean-shift identity.

Ridge belongs inside the perspective: (||r-u||²+ridge/2*||beta||²)/sigma.
R3 reuses one fixed quadratic homotopy, reparameterized by lambda=t/sigma.
R4 uses the same homotopy for each mean-shift reduction, as the original R4
solver does. Identity columns for mean shifts are sparse, not dense n-by-n.
No conic package is imported or required by this module.
"""
import numpy as np
from scipy import sparse
from scipy.optimize import brentq, least_squares, minimize
from ._homotopy_geometry import initialize_path


def _quadratic_path(X,C,y,w,diagonal,intercept,e,target,budget=0,budget_features=None):
    from .elastic_net import problem_elastic_net
    from ._quadratic_homotopy import solve_path_quadratic
    pb=problem_elastic_net((X,C,y),typ='R1',w=w,intercept=intercept)
    pb.ridge_diagonal=np.r_[np.zeros(pb.feature_offset),diagonal]
    if budget_features is not None:
        pb._budget_rows=np.r_[np.zeros(pb.feature_offset,bool),
                              np.arange(X.shape[1])<budget_features]
    def scale(B):
        return np.sqrt(max(0.,pb.loss_gradient(B)[0])/e)
    def mapped(B,t):
        sigma=scale(B)
        return t/sigma if sigma else 0.
    if target>=mapped(pb.init,pb.lambdamax) or scale(pb.init)==0:
        return pb,[pb.init.copy()],np.array([pb.lambdamax]),scale
    # Between quadratic knots, solve the scalar norm equation exactly. This
    # caps the final segment without tracing a dense tail beyond the request.
    def endpoint(B,t,D,cap):
        if target==0: return cap
        def equation(delta):
            return t-delta-target*scale(B+delta*D)
        return brentq(equation,0.,cap,xtol=1e-12*max(1.,t),rtol=1e-13)
    pb._path_endpoint=endpoint
    pb._path_reached=lambda B,t: mapped(B,t)<=target+1e-9*max(1.,target)
    B,L=solve_path_quadratic(pb,lamin=0.,n_active=budget,tol=1e-9,max_events=20000)
    return pb,B,np.asarray(L)*pb.lambdamax,scale


def _evaluate(B,T,scale,target):
    mapped=np.array([t/scale(b) if scale(b) else 0. for b,t in zip(B,T)])
    if target>=mapped[0]: return B[0].copy(),scale(B[0])
    if target<=mapped[-1]+1e-10*max(1.,target): return B[-1].copy(),scale(B[-1])
    i=min(np.searchsorted(-mapped,-target,side='right')-1,len(B)-2)
    delta=B[i+1]-B[i]
    def equation(fraction):
        return (1-fraction)*T[i]+fraction*T[i+1]-target*scale(B[i]+fraction*delta)
    fraction=brentq(equation,0.,1.,xtol=1e-13)
    beta=B[i]+fraction*delta
    return beta,scale(beta)


def _r4_baseline(X,C,y,w,intercept,e,rho,ridge):
    """Bias/scale optimum at beta=0 and its true constrained lambda_max."""
    n=len(y);bias=float(np.median(y)) if intercept else 0.
    residual=bias-y
    nonzero=residual!=0
    q=2*rho*np.sign(residual)
    zeros=np.flatnonzero(~nonzero)
    if intercept and len(zeros): q[zeros]=-q.sum()/len(zeros)
    if q.dot(q)/4 <= e+1e-12*max(1.,e):
        # Closed scale-zero case. Ridge changes the subgradient set at the
        # joint origin, so minimize lambda over ALL compatible certificates.
        initial=initialize_path(X/w,-q[:,None],C/w)
        m=len(C);z=len(zeros)
        start=np.r_[q[zeros],initial['dual'][:,0],initial['lambda_max']]
        def unpack(v):
            response=q.copy();response[zeros]=v[:z]
            g=X.T.dot(response)+C.T.dot(v[z:z+m])
            soft=np.sign(g)*np.maximum(abs(g)-v[-1]*w,0.)
            return response,soft
        def feasible(v):
            response,soft=unpack(v)
            return e-response.dot(response)/4-soft.dot(soft)/(2*ridge)
        def jac(v):
            response,soft=unpack(v)
            return np.r_[-response[zeros]/2-X[zeros].dot(soft)/ridge,
                         -C.dot(soft)/ridge,np.dot(w,abs(soft))/ridge]
        constraints=[dict(type='ineq',fun=feasible,jac=jac)]
        if intercept and z:
            row=np.r_[np.ones(z),np.zeros(m+1)]
            constraints.append(dict(type='eq',fun=lambda v: q[nonzero].sum()+v[:z].sum(),jac=lambda v:row))
        objective=np.r_[np.zeros(z+m),1.]
        fit=minimize(lambda v:(v[-1],objective),start,jac=True,method='SLSQP',
            bounds=[(-2*rho,2*rho)]*z+[(None,None)]*m+[(0,None)],
            constraints=constraints,options=dict(ftol=1e-12,maxiter=3000))
        if not fit.success or feasible(fit.x)<-1e-7*max(1.,e):
            raise RuntimeError('R4 zero-scale threshold certificate failed: '+fit.message)
        return bias,0.,float(fit.x[-1])
    # Positive scale: solve the two free stationarity equations. Their
    # solution is globally optimal because the perspective is convex.
    initial_scale=max(np.linalg.norm(residual)/np.sqrt(e),np.finfo(float).tiny)
    def equations(v):
        b=v[0] if intercept else 0.;sigma=np.exp(v[-1])
        standardized=(b-y)/sigma
        response=2*np.clip(standardized,-rho,rho)
        return np.r_[response.sum() if intercept else [],
                     e-np.minimum(standardized**2,rho**2).sum()]
    start=np.r_[bias,np.log(initial_scale)] if intercept else np.array([np.log(initial_scale)])
    fit=least_squares(equations,start,xtol=1e-13,ftol=1e-13,gtol=1e-13,max_nfev=2000)
    if np.max(abs(equations(fit.x)))>1e-8*max(1.,e):
        raise RuntimeError('R4 baseline bias/scale stationarity failed')
    bias=float(fit.x[0]) if intercept else 0.;sigma=float(np.exp(fit.x[-1]))
    response=2*np.clip((bias-y)/sigma,-rho,rho)
    maximum=initialize_path(X/w,-response[:,None],C/w)['lambda_max']
    return bias,sigma,maximum


def _r4_zero_lambda(X,C,y,ridge,w,intercept,e,rho,baseline_scale):
    """Profile scale using ordinary constrained ridge-Huber (R2) solves.

    At fixed sigma, multiplying the objective by sigma gives Huber threshold
    rho*sigma and the SAME quadratic ridge. Only a scalar root remains.
    """
    from .elastic_net import problem_elastic_net
    from ._path_interpolation import evaluate_path
    cache={}
    def fit(sigma):
        if sigma not in cache:
            pb=problem_elastic_net((X,C,y),'R2',ridge=ridge,w=w,
                                   intercept=intercept,rho=rho*sigma)
            B=evaluate_path(pb,[0.])[0][:,0]
            beta=B[int(intercept):]
            residual=X.dot(beta)+(B[0] if intercept else 0.)-y
            derivative=e-np.minimum((residual/sigma)**2,rho**2).sum()-ridge*beta.dot(beta)/(2*sigma*sigma)
            cache[sigma]=B,derivative
        return cache[sigma]
    upper=max(baseline_scale,np.linalg.norm(y)/np.sqrt(e),1.)
    for _ in range(60):
        if fit(upper)[1]>0: break
        upper*=2
    else: raise RuntimeError('Could not bracket R4 scale above')
    lower=upper/2
    for _ in range(60):
        if fit(lower)[1]<0: break
        lower/=2
    else: raise RuntimeError('Could not bracket positive R4 scale below')
    sigma=brentq(lambda s:fit(s)[1],lower,upper,xtol=1e-11,rtol=1e-11)
    return fit(sigma)[0],sigma


def concomitant_path(matrix,values,typ,ridge,w,intercept,e,rho,true_lam,n_active):
    X,C,y=matrix
    X,y=np.asarray(X,float),np.asarray(y,float)
    if X.ndim!=2 or not all(X.shape) or y.shape!=(len(X),) or not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError('Require finite nonempty X and a matching outcome vector')
    n,p=X.shape
    C=np.empty((0,p)) if C is None else np.asarray(C,float)
    w=np.ones(p) if w is None else np.asarray(w,float)
    if C.ndim!=2 or C.shape[1]!=p or not np.isfinite(C).all(): raise ValueError('Invalid C')
    if w.shape!=(p,) or not np.isfinite(w).all() or np.any(w<=0): raise ValueError('Invalid w')
    e=(n/2 if typ=='R3' else n) if e is None else e
    if np.ndim(e) or not np.isfinite(e) or e<=0: raise ValueError('e must be finite and positive')
    if typ=='R4' and (not np.isfinite(rho) or rho<=0): raise ValueError('rho must be finite and positive')
    if typ=='R3':
        bias=float(y.mean()) if intercept else 0.
        initial_scale=np.linalg.norm(y-bias)/np.sqrt(e)
        maximum=(initialize_path(X/w,(2*(y-bias)/initial_scale)[:,None],C/w)['lambda_max'] if initial_scale else 0.)
        real=values if true_lam else values*maximum
        if initial_scale==0:
            B=np.r_[bias,np.zeros(p)] if intercept else np.zeros(p)
            return np.repeat(B[None],len(real),axis=0),real,np.zeros(len(real)),maximum
        pb,B,T,scale=_quadratic_path(X,C,y,w,np.full(p,ridge),intercept,e,float(real[-1]),n_active)
        result=[_evaluate(B,T,scale,float(lam)) for lam in real if
                not hasattr(pb,'path_info') or pb.path_info['stop_reason']!='n_active' or lam>=T[-1]/scale(B[-1])]
        actual=real[:len(result)]
        if hasattr(pb,'path_info') and pb.path_info['stop_reason']=='n_active':
            stop=T[-1]/scale(B[-1])
            if not len(actual) or abs(actual[-1]-stop)>1e-10*max(1.,stop):
                result.append((B[-1],scale(B[-1])));actual=np.r_[actual,stop]
        return np.asarray([b[:,0] for b,s in result]),np.asarray(actual),np.asarray([s for b,s in result]),maximum
    bias,initial_scale,maximum=_r4_baseline(X,C,y,w,intercept,e,rho,ridge)
    real=values if true_lam else values*maximum
    betas,scales,actual=[],[],[]
    # Mean-shift columns have no tree constraint or quadratic ridge. They
    # encode only observation outliers, not additional compositional features.
    design=sparse.hstack([sparse.csr_matrix(X),-sparse.eye(n)],format='csr')
    constraints=np.c_[C,np.zeros((len(C),n))]
    for lam in real:
        if lam>=maximum:
            B=np.r_[bias,np.zeros(p)] if intercept else np.zeros(p);sigma=initial_scale
        elif lam==0:
            B,sigma=_r4_zero_lambda(X,C,y,ridge,w,intercept,e,rho,initial_scale)
        else:
            weights=np.r_[w,np.full(n,2*rho/lam)]
            diagonal=np.r_[np.full(p,ridge),np.zeros(n)]
            pb,knots,T,scale=_quadratic_path(design,constraints,y,weights,diagonal,intercept,e,float(lam),n_active,p)
            if pb.path_info['stop_reason']=='n_active': break
            full,sigma=_evaluate(knots,T,scale,float(lam))
            B=full[:p+int(intercept),0]
        if n_active and np.count_nonzero(abs(B[int(intercept):])>1e-7)>n_active: break
        betas.append(B);scales.append(sigma);actual.append(float(lam))
    if not betas: raise RuntimeError('First requested fit exceeds n_active; request a larger lambda')
    return np.asarray(betas),np.asarray(actual),np.asarray(scales),maximum
