"""Small-support Q1 event solve in the original residual coordinates."""
import numpy as np
from ._classification_path import _nullspace, _solve_design, _dual_segment


def curved_direction(Z,C,y,b,lam,signed,tau,k,divisor,free):
    """Solve only the tight coefficient face; keep residual-zero off the event list.

    Ambiguous cones fall back to the shared critical-cone QP, never to a ridge.
    """
    nonzero=abs(b)>1e-9
    tight=abs(signed)>=lam-2e-8*max(1.,lam)
    work=np.flatnonzero(nonzero|tight|(np.arange(len(b))<free))
    A=Z[:,work];N=_nullspace(C[:,work])
    if not N.shape[1]:raise np.linalg.LinAlgError('no curved tangent')
    signs=np.where(nonzero[work],np.sign(b[work]),np.sign(signed[work]));signs[work<free]=0.
    r=y-Z[:,nonzero]@b[nonzero]
    L,U=-(1-tau)*k,tau*k
    eps=2e-10*max(1.,k)
    interior=(r>L+eps)&(r<U-eps);lo=abs(r-L)<=eps;hi=abs(r-U)<=eps
    region=interior|lo|hi;seen=set()
    for _ in range(12):
        if region.tobytes() in seen:raise np.linalg.LinAlgError('outgoing residual regions cycled')
        seen.add(region.tobytes())
        design=A[region]@N/np.sqrt(2*k*divisor)
        v=_solve_design(design,N.T@signs)
        local=N@v;rdot=-A@local
        revised=interior|(lo&(rdot>0))|(hi&(rdot<0))
        # A tangent boundary with zero rate contributes no curvature either way.
        bad=(region!=revised)&(abs(rdot)>1e-11)
        if not bad.any():break
        region=revised
    else:raise np.linalg.LinAlgError('outgoing residual regions unresolved')
    zero=~nonzero[work]&(work>=free)
    if np.min(signs[zero]*local[zero],initial=0.) < -1e-9:
        raise np.linalg.LinAlgError('direction lies outside coefficient critical cone')
    D=np.zeros_like(b);D[work]=local
    active=np.flatnonzero(nonzero|(abs(D)>1e-9)|(np.arange(len(b))<free))
    return D,active,region


def _dual_interval(C,gradient,gdot,lam,upper,active,signs,free):
    """Analytic inactive entries when active stationarity fixes multipliers."""
    E=C[:,active].T
    rhs=np.c_[-gradient[active]-lam*signs[active],signs[active]-gdot[active]]
    if len(C):
        solution,_,rank,_=np.linalg.lstsq(E,rhs,rcond=1e-12)
        if rank<len(C):return _dual_segment(C,gradient,gdot,lam,upper,active,signs,free)
        nu,dnu=solution.T
    else:nu=dnu=np.empty(0)
    if np.max(abs(E@nu-rhs[:,0]),initial=0.)>1e-9 or upper*np.max(abs(E@dnu-rhs[:,1]),initial=0.)>1e-9:
        raise RuntimeError('Q1 active derivative is not stationary')
    adjusted=gradient+C.T@nu;rate=gdot+C.T@dnu
    inactive=np.ones(len(gradient),bool);inactive[active]=False;inactive[:free]=False
    bound=upper
    for numerator,denominator in [(lam-adjusted,rate+1),(lam+adjusted,1-rate)]:
        mask=inactive&(denominator>1e-12)
        candidates=numerator[mask]/denominator[mask]
        bound=min(bound,float(np.min(candidates,initial=upper)))
    delta=float(bound)
    return delta,nu+delta*dnu


def next_event(Z,C,y,b,lam,D,active,region,cap,tau,k,divisor,free):
    nonzero=abs(b)>1e-9
    r=y-Z[:,nonzero]@b[nonzero]
    moving=np.flatnonzero(abs(D)>0)
    rate=-Z[:,moving]@D[moving]
    eps=64*np.finfo(float).eps*max(1.,lam)
    bound=cap
    for threshold in (-(1-tau)*k,tau*k):
        roots=(threshold-r[rate!=0])/rate[rate!=0]
        bound=min(bound,float(np.min(roots[roots>eps],initial=cap)))
    exits=nonzero&(np.arange(len(b))>=free)&(b*D<0)
    upper=min(bound,float(np.min(-b[exits]/D[exits],initial=cap)))
    gradient=-Z.T@np.clip(r/k,tau-1,tau)/divisor
    gdot=-Z.T@(region*rate)/(k*divisor)
    signs=np.sign(b);signs[~nonzero]=np.sign(D[~nonzero]);signs[:free]=0.
    delta,dual=_dual_interval(C,gradient,gdot,lam,upper,active,signs,free)
    if delta<=eps:raise RuntimeError('Q1 zero-length curved interval')
    endpoint=b+delta*D
    rr=r+delta*rate
    adjusted=-Z.T@np.clip(rr/k,tau-1,tau)/divisor+C.T@dual
    error=max(np.max(abs(adjusted[active]+(lam-delta)*signs[active]),initial=0.),
              np.max(np.maximum(abs(adjusted[free:])-(lam-delta),0.),initial=0.),
              np.max(abs(C@endpoint),initial=0.))
    if error>5e-7*max(1.,lam):raise RuntimeError('Q1 curved endpoint KKT')
    return delta
