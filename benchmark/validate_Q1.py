import sys, pathlib, json,time,platform
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.append(str(ROOT.parent/'.comparison-deps'))
import numpy as np, scipy, cvxpy as cp
from classo.solve_Q1 import quantile_path,loss
from classo.compact_func import pathlasso

def reference(X,y,lams,tau,k,C,intercept,divisor):
    # Moreau envelope representation, independently modeled as a convex QP.
    p=X.shape[1]; b=cp.Variable(p); a=cp.Variable() if intercept else 0.
    v=cp.Variable(len(y));pos=cp.Variable(len(y),nonneg=True);neg=cp.Variable(len(y),nonneg=True)
    lam=cp.Parameter(nonneg=True)
    objective=cp.sum_squares(v)/(2*k*divisor)+(tau*cp.sum(pos)+(1-tau)*cp.sum(neg))/divisor+lam*cp.norm1(b)
    constraints=[y-X@b-a==v+pos-neg]
    if len(C):constraints.append(C@b==0)
    problem=cp.Problem(cp.Minimize(objective),constraints)
    result=[]
    for value in lams:
        lam.value=value
        problem.solve(solver='CLARABEL',tol_gap_abs=1e-10,tol_feas=1e-10,tol_gap_rel=1e-10)
        if problem.status!='optimal':raise RuntimeError(problem.status)
        result.append(np.r_[a.value,b.value] if intercept else b.value)
    return np.array(result)

def objective(X,y,B,lam,tau,k,divisor,intercept):
    return loss(y-(X@B[1:]+B[0] if intercept else X@B),tau,k).sum()/divisor+lam*abs(B[int(intercept):]).sum()

rows=[];failures=[]
for seed in range(8):
    rng=np.random.default_rng(seed);n,p=(40,9) if seed<6 else (24,30)
    X=rng.normal(size=(n,p));C=np.ones((1,p));intercept=seed%2==0
    if seed==2:X[:,1]=X[:,0] # dependent design
    if seed==3:C=np.vstack([C,2*C]) # redundant constraints
    y=1.3+X[:,0]-X[:,2]+rng.standard_t(3,size=n)*.4
    if seed==4:y=np.round(y,0)
    divisor=n if seed!=5 else 12 # subject rather than row normalization
    lams=np.geomspace(.5,.01,14);k=.8
    for tau in (.2,.5,.8):
        t=time.perf_counter();ref=reference(X,y,lams,tau,k,C,intercept,divisor);rt=time.perf_counter()-t
        for method in ('warm','events','DR'):
            try:
                t=time.perf_counter();fit=quantile_path(X,y,lams,tau,k,C,intercept,divisor,method,store_segments=(method=='events'));elapsed=time.perf_counter()-t
                gaps=[abs(objective(X,y,b,l,tau,k,divisor,intercept)-objective(X,y,r,l,tau,k,divisor,intercept)) for b,r,l in zip(fit['coefficients'],ref,lams)]
                pred=[np.max(abs((np.c_[np.ones(n),X] if intercept else X)@(b-r))) for b,r in zip(fit['coefficients'],ref)]
                interior_gap=0.
                if method=='events':
                    segments=[e for e in fit['events'] if e['delta']>1e-12]
                    selected=[segments[i] for i in np.unique(np.linspace(0,len(segments)-1,min(3,len(segments)),dtype=int))] if segments else []
                    mid_lams=[e['start_lambda']-.5*e['delta'] for e in selected]
                    if selected:
                        mid_ref=reference(X,y,mid_lams,tau,k,C,intercept,divisor)
                        for e,r,l in zip(selected,mid_ref,mid_lams):
                            mid=e['coefficients']+.5*e['delta']*e['direction']
                            interior_gap=max(interior_gap,abs(objective(X,y,mid,l,tau,k,divisor,intercept)-objective(X,y,r,l,tau,k,divisor,intercept)))
                        assert interior_gap<2e-7,interior_gap
                row=dict(interior_objective_gap=interior_gap,seed=seed,tau=tau,method=method,n=n,p=p,intercept=intercept,seconds=elapsed,reference_seconds=rt,max_objective_gap=max(gaps),max_prediction_difference=max(pred),kkt=float(max(fit['kkt'])),events=len(fit['events']),iterations=fit['iterations'])
                assert max(gaps)<2e-7,row
                rows.append(row)
            except Exception as e:failures.append(dict(seed=seed,tau=tau,method=method,error=str(e)))
    print('seed',seed,'done',flush=True)
# Correctly scaled comparison with upstream R2; its objective is sum H_repo.
rng=np.random.default_rng(120);X=rng.normal(size=(50,10));y=2+X[:,0]-X[:,1]+rng.normal(size=50);C=np.ones((1,10));k=1.2;lams=np.geomspace(.3,.01,20)
q=quantile_path(X,y,lams,.5,k,C,True,method='events')
t=time.perf_counter();h=pathlasso((X,C,y),lambdas=2*k*len(y)*lams,typ='R2',meth='Path-Alg',rho=k/2,true_lam=True,intercept=True);ht=time.perf_counter()-t
if isinstance(h,tuple):h=h[0]
h=np.asarray(h)
huber=dict(max_coefficient_difference=float(np.max(abs(h-q['coefficients']))),seconds=ht)
result=dict(environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,cvxpy=cp.__version__),rows=rows,failures=failures,huber=huber)
(ROOT/'benchmark'/'q1-results'/'validation.json').write_text(json.dumps(result,indent=2))
print(json.dumps(dict(successes=len(rows),failures=failures,huber=huber),indent=2))
