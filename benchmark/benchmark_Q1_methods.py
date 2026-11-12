"""Matched Q1 speed comparison: event paths, reused DR, and both reused PDS schemes."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import pathlib,sys,json,time,platform
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np,scipy
from classo import Q1_path
from classo.solve_Q1 import loss
from threadpoolctl import threadpool_limits, threadpool_info
# This OpenBLAS build uses OpenMP; OPENBLAS_NUM_THREADS alone is insufficient.
_thread_limits=threadpool_limits(limits=1)
assert all(pool['num_threads']==1 for pool in threadpool_info())

OUT=ROOT/'benchmark/q1-results/method-speed.json'
METHODS=['events','DR','P-PDS','PF-PDS']
rows=[]
environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
                 host=platform.platform(),machine=platform.machine(),blas_threads=1,threadpools=[{k:v for k,v in pool.items() if k in ('user_api','internal_api','num_threads','version','threading_layer','architecture')} for pool in threadpool_info()])

def save():
    OUT.write_text(json.dumps(dict(environment=environment,rows=rows),indent=2))

# Untimed warmup loads each backend, including shared SciPy components.
rng=np.random.default_rng(222);W=rng.normal(size=(20,5));v=W[:,0]-W[:,1]+rng.normal(size=20)*.3
for method in METHODS:Q1_path(W,v,[.05],.5,.8,method=method,tol=1e-8)

cases=[]
for tau in [.2,.5,.8]:
    rng=np.random.default_rng(0);X=rng.normal(size=(40,9));y=1.3+X[:,0]-X[:,2]+rng.standard_t(3,size=40)*.4
    cases.append((X,y,tau,'14-point-path',3))
for n,p,taus,repeats in [(80,500,[.2,.5,.8],3),(100,2000,[.5],1)]:
    rng=np.random.default_rng(804+p);X=rng.normal(size=(n,p));y=1+X[:,0]-X[:,1]+.4*rng.standard_t(4,size=n)
    for tau in taus:
        for regime in ['large-lambda','support-8']:cases.append((X,y,tau,regime,repeats))

for X,y,tau,regime,repeats in cases:
    n,p=X.shape;k=.8
    if regime=='14-point-path':
        lambdas=np.geomspace(.5,.01,14)
        baseline=Q1_path(X,y,lambdas,tau,k,method='events',tol=1e-8)
        def run(method):return Q1_path(X,y,lambdas,tau,k,method=method,tol=1e-8)
    else:
        # Target selection is outside every timed call. All timed backends
        # still compute their own intercept and lambda_max and solve cold.
        prefix=Q1_path(X,y,[1.,0.],tau,k,method='events',n_active=8,tol=1e-8)
        target=.9*prefix['lambdamax'] if regime=='large-lambda' else float(prefix['evaluated_lambdas'][-1])
        baseline=Q1_path(X,y,[target],tau,k,method='events',tol=1e-8)
        def run(method):return Q1_path(X,y,[target],tau,k,method=method,tol=1e-8)
    wanted=baseline['evaluated_lambdas']
    Z=np.c_[np.ones(n),X]
    objective=lambda b,l:loss(y-Z@b,tau,k).sum()/n+l*abs(b[1:]).sum()
    records={m:dict(seconds=[],iterations=[],kkt=[],objective_gap=[]) for m in METHODS}
    failures={}
    for repeat in range(repeats):
        order=METHODS[repeat%len(METHODS):]+METHODS[:repeat%len(METHODS)]
        for method in order:
            if method in failures:continue
            try:
                t=time.perf_counter();fit=run(method);elapsed=time.perf_counter()-t
                np.testing.assert_allclose(fit['evaluated_lambdas'],wanted,atol=1e-14,rtol=1e-12)
                gap=max(abs(objective(b,l)-objective(r,l)) for b,r,l in zip(fit['coefficients'],baseline['coefficients'],wanted))
                if gap>2e-6 or max(fit['kkt'])>1.001e-8:raise RuntimeError(f'accuracy: gap={gap}, KKT={max(fit["kkt"])}')
                record=records[method]
                record['seconds'].append(elapsed);record['iterations'].append(fit['iterations'])
                record['kkt'].append(float(max(fit['kkt'])));record['objective_gap'].append(float(gap))
            except Exception as e:failures[method]=str(e)
    for method in METHODS:
        record=records[method]
        if record['seconds']:record['median_seconds']=float(np.median(record['seconds']))
        if method in failures:record['error']=failures[method]
    row=dict(n=n,p=p,tau=tau,regime=regime,repetitions=repeats,
             lambdas=[float(x) for x in wanted],relative_final_lambda=float(wanted[-1]/baseline['lambdamax']),
             reference_support=int(np.count_nonzero(abs(baseline['coefficients'][-1,1:])>1e-9)),
             event_steps=len(baseline['events']),methods=records)
    rows.append(row);save()
    print(json.dumps(dict(n=n,p=p,tau=tau,regime=regime,times={m:r.get('median_seconds',r.get('error')) for m,r in records.items()})),flush=True)
if any('error' in r for row in rows for r in row['methods'].values()):sys.exit(1)
