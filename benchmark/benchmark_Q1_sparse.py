"""Large-lambda / bounded-support Q1 timing; end-to-end and paired targets."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import pathlib,sys,json,time,platform
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,scipy
from classo import Q1_path
from classo.solve_Q1 import loss
from threadpoolctl import threadpool_info, threadpool_limits
_thread_limits=threadpool_limits(limits=1)
assert all(pool['num_threads']==1 for pool in threadpool_info())
rows=[]
for n,p in [(80,500),(100,2000),(100,5000)]:
 rng=np.random.default_rng(804+p);X=rng.normal(size=(n,p));y=1+X[:,0]-X[:,1]+.4*rng.standard_t(4,size=n)
 for tau in [.2,.5,.8]:
  for regime in ['large-lambda','support-8']:
   try:
    event_times=[];dr_times=[]
    repetitions=1 if p==5000 else 3
    for repeat in range(repetitions):
     # Include lambda_max, certificates, and allocations in event timing.
     t=time.perf_counter()
     if regime=='large-lambda':
      # Compute lambda_max as part of this timed invocation as well.
      from classo.solve_Q1 import problem_Q1
      pb=problem_Q1((X,np.ones((1,p)),y),'Path-Alg',tau,.8,intercept=True)
      target=.9*pb.lambdamax
      fit=Q1_path(X,y,[target],tau,.8,method='events')
     else:fit=Q1_path(X,y,[1.,0.],tau,.8,method='events',n_active=8)
     event_times.append(time.perf_counter()-t)
     target=float(fit['evaluated_lambdas'][-1]);b=fit['coefficients'][-1]
     t=time.perf_counter();dr=Q1_path(X,y,[target],tau,.8,method='DR');dr_times.append(time.perf_counter()-t)
     c=dr['coefficients'][-1]
     objective=lambda v:loss(y-v[0]-X@v[1:],tau,.8).sum()/n+target*abs(v[1:]).sum()
     gap=abs(objective(b)-objective(c))
     if gap>2e-6:raise RuntimeError(f'objective gap {gap}')
    row=dict(n=n,p=p,tau=tau,regime=regime,repetitions=repetitions,path_seconds=event_times,dr_seconds=dr_times,
             median_path_seconds=float(np.median(event_times)),median_dr_seconds=float(np.median(dr_times)),
             dr_over_path=float(np.median(dr_times)/np.median(event_times)),lambda_absolute=target,
             relative_lambda=target/fit['lambdamax'],support=int(np.count_nonzero(abs(b[1:])>1e-9)),
             knots=len(fit['events']),flat_pivots=sum(e['kind']=='flat-pivot' for e in fit['events']),
             objective_gap=gap,path_kkt=float(max(fit['kkt'])),dr_kkt=float(max(dr['kkt'])),dr_iterations=dr['iterations'])
   except Exception as e:row=dict(n=n,p=p,tau=tau,regime=regime,error=str(e))
   rows.append(row);print(row,flush=True)
   (ROOT/'benchmark/q1-results/sparse-benchmark.json').write_text(json.dumps(dict(environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,blas_threads=1,threadpools=[{k:v for k,v in pool.items() if k in ('user_api','internal_api','num_threads','version','threading_layer','architecture')} for pool in threadpool_info()]),rows=rows),indent=2))
