import numpy as np
from scipy.optimize import minimize,LinearConstraint


def reference(X,C,y,rho,lam,intercept):
 n,p=X.shape;Z=np.c_[np.ones(n),X] if intercept else X;q=Z.shape[1];o=int(intercept)
 # Variables [coefficients, absolute coefficients, quadratic residual, tail].
 size=q+p+2*n
 def objective(v):
  residual=v[q+p:q+p+n]
  value=residual@residual+2*(1-rho)*v[-n:].sum()+lam*v[q:q+p].sum()
  gradient=np.r_[np.zeros(q),np.full(p,lam),2*residual,np.full(n,2*(1-rho))]
  return value,gradient
 F=np.c_[np.zeros((p,o)),np.eye(p)]
 epigraph=np.block([[F,-np.eye(p),np.zeros((p,2*n))],[-F,-np.eye(p),np.zeros((p,2*n))]])
 margins=np.c_[y[:,None]*Z,np.zeros((n,p)),np.eye(n),np.eye(n)]
 constraints=[LinearConstraint(epigraph,-np.inf,0),LinearConstraint(margins,1,np.inf)]
 if len(C):constraints.append(LinearConstraint(np.c_[np.zeros((len(C),o)),C,np.zeros((len(C),p+2*n))],0,0))
 initial=np.r_[np.zeros(q+p),np.full(n,min(1.,1-rho)),np.full(n,max(rho,0.))]
 bounds=[(None,None)]*q+[(0,None)]*(p+2*n)
 fit=minimize(objective,initial,jac=True,method='SLSQP',constraints=constraints,bounds=bounds,options=dict(ftol=1e-9,maxiter=3000))
 if not fit.success:raise RuntimeError('Independent slack QP: '+fit.message)
 return fit.x[:q],float(fit.fun)
