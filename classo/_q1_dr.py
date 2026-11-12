"""Q1 mean-shift adapter to the existing R1 Douglas-Rachford solver."""
import numpy as np
from scipy.sparse.linalg import LinearOperator
from .solve_R1 import problem_R1, Classo_R1


class _Q1Quadratic(problem_R1):
    """Reuse the R1 iteration without allocating a p-by-p Gram or projector."""
    def compute_param(self):
        A,C,y=self.matrix
        self.Anorm=np.linalg.norm(A,'fro')
        self.AtA=None
        self.AAt=A@A.T
        # R1's DR gamma=gam/(4*AtAnorm*relative_lambda); relative lambda is 1.
        self.AtAnorm=1.
        self.gam=2.
        self.tauN=0.
        self.c=1.


class Q1DR:
    """Exact Moreau mean-shift reduction, matching R2's reuse of R1.

    Multiply Q1 by 2*k*D:
      min ||Zb+o-y||² + 2*k*D*lambda ||b_slopes||_1
          + k ||o||_1 + 2*k*(tau-.5) sum(o), Cb=0.
    The asymmetry is solely a linear shift in the auxiliary L1 prox.
    """
    def __init__(self,Z,C,y,tau,k,divisor,free):
        self.Z,self.y,self.tau,self.k,self.divisor,self.free=Z,y,tau,k,divisor,free
        self.q=Z.shape[1]
        slope_scale=max(1.,np.linalg.norm(Z[:,free:],'fro')/np.sqrt(len(y)))
        self.scale=np.full(self.q,slope_scale)
        # The unpenalized intercept has a different column norm. Scaling it
        # like p slopes creates an artificial p-dependent convergence rate.
        self.scale[:free]=max(1.,np.sqrt(len(y)))
        A=np.c_[Z/self.scale,np.eye(len(y))]
        constraints=np.c_[C,np.zeros((len(C),len(y)))]
        pb=_Q1Quadratic((A,constraints,y),'DR')
        pb.matrix_free=True
        # No dense projector: C has orthonormal independent rows.
        project=lambda v:v-constraints.T@(constraints@v)
        pb.projector=LinearOperator((A.shape[1],A.shape[1]),matvec=project,matmat=project,dtype=float)
        pb.linear=np.r_[np.zeros(self.q),np.full(len(y),2*k*(tau-.5))]
        pb.lambdamax=1.
        pb.regpath=True
        pb.compute_param()
        self.pb=pb
        self.initialized=False

    def solve(self,lam,initial,certificate,tol=1e-8,max_iter=100000):
        pb=self.pb
        weights=np.r_[np.full(self.q,2*self.k*self.divisor*lam/self.scale),np.full(len(self.y),self.k)]
        weights[:self.free]=0.
        pb.weights=weights
        if not self.initialized:
            r=self.y-self.Z@initial
            o=r-self.k*np.clip(r/self.k,self.tau-1,self.tau)
            z=np.r_[initial*self.scale,o]
            pb.init=z.copy(),z.copy(),z.copy()
            self.initialized=True
        pb.N=max_iter
        pb.tol=1e-7*pb.Anorm/max(1.,np.linalg.norm(self.y))
        pb.certificate_check=lambda z:certificate(z[:self.q]/self.scale,lam)[0]<=tol
        result,state=Classo_R1(pb,1.)
        pb.init=state
        return result[:self.q]/self.scale,pb.n_iter
