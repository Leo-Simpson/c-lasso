"""Minimal entry/zero-face counterexamples supplied in c1_pr28_pr29.ipynb."""
import numpy as np
import pytest
from scipy.optimize import minimize,LinearConstraint
from classo.path_alg import solve_path,parameters_for_update,up_cl,check_C1_certificate


@pytest.mark.parametrize('typ,rho',[('C1',0.),('C2',-1.)])
@pytest.mark.parametrize('eps',[0.,1e-12,-1e-12,1e-8])
def test_constrained_near_tied_entry(typ,rho,eps):
    X=np.array([[1.,-(1.+eps)],[-1.,1.+eps]])
    y=np.array([1.,-1.]);C=np.ones((1,2));penalty=2.
    initial=parameters_for_update((X,C,y),0.,rho,typ)
    B,L=solve_path((X,C,y),penalty/initial.lambdamax,0,rho,typ)
    a=2.+eps;t=(1.-penalty/(2.*a))/a
    np.testing.assert_allclose(B[-1],[t,-t],atol=1e-9)
    np.testing.assert_allclose(C@B[-1],0.,atol=1e-10)


@pytest.mark.parametrize('typ,rho',[('C1',0.),('C2',-1.)])
@pytest.mark.parametrize('scale',[.01,1.,100.])
@pytest.mark.parametrize('permutation',[(0,1,2),(1,2,0),(2,0,1)])
def test_zero_coefficient_selects_feasible_outgoing_sign(typ,rho,scale,permutation):
    X=np.array([[-1.,-1.,0.],[-1.,2.,0.]])
    X=scale*X[:,permutation]
    y=np.array([1.,-1.]);C=np.ones((1,3));penalty=.6*scale
    initial=parameters_for_update((X,C,y),0.,rho,typ)
    B,L=solve_path((X,C,y),penalty/initial.lambdamax,0,rho,typ)
    np.testing.assert_allclose(B[-1],np.array([0.,-.48,.48])[list(permutation)]/scale,atol=1e-8)
    # Check every individual event with the unmodified actual-loss certificate.
    state=parameters_for_update((X,C,y),penalty/initial.lambdamax,rho,typ)
    for _ in range(30):
        check_C1_certificate(state)
        if state.lam==state.lamin:break
        previous=state.lam
        up_cl(state)
        assert state.lam<=previous
    else:pytest.fail('Boundary continuation did not reach the requested endpoint')
    check_C1_certificate(state)


def _independent_squared_hinge_qp(X,C,y,lam,bias):
 n,p=X.shape;Z=np.c_[np.ones(n),X] if bias else X;q=Z.shape[1];off=int(bias)
 F=np.c_[np.zeros((p,off)),np.eye(p)]
 constraints=[LinearConstraint(np.block([[F,-np.eye(p),np.zeros((p,n))],[-F,-np.eye(p),np.zeros((p,n))]]),-np.inf,0),
 LinearConstraint(np.c_[y[:,None]*Z,np.zeros((n,p)),np.eye(n)],1,np.inf)]
 if len(C):constraints.append(LinearConstraint(np.c_[np.zeros((len(C),off)),C,np.zeros((len(C),p+n))],0,0))
 def fun(v):return v[-n:]@v[-n:]+lam*v[q:q+p].sum(),np.r_[np.zeros(q),np.full(p,lam),2*v[-n:]]
 fit=minimize(fun,np.r_[np.zeros(q+p),np.ones(n)],jac=True,method='SLSQP',constraints=constraints,bounds=[(None,None)]*q+[(0,None)]*(p+n),options=dict(ftol=1e-10,maxiter=3000))
 if not fit.success:raise RuntimeError('Reference: '+fit.message)
 return fit.fun

@pytest.mark.parametrize('seed',range(10))
@pytest.mark.parametrize('bias',[False,True])
@pytest.mark.parametrize('constrained',[False,True])
def test_small_paths_match_independent_quadratic_program(seed,bias,constrained):
    from threadpoolctl import threadpool_limits
    from classo.path_alg import pathalgo_general
    rng=np.random.default_rng(seed)
    n=[2,3,8,15,24][seed%5];p=[3,5,7][seed%3]
    X=rng.normal(size=(n,p));y=np.r_[1.,-1.,rng.choice([-1.,1.],n-2)]
    C=np.ones((1,p)) if constrained else np.empty((0,p))
    fractions=[.8,.4,.1,.05]
    with threadpool_limits(limits=1):
        initial=parameters_for_update((X,C,y),.05,0,'C1',intercept=bias)
        B=pathalgo_general((X,C,y),fractions,'C1',intercept=bias)
        Z=np.c_[np.ones(n),X] if bias else X
        for fraction,beta in zip(fractions,B):
            penalty=fraction*initial.lambdamax
            residual=np.maximum(1-y*(Z@beta),0)
            value=residual@residual+penalty*np.abs(beta[int(bias):]).sum()
            expected=_independent_squared_hinge_qp(X,C,y,penalty,bias)
            np.testing.assert_allclose(value,expected,atol=1e-7,rtol=1e-8)
