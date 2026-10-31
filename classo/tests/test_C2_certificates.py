"""Huberized-hinge path checked against actual margins and an independent solve."""
import numpy as np
import pytest
from scipy.optimize import minimize,LinearConstraint
from classo.path_alg import parameters_for_update,up_huber_cl,solve_path,check_C2_certificate


def sample(seed=0):
    rng=np.random.default_rng(seed);X=rng.normal(size=(24,5))
    y=np.where(X[:,0]-X[:,1]+rng.normal(size=len(X))>.1,1.,-1.)
    return X,np.ones((1,5)),y


def reference(X,C,y,rho,lam,intercept):
    n,p=X.shape;Z=np.c_[np.ones(n),X] if intercept else X
    q=Z.shape[1];offset=int(intercept)
    def objective(v):
        beta=v[:q];margin=y*(Z@beta)
        loss=np.where(margin<rho,(1-rho)**2+2*(1-rho)*(rho-margin),np.maximum(1-margin,0)**2)
        derivative=2*y*np.minimum(np.maximum(margin,rho)-1,0)
        return loss.sum()+lam*v[q:].sum(),np.r_[Z.T@derivative,np.full(p,lam)]
    F=np.c_[np.zeros((p,offset)),np.eye(p)]
    constraints=[LinearConstraint(np.block([[F,-np.eye(p)],[-F,-np.eye(p)]]),-np.inf,0)]
    if len(C):constraints.append(LinearConstraint(np.c_[np.zeros((len(C),offset)),C,np.zeros((len(C),p))],0,0))
    fit=minimize(objective,np.zeros(q+p),jac=True,method='SLSQP',constraints=constraints,options=dict(ftol=1e-12,maxiter=3000))
    assert fit.success,fit.message
    return fit.x[:q],fit.fun


def test_C2_tracks_current_intercept_after_region_events():
    X,C,y=sample();pb=parameters_for_update((X,C,y),.02,-1,'C2',intercept=True)
    changed=False
    for _ in range(200):
        np.testing.assert_allclose(pb.r,y*(X@pb.beta+pb.beta0),atol=1e-11)
        check_C2_certificate(pb)
        if pb.lam<=.02:break
        old=pb.F.copy();up_huber_cl(pb);changed|=not np.array_equal(old,pb.F)
    else:pytest.fail('C2 did not reach target')
    assert changed and pb.eps_L2==0


@pytest.mark.parametrize('rho',[-1.,-.5,0.])
@pytest.mark.parametrize('intercept',[False,True])
@pytest.mark.parametrize('constrained',[False,True])
def test_C2_matches_independent_unregularized_objective(rho,intercept,constrained):
    X,C,y=sample()
    if not constrained:C=np.empty((0,len(X[0])))
    initial=parameters_for_update((X,C,y),.08,rho,'C2',intercept=intercept)
    out=solve_path((X,C,y),.08,0,rho,'C2',intercept=intercept)
    if intercept:b0,B,L=out;beta=np.r_[b0[-1],B[-1]]
    else:B,L=out;beta=B[-1]
    expected,_=reference(X,C,y,rho,L[-1]*initial.lambdamax,intercept)
    np.testing.assert_allclose(beta,expected,atol=5e-6,rtol=5e-6)


def test_C2_switches_simultaneous_margins():
    X,C,y=sample();X[-1]=X[0];y[-1]=y[0];X[-2]=X[1];y[-2]=y[1]
    pb=parameters_for_update((X,C,y),.02,-1,'C2',intercept=True)
    for _ in range(200):
        check_C2_certificate(pb)
        margin=y*(X@pb.beta+pb.beta0)
        assert not np.any(pb.F&((margin>1+1e-8)|(margin<-1-1e-8)))
        if pb.lam<=.02:break
        up_huber_cl(pb)
    else:pytest.fail('C2 did not finish')


def test_C2_continues_flat_start_instead_of_silently_truncating():
    X=np.array([[1.,0.],[-1.,0.],[2.,1.],[-2.,-1.]])
    b0,B,L=solve_path((X,np.empty((0,2)),np.array([1.,-1.,1.,-1.])),.01,0,.5,'C2',intercept=True)
    assert L[-1]==.01 and len(L)>2
    assert np.isfinite(B).all() and np.isfinite(b0).all()
