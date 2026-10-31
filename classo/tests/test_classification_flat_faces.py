"""Flat classifier loss regions: LP pivots, constant tails and interpolation."""
import numpy as np
import pytest
from classo import Classo,pathlasso
from classo.path_alg import parameters_for_update,solve_path,pathalgo_general,check_C1_certificate,up_cl
from classo.tests.test_C2_certificates import reference


def data(constrained):
    X=np.array([[1.,0.],[-1.,0.],[2.,1.],[-2.,-1.]])
    return X,np.ones((1,2)) if constrained else np.empty((0,2)),np.array([1.,-1.,1.,-1.])


def loss_value(X,y,beta,rho,lam,intercept):
    Z=np.c_[np.ones(len(X)),X] if intercept else X
    m=y*(Z@beta)
    loss=np.where(m<rho,(1-rho)**2+2*(1-rho)*(rho-m),np.maximum(1-m,0)**2).sum()
    return loss+lam*np.abs(beta[int(intercept):]).sum()


@pytest.mark.parametrize('rho',[.25,.5,.8])
@pytest.mark.parametrize('intercept',[False,True])
@pytest.mark.parametrize('constrained',[False,True])
def test_C2_flat_start_and_interpolated_path_match_independent_objective(rho,intercept,constrained):
    X,C,y=data(constrained)
    initial=parameters_for_update((X,C,y),.03,rho,'C2',intercept=intercept)
    assert not initial.F.any()
    fractions=np.array([1.,.99,.8,.5,.2,.03])
    coefficients=pathalgo_general((X,C,y),fractions,'C2',rho=rho,intercept=intercept)
    for fraction,beta in zip(fractions,coefficients):
        _,expected=reference(X,C,y,rho,fraction*initial.lambdamax,intercept)
        np.testing.assert_allclose(loss_value(X,y,beta,rho,fraction*initial.lambdamax,intercept),expected,atol=2e-8,rtol=2e-8)
        np.testing.assert_allclose(C@beta[int(intercept):],0,atol=1e-10)


def test_flat_face_pivot_preserves_lambda_and_objective_then_recovers_curvature():
    X,C,y=data(False)
    p=parameters_for_update((X,C,y),.03,.5,'C2',intercept=True)
    before=p.lam;old=np.r_[p.beta0,p.beta]
    value=loss_value(X,y,old,.5,p.lam*p.lambdamax,True)
    up_cl(p)
    assert p.lam==before and p.flat_events[-1]['kind']=='linear_face_pivot'
    assert np.any(p.F) and np.linalg.norm(p.beta)>0
    check_C1_certificate(p)
    np.testing.assert_allclose(loss_value(X,y,np.r_[p.beta0,p.beta],.5,p.lam*p.lambdamax,True),value,atol=1e-9)


def test_flat_constrained_start_first_lowers_lambda_without_moving_primal():
    X,C,y=data(True)
    p=parameters_for_update((X,C,y),.03,.5,'C2',intercept=True)
    up_cl(p)
    assert p.lam<1 and p.flat_events[-1]['kind']=='constant_primal'
    np.testing.assert_allclose(p.beta,0,atol=1e-12)
    check_C1_certificate(p)


@pytest.mark.parametrize('typ',['C1','C2'])
def test_empty_zero_loss_region_can_finish_lambda_zero(typ):
    X,C,y=data(False)
    p=parameters_for_update((X,C,y),0.,-1.,typ,intercept=True)
    # A certified lambda-zero separator with every margin strictly above one.
    p.beta=np.array([2.,0.]);p.beta0=0.;p.r=y*(X@p.beta);p.F=np.zeros(len(y),bool);p.lam=0.
    up_cl(p)
    assert p.lam==0 and np.isfinite(p.M).all()
    check_C1_certificate(p)


@pytest.mark.parametrize('typ',['C1','C2'])
def test_zero_lambda_max_returns_constant_path(typ):
    X=np.arange(12,dtype=float).reshape(6,2);C=np.empty((0,2));y=np.ones(6)
    out=solve_path((X,C,y),.01,0,-1.,typ,intercept=True)
    b0,B,L=out
    assert L==[1.,.01]
    np.testing.assert_allclose(B,0)
    np.testing.assert_allclose(b0,1)


@pytest.mark.parametrize('typ',['C1','C2'])
def test_public_absolute_lambda_with_zero_lambda_max(typ):
    X=np.arange(12,dtype=float).reshape(6,2);C=np.empty((0,2));y=np.ones(6)
    beta=Classo((X,C,y),0.,typ=typ,true_lam=True,intercept=True)
    np.testing.assert_allclose(beta,[1.,0.,0.])
    B,L=pathlasso((X,C,y),lambdas=[1.,.1,0.],typ=typ,true_lam=True,intercept=True)
    np.testing.assert_allclose(B,[[1.,0.,0.]]*3)
    np.testing.assert_allclose(L,[1.,.1,0.])


@pytest.mark.parametrize('rho',[.2,.5,.8])
@pytest.mark.parametrize('intercept',[False,True])
@pytest.mark.parametrize('constrained',[False,True])
def test_repeated_flat_cells_against_independent_slack_qp(rho,intercept,constrained):
    from classo.tests.reference_classification import reference as qp_reference
    rng=np.random.default_rng(0);y=np.r_[np.ones(12),-np.ones(12)];rng.shuffle(y)
    X=rng.normal(size=(24,4));X[:,0]+=.8*y
    C=np.ones((1,4)) if constrained else np.empty((0,4))
    p=parameters_for_update((X,C,y),.05,rho,'C2',intercept=intercept)
    fractions=[.8,.2,.05]
    B=pathalgo_general((X,C,y),fractions,'C2',rho=rho,intercept=intercept)
    for fraction,beta in zip(fractions,B):
        _,value=qp_reference(X,C,y,rho,fraction*p.lambdamax,intercept)
        np.testing.assert_allclose(loss_value(X,y,beta,rho,fraction*p.lambdamax,intercept),value,atol=1e-7,rtol=1e-8)
