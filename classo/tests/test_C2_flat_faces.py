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
