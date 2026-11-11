"""Shared objective, constraint and unit invariants across the six formulations."""
import numpy as np
import pytest
from classo import Classo,pathlasso
from classo.misc_functions import proj_c,unpenalized

TYPES=['R1','R2','R3','R4','C1','C2']


def data(typ):
    rng=np.random.default_rng(42)
    X=rng.normal(size=(16,4))
    y=1.5+X[:,0]-X[:,1]+rng.normal(size=16)*.4
    y[0]+=3
    if typ.startswith('C'):
        y=np.where(y>np.median(y),1.,-1.)
    return X,np.ones((1,4)),y


def coefficients(result,typ):
    return np.asarray(result[0] if typ in ['R3','R4'] else result)


@pytest.mark.parametrize('typ',TYPES)
@pytest.mark.parametrize('intercept',[False,True])
def test_constraint_representation_does_not_change_fit(typ,intercept):
    X,C,y=data(typ)
    other=np.vstack([1e-8*C,-1e8*C,np.zeros_like(C),C])
    kwargs=dict(typ=typ,meth='Path-Alg',true_lam=True,intercept=intercept)
    a=coefficients(Classo((X,C,y),1.2,**kwargs),typ)
    b=coefficients(Classo((X,other,y),1.2,**kwargs),typ)
    np.testing.assert_allclose(a,b,atol=2e-6,rtol=2e-6)


@pytest.mark.parametrize('typ',TYPES)
def test_feature_units_with_matching_penalty_factors(typ):
    X,C,y=data(typ);units=np.array([.2,2.,.5,3.])
    kwargs=dict(typ=typ,meth='Path-Alg',true_lam=True,intercept=True)
    a=coefficients(Classo((X,C,y),1.2,**kwargs),typ)
    b=coefficients(Classo((X*units,C*units,y),1.2,w=units,**kwargs),typ)
    np.testing.assert_allclose(a,np.r_[b[0],b[1:]*units],atol=2e-6,rtol=2e-6)


@pytest.mark.parametrize('typ',TYPES)
def test_single_fit_agrees_with_path_including_zero(typ):
    X,C,y=data(typ)
    # Positive penalties only for classification: separable zero-penalty
    # classification has a nonunique optimum, and need not match coefficients.
    penalties=[2.,1.2,.4] if typ.startswith('C') else [2.,1.2,.4,0.]
    kwargs=dict(typ=typ,meth='Path-Alg',true_lam=True,intercept=True)
    path=pathlasso((X,C,y),lambdas=penalties,**kwargs)
    for beta,lam in zip(path[0],penalties):
        single=coefficients(Classo((X,C,y),lam,**kwargs),typ)
        np.testing.assert_allclose(X@beta[1:]+beta[0],X@single[1:]+single[0],atol=2e-5,rtol=2e-5)


@pytest.mark.parametrize('typ',['R1','R2','R3','R4'])
def test_response_units_preserve_mapped_problem(typ):
    X,C,y=data(typ);scale=10.
    kwargs=dict(typ=typ,meth='Path-Alg',true_lam=True,intercept=True)
    a=Classo((X,C,y),1.2,**kwargs)
    kwargs['rho']=1.345*scale if typ=='R2' else 1.345
    b=Classo((X,C,y*scale),1.2*(scale if typ in ['R1','R2'] else 1.),**kwargs)
    np.testing.assert_allclose(coefficients(a,typ),coefficients(b,typ)/scale,atol=3e-6,rtol=3e-6)
    if typ in ['R3','R4']:
        assert b[1]/scale==pytest.approx(a[1],rel=3e-6,abs=3e-6)


@pytest.mark.parametrize('typ',TYPES)
def test_zero_design_absolute_penalties_and_empty_constraints(typ):
    X,C,y=data(typ);X=np.zeros_like(X);C=np.empty((0,4))
    kwargs=dict(typ=typ,meth='Path-Alg',true_lam=True,intercept=True)
    a=coefficients(Classo((X,C,y),1.2,**kwargs),typ)
    assert np.isfinite(a).all()
    np.testing.assert_allclose(a[1:],0.,atol=1e-12)
    path=pathlasso((X,C,y),lambdas=[1.2,.2,0.],**kwargs)
    np.testing.assert_allclose(path[0],np.tile(a,(3,1)),atol=1e-8)
    np.testing.assert_allclose(path[1],[1.2,.2,0.])


@pytest.mark.parametrize('typ',['R1','R2'])
def test_small_positive_penalty_is_not_unpenalized(typ):
    if typ=='R1':
        matrix=(np.ones((1,1)),np.empty((0,1)),np.ones(1))
    else:
        matrix=(np.ones((3,1)),np.empty((0,1)),np.array([0.,1.,10.]))
    beta=Classo(matrix,1e-6,typ=typ,meth='Path-Alg',rho=1.)
    np.testing.assert_allclose(beta,[1.-1e-6],atol=1e-11,rtol=0.)


@pytest.mark.parametrize('method',['Path-Alg','DR','P-PDS','PF-PDS'])
def test_zero_penalty_huber_remains_huber(method):
    matrix=(np.ones((3,1)),np.empty((0,1)),np.array([0.,1.,10.]))
    beta=Classo(matrix,0.,typ='R2',meth=method,rho=1.)
    np.testing.assert_allclose(beta,[1.],atol=1e-9)


def test_projection_is_exact_with_redundant_rescaled_rows():
    C=np.array([[1.,1.,1.],[1e-100,1e-100,1e-100],[-1e100,-1e100,-1e100],[0.,0.,0.]])
    P=proj_c(C,3)
    np.testing.assert_allclose(P,P.T,atol=1e-15)
    np.testing.assert_allclose(P@P,P,atol=1e-15)
    np.testing.assert_allclose(np.ones(3)@P,0.,atol=1e-15)
    np.testing.assert_allclose(proj_c(np.empty((0,3)),3),np.eye(3))


def test_unpenalized_fit_retains_weak_design_direction():
    matrix=(np.diag([1.,1e-9]),np.empty((0,2)),np.ones(2))
    np.testing.assert_allclose(unpenalized(matrix),[1.,1e9],rtol=1e-12)


def objective(result,X,y,lam,typ):
    beta=coefficients(result,typ);residual=X@beta[1:]+beta[0]-y
    if typ=='R1':
        loss=residual@residual
    elif typ=='R2':
        loss=np.where(abs(residual)<=1.345,residual**2,2*1.345*abs(residual)-1.345**2).sum()
    elif typ=='R3':
        loss=2*np.sqrt(len(y)/2)*np.linalg.norm(residual)
    else:
        sigma=result[1]
        loss=(sigma*np.where(abs(residual/sigma)<=1.345,(residual/sigma)**2,
                             2*1.345*abs(residual/sigma)-1.345**2).sum()+len(y)*sigma
              if sigma>1e-12 else 2*1.345*abs(residual).sum())
    return loss+lam*np.linalg.norm(beta[1:],1)


@pytest.mark.parametrize('typ,method',[(t,m) for t in ['R1','R2'] for m in ['DR','P-PDS','PF-PDS']]+[('R3','DR'),('R4','DR')])
def test_iterative_solver_matches_path_objective_with_redundant_constraints(typ,method):
    X,C,y=data(typ);C=np.vstack([C,1e-7*C,-3.*C,np.zeros_like(C)])
    kwargs=dict(typ=typ,true_lam=True,intercept=True)
    path=Classo((X,C,y),1.2,meth='Path-Alg',**kwargs)
    iterative=Classo((X,C,y),1.2,meth=method,**kwargs)
    np.testing.assert_allclose(objective(path,X,y,1.2,typ),objective(iterative,X,y,1.2,typ),rtol=3e-5,atol=3e-4)
    assert abs(np.sum(coefficients(iterative,typ)[1:]))<2e-4


@pytest.mark.parametrize('typ,method',[(t,m) for t in ['R1','R2'] for m in ['DR','P-PDS','PF-PDS']]+[('R3','DR'),('R4','DR')])
def test_zero_response_iterative_paths_have_finite_outputs(typ,method):
    matrix=(np.zeros((4,2)),np.empty((0,2)),np.zeros(4))
    with np.errstate(all='raise'):
        single=Classo(matrix,1.2,typ=typ,meth=method,true_lam=True)
        path=pathlasso(matrix,lambdas=[1.2,.2,0.],typ=typ,meth=method,true_lam=True)
    np.testing.assert_allclose(coefficients(single,typ),0.)
    np.testing.assert_allclose(path[0],0.)
    np.testing.assert_allclose(path[1],[1.2,.2,0.])


@pytest.mark.parametrize('typ',TYPES)
@pytest.mark.parametrize('lam',[-1.,float('nan'),float('inf')])
def test_invalid_penalty_rejected_before_dispatch(typ,lam):
    with pytest.raises(ValueError,match='lambda'):
        Classo(data(typ),lam,typ=typ)


@pytest.mark.parametrize('grid',[[],[1.,np.nan],[1.,-.1],[1.,.1,.5]])
def test_invalid_penalty_path_is_rejected(grid):
    with pytest.raises(ValueError,match='lambda'):
        pathlasso(data('R1'),lambdas=grid)


def test_zero_gradient_default_absolute_path_preserves_requested_grid():
    matrix=(np.zeros((4,2)),np.empty((0,2)),np.zeros(4))
    beta,penalties=pathlasso(matrix,typ='R1',true_lam=True)
    np.testing.assert_allclose(beta,0.)
    np.testing.assert_allclose(penalties,np.geomspace(1.,.01,100))


def test_huber_lambda_max_uses_huber_intercept_not_response_mean():
    X=np.array([[1.],[2.],[3.]])
    matrix=(X,np.empty((0,1)),np.array([0.,1.,100.]))
    maximum,beta=Classo(matrix,1.,typ='R2',meth='Path-Alg',rho=1.,
                       intercept=True,get_lambdamax=True)
    assert maximum==pytest.approx(4.)
    np.testing.assert_allclose(beta,[1.,0.],atol=1e-10)
    path=pathlasso(matrix,lambdas=[1.,.2],typ='R2',meth='Path-Alg',rho=1.,intercept=True)
    np.testing.assert_allclose(path[1],[4.,.8],atol=1e-10)


@pytest.mark.parametrize('typ',TYPES)
def test_full_rank_constraints_leave_only_the_intercept(typ):
    X,_,y=data(typ);C=np.eye(X.shape[1])
    result=Classo((X,C,y),1.2,typ=typ,meth='Path-Alg',true_lam=True,intercept=True)
    beta=coefficients(result,typ)
    np.testing.assert_allclose(beta[1:],0.,atol=2e-8)
    assert np.isfinite(beta[0])


@pytest.mark.parametrize('typ',['R3','R4'])
@pytest.mark.parametrize('e',[0.,-1.,float('nan')])
def test_invalid_scale_penalty_is_rejected(typ,e):
    with pytest.raises(ValueError,match='e must'):
        Classo(data(typ),.2,typ=typ,e=e)
    with pytest.raises(ValueError,match='e must'):
        pathlasso(data(typ),typ=typ,e=e)
