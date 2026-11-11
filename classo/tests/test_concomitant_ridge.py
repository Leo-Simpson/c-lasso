"""Concomitant ridge works with SciPy alone and preserves tree constraints."""
import builtins
import numpy as np
import pytest
from classo import Classo, pathlasso
from classo.tests.test_elastic_net import data, reference


@pytest.mark.parametrize('typ',['R3','R4'])
def test_no_conic_dependency(monkeypatch,typ):
    original=builtins.__import__
    def blocked(name,*args,**kwargs):
        if name.split('.')[0] in ('cvxpy','clarabel','scs'):
            raise AssertionError('Production solver imported a conic package')
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',blocked)
    X,C,y=data(37)
    B,L,S=pathlasso((X,C,y),lambdas=[1.,.5,.2,0.],typ=typ,ridge=2.,
                    intercept=True,return_sigm=True)
    assert len(B)==len(L)==len(S)==4
    np.testing.assert_allclose(B[:,1:]@C.T,0,atol=1e-7)
    assert np.isfinite(B).all() and np.min(S)>=0


@pytest.mark.parametrize('typ',['R3','R4'])
def test_multiple_redundant_constraints(typ):
    pytest.importorskip('cvxpy')
    X,_,y=data(19)
    C=np.array([[1,1,1,1,1],[1,-1,0,0,0],[2,2,2,2,2],[0,0,0,0,0]],float)
    w=np.array([.6,2.,1.3,.8,1.])
    maximum,B,sigma=Classo((X,C,y),.37,typ=typ,ridge=3.,w=w,
                          intercept=True,get_lambdamax=True)
    expected=reference(X,C,y,typ,.37*maximum,3.,w)
    np.testing.assert_allclose(B,expected,atol=1e-4)
    np.testing.assert_allclose(C@B[1:],0,atol=1e-8)
    r=X@B[1:]+B[0]-y
    if typ=='R3':
        assert sigma==pytest.approx(np.sqrt((r@r+1.5*(B[1:]@B[1:]))/(len(y)/2)),abs=1e-8)
    else:
        check=np.minimum((r/sigma)**2,1.345**2).sum()+1.5*(B[1:]@B[1:])/sigma**2
        assert check==pytest.approx(len(y),abs=1e-5)


def test_sparse_quadratic_engine_matches_dense():
    from scipy import sparse
    from classo import problem_elastic_net,solve_path_elastic_net
    X,C,y=data()
    dense=problem_elastic_net((X,C,y),ridge=2.,intercept=True)
    csr=problem_elastic_net((sparse.csr_matrix(X),C,y),ridge=2.,intercept=True)
    B,L=solve_path_elastic_net(dense,lamin=.05)
    E,T=solve_path_elastic_net(csr,lamin=.05)
    np.testing.assert_allclose(L,T,atol=1e-9)
    np.testing.assert_allclose(B,E,atol=1e-8)


def test_r3_sparse_budget_returns_only_achieved_values():
    X,C,y=data(4)
    B,L,S=pathlasso((X,C,y),lambdas=np.geomspace(1.,.001,30),typ='R3',
                   ridge=2.,intercept=True,n_active=2,return_sigm=True)
    assert len(B)==len(L)==len(S)
    assert L[-1]>L[0]*.001
    assert np.all(np.diff(L)<0)
    assert np.max(np.sum(abs(B[:,1:])>1e-7,axis=1))<=2


@pytest.mark.parametrize('fraction',[.999,.1,.01])
def test_r4_zero_scale_threshold_and_inner_fits(fraction):
    pytest.importorskip('cvxpy')
    X,C,y=data(12)
    maximum,B,sigma=Classo((X,C,y),fraction,typ='R4',ridge=2.,
        intercept=True,rho=.5,e=30.,get_lambdamax=True)
    expected=reference(X,C,y,'R4',fraction*maximum,2.,np.ones(X.shape[1]),rho=.5,e=30.)
    np.testing.assert_allclose(B,expected,atol=1e-4)
    assert sigma>0
