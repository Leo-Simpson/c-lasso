"""Transfer scalar solver numerical invariants to the multiclass development engine."""
import numpy as np
import pytest
from classo import Classo,pathlasso
from classo.mc1_path_engine import _linear_direction


def data():
    rng=np.random.default_rng(42);X=rng.normal(size=(16,4))
    score=X[:,0]-X[:,1]+.3*rng.normal(size=16)
    y=np.digitize(score,np.quantile(score,[1/3,2/3]))
    return X,np.ones((1,4)),y


@pytest.mark.parametrize('typ',['MC1','MC2'])
@pytest.mark.parametrize('ridge',[0.,2.])
@pytest.mark.parametrize('intercept',[False,True])
def test_constraint_row_units_and_redundancy(typ,ridge,intercept):
    X,C,y=data();other=np.vstack([C,1e-8*C,-1e8*C,np.zeros_like(C)])
    kwargs=dict(meth="Path-Alg",typ=typ,ridge=ridge,intercept=intercept,true_lam=True)
    a=Classo((X,C,y),1.2,**kwargs);b=Classo((X,other,y),1.2,**kwargs)
    np.testing.assert_allclose(X@a[int(intercept):],X@b[int(intercept):],rtol=2e-6,atol=2e-6)
    np.testing.assert_allclose(C@b[int(intercept):],0.,atol=1e-7)


@pytest.mark.parametrize('typ',['MC1','MC2'])
def test_weighted_feature_units_and_path_parity(typ):
    X,C,y=data();units=np.array([.2,2.,.5,3.])
    kwargs=dict(meth="Path-Alg",typ=typ,intercept=True,true_lam=True)
    a=Classo((X,C,y),1.2,**kwargs)
    b=Classo((X*units,C*units,y),1.2,w=units,**kwargs)
    np.testing.assert_allclose(a,np.r_[b[:1],units[:,None]*b[1:]],atol=2e-6,rtol=2e-6)
    path=pathlasso((X,C,y),lambdas=[2.,1.2],**kwargs)
    np.testing.assert_allclose(a,path[0][-1],atol=2e-6,rtol=2e-6)


def test_weak_curvature_survives_directional_fast_solve():
    design=np.diag([1.,1e-7]);N=np.vstack([np.eye(2),np.eye(2)])
    weights=np.array([2.,2e-7])
    def objective(z):
        residual=design@z[:2]
        return residual@residual-weights@z[2:],np.r_[2*design.T@residual,-weights]
    result=_linear_direction(N,np.empty((0,2)),design,np.ones((2,1)),
                             np.ones((2,1),bool),np.zeros((2,1),bool),weights,objective,1e-8)
    assert result is not None
    np.testing.assert_allclose(result,[1.,1e7,1.,1e7],rtol=1e-12)


def test_fixed_face_retains_weak_curvature_without_a_ridge():
    from classo.multiclass_path import face_direction
    X=np.diag([1.,1e-7]);C=np.empty((0,2))
    db,dt=face_direction(X,C,[0,1],[[(0,1)],[(0,1)]],1,
                         weights=np.array([1.,1e-7]))
    np.testing.assert_allclose(db[:,0],[-1.,-1e7],rtol=1e-10)
    np.testing.assert_allclose(dt,db[:,0],rtol=1e-10)
