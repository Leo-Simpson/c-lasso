"""Constraint invariance and weak-curvature checks for the scalar ridge branch."""
import numpy as np
import pytest
from classo import Classo
from classo._homotopy_geometry import face_direction


@pytest.mark.parametrize('typ', ['R1','R2','C1','C2'])
@pytest.mark.parametrize('intercept', [False,True])
def test_ridge_constraint_row_units_and_redundancy(typ,intercept):
    rng=np.random.default_rng(42);X=rng.normal(size=(16,4))
    y=X[:,0]-X[:,1]+.3*rng.normal(size=16)
    if typ.startswith('C'): y=np.where(y>0,1,-1)
    C=np.ones((1,4));other=np.vstack([C,1e-8*C,-1e8*C,np.zeros_like(C)])
    kwargs=dict(typ=typ,ridge=2.,intercept=intercept,true_lam=True)
    a=Classo((X,C,y),1.2,**kwargs);b=Classo((X,other,y),1.2,**kwargs)
    np.testing.assert_allclose(X@a[int(intercept):],X@b[int(intercept):],rtol=2e-6,atol=2e-6)
    np.testing.assert_allclose(C@b[int(intercept):],0,atol=1e-7)


def test_shared_direction_retains_weak_curvature():
    X=np.diag([1.,1e-7]);C=np.empty((0,2))
    db,dt=face_direction(X,C,[0,1],[[(0,1)],[(0,1)]],1,
                         weights=np.array([1.,1e-7]))
    np.testing.assert_allclose(db[:,0],[-1.,-1e7],rtol=1e-10)
    np.testing.assert_allclose(dt,db[:,0],rtol=1e-10)
