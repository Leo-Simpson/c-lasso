"""Numerical equivalence of direct KKT solves and centered scatter updates."""
from types import SimpleNamespace
import numpy as np
import pytest
from classo.path_alg import _C1_system, _C1_inverse, _classification_gram


def test_factorized_KKT_direction_satisfies_original_system():
    rng=np.random.default_rng(62);X=rng.normal(size=(40,18));C=rng.normal(size=(2,18))
    M=np.block([[2*X.T@X,C.T],[C,np.zeros((2,2))]])
    rhs=np.r_[np.sign(rng.normal(size=18)),np.zeros(2)]
    system=_C1_system(M)
    assert hasattr(system,'solve')
    direction=system.solve(rhs)
    np.testing.assert_allclose(M@direction,rhs,atol=1e-12)
    np.testing.assert_allclose(direction,np.linalg.solve(M,rhs),atol=1e-12)


def test_dependent_face_retains_equilibrated_minimum_norm_policy():
    rng=np.random.default_rng(2);X=rng.normal(size=(12,10));X[:,-1]=X[:,0]
    C=np.ones((1,10));M=np.block([[2*X.T@X,C.T],[C,np.zeros((1,1))]])
    system=_C1_system(M)
    assert not hasattr(system,'solve')
    np.testing.assert_array_equal(system,_C1_inverse(M))


def test_large_feature_units_preserve_original_KKT_direction():
    rng=np.random.default_rng(4);X=rng.normal(size=(50,12));C=np.ones((1,12))
    M=np.block([[2*X.T@X,C.T],[C,np.zeros((1,1))]])
    units=np.r_[np.geomspace(1e-6,1e6,12),1.]
    rhs=np.r_[np.sign(rng.normal(size=12)),0.]
    scaled=units[:,None]*M*units[None,:]
    system=_C1_system(scaled)
    direction=system.solve(units*rhs) if hasattr(system,'solve') else system@(units*rhs)
    np.testing.assert_allclose(units*direction,np.linalg.solve(M,rhs),atol=2e-9,rtol=2e-8)


@pytest.mark.parametrize('intercept',[False,True])
def test_row_updates_match_independent_recentered_Gram(intercept):
    rng=np.random.default_rng(9);X=rng.normal(size=(80,40))+rng.normal(size=40)*10
    pb=SimpleNamespace(A=X,intercept=intercept,M=np.zeros((40,40)))
    mask=np.zeros(80,bool)
    # Include empty/singleton cells, additions, removals, and periodic rebuilds.
    for i in range(180):
        row=i if i<80 else int(rng.integers(80))
        mask[row]=not mask[row]
        _classification_gram(pb,mask)
        Z=X[mask]
        if intercept and len(Z):Z=Z-Z.mean(axis=0)
        np.testing.assert_allclose(pb.M,2*Z.T@Z,rtol=2e-10,atol=2e-9)
    _classification_gram(pb,np.zeros(80,bool),rebuild=True)
    np.testing.assert_array_equal(pb.M,0.)
