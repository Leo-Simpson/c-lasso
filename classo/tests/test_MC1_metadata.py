import numpy as np
import pytest
from classo import Classo,pathlasso
from classo.solve_MC1 import problem_MC1
from classo.mc1_path_engine import solve_path_MC1
from classo.tests.test_MC1_binary import reference_MC1


@pytest.mark.parametrize('seed',[2,8])
def test_joint_metadata_matches_independent_epigraph_reference(seed):
    r=np.random.default_rng(seed)
    X=r.normal(size=(27,3));Z=r.normal(size=(27,2));y=np.arange(27)%3
    pb=problem_MC1((X,np.ones((1,3)),y),intercept=True,covariates=Z)
    assert pb.feature_offset==3
    np.testing.assert_allclose(pb.loss_gradient(pb.init)[1][:3],0,atol=2e-8)
    B=Classo((X,np.ones((1,3)),y),.4,typ='MC1',meth='Path-Alg',intercept=True,covariates=Z)
    expected=reference_MC1(pb,.4*pb.lambdamax)
    np.testing.assert_allclose(B,expected,atol=4e-6,rtol=4e-6)
    np.testing.assert_allclose(B[3:].sum(0),0,atol=1e-9)


def test_covariates_are_free_and_excluded_from_feature_budget():
    r=np.random.default_rng(3);X=r.normal(size=(24,4));Z=r.normal(size=(24,2));y=np.arange(24)%3
    pb=problem_MC1((X,np.ones((1,4)),y),intercept=True,covariates=Z)
    B,L=solve_path_MC1(pb,.02,n_active=2)
    assert all(np.count_nonzero(np.max(abs(b[pb.feature_offset:]),axis=1)>1e-7)<=2 for b in B)
    assert np.max(abs(B[0][1:3]))>1e-3
    assert np.all(pb.weights[:3]==0) and np.all(pb.weights[3:]==1)


def test_metadata_only_when_all_microbial_columns_are_zero():
    r=np.random.default_rng(10);Z=r.normal(size=(24,2));y=np.arange(24)%3
    pb=problem_MC1((np.zeros((24,3)),None,y),intercept=True,covariates=Z)
    B,L=solve_path_MC1(pb,.01)
    assert pb.lambdamax==0
    np.testing.assert_allclose(B[0],B[-1])
    np.testing.assert_array_equal(B[-1][3:],0)


def test_redundant_metadata_encoding_is_rejected():
    with pytest.raises(ValueError,match='full column rank'):
        problem_MC1((np.eye(6),None,np.arange(6)%3),intercept=True,covariates=np.ones((6,1)))


def test_binary_metadata_columns_are_antisymmetric():
    r=np.random.default_rng(7);X=r.normal(size=(24,3));Z=r.normal(size=(24,1))
    y=np.where(X[:,0]+Z[:,0]+r.normal(size=24)>0,1,-1)
    B=Classo((X,np.ones((1,3)),y),.2,typ='MC1',meth='Path-Alg',intercept=True,covariates=Z)
    np.testing.assert_allclose(B[:,0],-B[:,1],atol=1e-8)


def test_flat_nuisance_direction_preserves_selected_margin_representative():
    from classo.multiclass_path import face_direction
    X=np.eye(2);C=np.empty((0,2));faces=[[],[(0,1),(1,1)]]
    H=np.array([np.diag([0.,2.])]*2)
    reference=np.array([3.,0.,-4.,0.,0.,0.])
    derivative,_=face_direction(X,C,[0,1],faces,2,hessians=H,weights=np.array([0.,1.]),reference=reference)
    np.testing.assert_allclose(derivative[0],[3.,-4.],atol=1e-10)
    np.testing.assert_allclose(derivative[1],[-.25,-.25],atol=1e-10)
