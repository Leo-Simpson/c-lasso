"""Degenerate entering signs and nearby faces, from student-case diagnosis.

Synthetic fixtures only: no external real-data matrices are needed by CI.
"""
import numpy as np
import pytest
from classo.path_alg import solve_path,parameters_for_update,up_cl,check_C1_certificate
from classo.solve_MC1 import problem_MC1
from classo.tests.test_MC1_binary import reference_MC1


def tree_like(seed):
    rng=np.random.default_rng(seed)
    A=rng.normal(size=(18,8))
    X=np.c_[A,A[:,:4].mean(1),A[:,4:].mean(1),A[:,:2].mean(1),A[:,2:4].mean(1)]
    C=np.ones((1,X.shape[1]))
    y=np.where(A[:,0]-A[:,1]+rng.normal(size=18)>0,1,-1)
    return X,C,y


@pytest.mark.parametrize('seed',[1,5,6,31])
def test_dependent_columns_release_incompatible_zero_faces(seed):
    X,C,y=tree_like(seed)
    a,B,L=solve_path((X,C,y),.005,0,0,'C1',intercept=True)
    assert L[-1]==.005
    maximum=parameters_for_update((X,C,y),.005,0,'C1',intercept=True).lambdamax
    pb=problem_MC1((X,C,y),intercept=True)
    for index in np.unique(np.linspace(0,len(L)-2,3).astype(int)):
        # Interior points verify the affine segments, not just returned knots.
        beta=(np.r_[a[index],B[index]]+np.r_[a[index+1],B[index+1]])/2
        lam=(L[index]+L[index+1])/2*maximum
        ref=reference_MC1(pb,2*lam)[:,1]
        def objective(b):
            h=np.maximum(1-y*(X@b[1:]+b[0]),0.)
            return h@h+lam*abs(b[1:]).sum()
        assert objective(beta)==pytest.approx(objective(ref),rel=2e-7,abs=2e-7)
    p=parameters_for_update((X,C,y),.005,0,'C1',intercept=True)
    for _ in range(300):
        check_C1_certificate(p)
        actual=y*(X@p.beta+p.beta0)
        assert not np.any(p.F&(actual>1+1e-8))
        assert not np.any(~p.F&(actual<1-1e-8))
        if p.lam<=.005:break
        up_cl(p)
    else:pytest.fail('Path failed to reach endpoint')


def quadratic_problem(tiny_coefficient):
    from classo.elastic_net import problem_elastic_net
    H=np.array([[2.,2.],[2.,4.]])
    X=np.linalg.cholesky(H/2).T
    B=np.array([[1.],[tiny_coefficient]])
    y=np.linalg.solve(X.T,(H@B[:,0]+1)/2)
    return problem_elastic_net((X,None,y),typ='R1'),B


def test_small_positive_coordinate_cannot_bind_negative_face():
    from classo.mc1_path_engine import _outgoing_direction
    pb,B=quadratic_problem(5e-9)
    pb.path_info=dict(direction_linear=0,direction_qp=0)
    certificate=-pb.loss_gradient(B)[1]
    D,dt,active,faces,_,_=_outgoing_direction(pb,B,1.,certificate,1e-9)
    assert faces[list(active).index(1)]==[(0,1)]


def test_refinement_keeps_stationary_dual_tight_row_inactive(monkeypatch):
    from classo import mc1_path_engine as engine
    pb,B=quadratic_problem(0.)
    pb.path_info=dict(direction_linear=0,direction_qp=0,certificate_linear=0,certificate_lp=0)
    original=engine._linear_direction
    def slightly_inexact(*args,**kwargs):
        z=original(*args,**kwargs).copy()
        # Mimic a QP error smaller than its stationarity tolerance.
        z[1]+=2e-9;z[3]+=2e-9
        return z
    monkeypatch.setattr(engine,'_linear_direction',slightly_inexact)
    cert=-pb.loss_gradient(B)[1]
    D,dt,active,faces,quadratic,_=engine._outgoing_direction(pb,B,1.,cert,1e-9)
    assert 1 not in active
    step,_,_,residual=engine._segment_end(pb,B,1.,D,dt,active,faces,quadratic,.5,1e-9)
    assert step==pytest.approx(.5)
    assert residual<1e-8
