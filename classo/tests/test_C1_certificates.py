"""C1 intercept margin tracking and unregularized KKT regressions."""
import numpy as np
import pytest
from classo.path_alg import parameters_for_update, up_cl, solve_path, check_C1_certificate
from scipy.optimize import minimize, LinearConstraint


def sample():
    rng=np.random.default_rng(0)
    X=rng.normal(size=(24,5))
    y=np.where(X[:,0]-X[:,1]+rng.normal(size=len(X))>.1,1,-1)
    return X,np.ones((1,5)),y


def test_C1_margins_follow_current_intercept_after_sample_events():
    X,C,y=sample()
    pb=parameters_for_update((X,C,y),.02,0,'C1',intercept=True)
    changed=False
    for _ in range(200):
        check_C1_certificate(pb)
        np.testing.assert_allclose(pb.r,y*(X@pb.beta+pb.beta0),atol=1e-11)
        if pb.lam<=.02:break
        before=pb.F.copy();up_cl(pb);changed|=not np.array_equal(before,pb.F)
    else:pytest.fail('C1 did not reach target')
    assert changed
    assert not np.allclose(X.mean(0),X[pb.F].mean(0))


@pytest.mark.parametrize('intercept',[False,True])
@pytest.mark.parametrize('constrained',[False,True])
def test_C1_endpoint_matches_independent_epigraph_solve(intercept,constrained):
    X,C,y=sample()
    if not constrained:C=np.empty((0,X.shape[1]))
    out=solve_path((X,C,y),.08,0,0,'C1',intercept=intercept)
    if intercept:
        b0,B,L=out;beta=np.r_[b0[-1],B[-1]]
    else:
        B,L=out;beta=B[-1]
    original=parameters_for_update((X,C,y),.08,0,'C1',intercept=intercept)
    lam=L[-1]*original.lambdamax
    design=np.c_[np.ones(len(X)),X] if intercept else X
    p=X.shape[1];d=design.shape[1]
    select=np.c_[np.zeros(p),np.eye(p)] if intercept else np.eye(p)
    inequalities=np.vstack([np.c_[-select,np.eye(p)],np.c_[select,np.eye(p)]])
    constraints=[LinearConstraint(inequalities,0,np.inf)]
    if len(C):
        E=np.c_[np.zeros(len(C)),C] if intercept else C
        constraints.append(LinearConstraint(np.c_[E,np.zeros((len(C),p))],0,0))

    def objective(z):
        residual=np.maximum(1-y*design.dot(z[:d]),0)
        return ((residual.dot(residual)+lam*z[d:].sum())/len(y),
                np.r_[-2*design.T.dot(y*residual),np.full(p,lam)]/len(y))

    reference=minimize(objective,np.zeros(d+p),jac=True,method="SLSQP",
                       constraints=constraints,options=dict(ftol=1e-12,maxiter=2000))
    assert reference.success,reference.message
    np.testing.assert_allclose(beta,reference.x[:d],atol=3e-6,rtol=3e-6)


def test_C1_certificate_rejects_corrupt_actual_coefficients():
    X,C,y=sample();pb=parameters_for_update((X,C,y),.1,0,'C1',intercept=True)
    check_C1_certificate(pb)
    pb.beta0+=.1
    with pytest.raises(RuntimeError,match='KKT certificate'):
        check_C1_certificate(pb)


def test_C1_certificate_handles_nonunique_constraint_multipliers():
    X,C,y=sample();C=np.vstack([C,2*C])
    b0,B,L=solve_path((X,C,y),.08,0,0,'C1',intercept=True)
    assert len(L)>2
    np.testing.assert_allclose(C@B[-1],0,atol=1e-10)


def test_C1_does_not_skip_nearly_coincident_margin_events():
    rng=np.random.default_rng(0)
    X=rng.normal(size=(24,5))
    y=np.where(X[:,0]-X[:,1]+rng.normal(size=24)>0,1,-1)
    X[-1]=X[0]+1e-6*rng.normal(size=5);y[-1]=y[0]
    X[-2]=X[1]+1e-6*rng.normal(size=5);y[-2]=y[1]
    pb=parameters_for_update((X,np.ones((1,5)),y),.01,0,'C1',intercept=True)
    for _ in range(200):
        check_C1_certificate(pb)
        actual=y*(X@pb.beta+pb.beta0)
        assert not np.any(pb.F & (actual>1+1e-8))
        assert not np.any(~pb.F & (actual<1-1e-8))
        if pb.lam<=.01:break
        up_cl(pb)
    else:pytest.fail('C1 did not reach target')
