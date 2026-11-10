"""Independent regression checks for the R1/R2 scalar paths."""
import numpy as np
import pytest
from scipy.optimize import minimize, LinearConstraint
from classo import Classo
from classo.path_alg import solve_path, parameters_for_update, up, check_C1_certificate


def reference(X,C,y,lam,typ,rho,intercept):
    free=int(intercept);Z=np.c_[np.ones(len(y)),X] if free else X
    q,p=Z.shape[1],X.shape[1]
    select=np.c_[np.zeros((p,free)),np.eye(p)]
    epi=np.vstack([np.c_[select,-np.eye(p)],np.c_[-select,-np.eye(p)]])
    constraints=[LinearConstraint(epi,-np.inf,0.)]
    if len(C):constraints.append(LinearConstraint(np.c_[np.zeros((len(C),free)),C,np.zeros((len(C),p))],0.,0.))
    def objective(v):
        r=Z@v[:q]-y
        loss=r*r if typ=='R1' else np.where(abs(r)<=rho,r*r,2*rho*abs(r)-rho*rho)
        score=r if typ=='R1' else np.clip(r,-rho,rho)
        return (loss.sum()+lam*v[q:].sum())/len(y),np.r_[2*Z.T@score,np.full(p,lam)]/len(y)
    fit=minimize(objective,np.zeros(q+p),jac=True,method='SLSQP',constraints=constraints,
                 options=dict(ftol=1e-11,maxiter=3000))
    assert fit.success,fit.message
    return fit.fun*len(y)


def test_R1_small_feature_has_no_hidden_ridge():
    b=Classo((np.array([[.01]]),np.empty((0,1)),np.ones(1)),.5,typ='R1',meth='Path-Alg')
    np.testing.assert_allclose(b,[50.],rtol=1e-10)


def test_R2_empty_quadratic_start_reaches_requested_lambda():
    X=np.array([[1.],[-1.]]);y=np.array([2.,-2.]);C=np.empty((0,1))
    b=Classo((X,C,y),.1,typ='R2',meth='Path-Alg',rho=1.)
    np.testing.assert_allclose(b,[1.9],atol=1e-10)
    assert solve_path((X,C,y),.1,0,1.,'R2')[-1][-1]==.1


@pytest.mark.parametrize('typ',['R1','R2'])
@pytest.mark.parametrize('intercept',[False,True])
@pytest.mark.parametrize('constrained',[False,True])
@pytest.mark.parametrize('dependent',[False,True])
def test_regression_path_interiors_match_independent_objective(typ,intercept,constrained,dependent):
    rng=np.random.default_rng(3)
    X=rng.normal(size=(20,6));y=2*X[:,0]-X[:,1]+rng.normal(size=20);y[0]+=8
    if dependent:X=np.c_[X,X[:,0],X[:,:3].mean(1)]
    # Repeated samples force tied sample events.
    X[-1]=X[0];y[-1]=y[0]
    C=np.ones((1,X.shape[1])) if constrained else np.empty((0,X.shape[1]))
    rho=.8;target=.03
    state=parameters_for_update((X,C,y),target,rho,typ,intercept=intercept)
    result=solve_path((X,C,y),target,0,rho,typ,intercept=intercept)
    if intercept:bias,B,L=result
    else:B,L=result;bias=np.zeros(len(B))
    assert L[-1]==target
    for i in np.unique(np.linspace(0,len(L)-2,3).astype(int)):
        b=(B[i]+B[i+1])/2;b0=(bias[i]+bias[i+1])/2
        lam=(L[i]+L[i+1])/2*state.lambdamax
        r=X@b+b0-y
        loss=r*r if typ=='R1' else np.where(abs(r)<=rho,r*r,2*rho*abs(r)-rho*rho)
        actual=loss.sum()+lam*abs(b).sum()
        assert actual==pytest.approx(reference(X,C,y,lam,typ,rho,intercept),abs=2e-7,rel=2e-7)
    for _ in range(300):
        check_C1_certificate(state)
        np.testing.assert_allclose(state.r,X@state.beta+state.beta0-y,atol=1e-10)
        if state.lam<=target:break
        up(state)
    else:pytest.fail('Regression path did not reach its endpoint')


@pytest.mark.parametrize('typ',['R1','R2'])
def test_regression_zero_lambda_max_returns_constant_path(typ):
    X=np.arange(12.).reshape(6,2)
    bias,B,L=solve_path((X,np.empty((0,2)),np.full(6,2.)),.01,0,1.,typ,intercept=True)
    assert L==[1.,.01]
    np.testing.assert_allclose(B,0.)
    np.testing.assert_allclose(bias,2.)


@pytest.mark.parametrize("scale", [1., 1e-8, 1e-12, 1e-100])
def test_zero_crossing_does_not_depend_on_response_units(scale):
    """An active coefficient must leave at relative step .1 at every scale."""
    from types import SimpleNamespace
    from classo.path_alg import up_LS
    gram = np.array([[1., 2.], [2., 5.]])
    state = SimpleNamespace(
        formulation="R3", lambdamax=scale, lamin=0., M=gram,
        C=np.empty((0, 2)), eps_L2=0., number_act=2, idr=[],
        Xt=np.linalg.inv(gram), activity=[True, True],
        beta=scale*np.array([1., .1]), s=np.ones(2), lam=1.)
    up_LS(state)
    assert state.lam == pytest.approx(.9)
    assert state.activity == [True, False]
    np.testing.assert_allclose(state.beta/scale, [1.3, 0.], atol=1e-14)
