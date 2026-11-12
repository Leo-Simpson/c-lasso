import sys,pathlib
import numpy as np
import pytest
from classo.solve_Q1 import quantile_path,loss
from classo._classification_path import _gradient

@pytest.mark.parametrize('tau',[.2,.5,.8])
def test_value_gradient_and_thresholds(tau):
    k=.7;D=13.;rng=np.random.default_rng(1);Z=rng.normal(size=(12,4));y=rng.normal(size=12);b=rng.normal(size=4)
    scale=np.sqrt(2*k*D);X=np.vstack([-Z,Z])/scale;offset=np.r_[1+y/scale,1-y/scale]
    lower=1-np.r_[np.full(12,(1-tau)*k),np.full(12,tau*k)]/scale
    expected=-Z.T@np.clip((y-Z@b)/k,tau-1,tau)/D
    np.testing.assert_allclose(_gradient(X,np.ones(24),b,lower,offset),expected,atol=1e-14)
    for boundary in (-(1-tau)*k,tau*k):
        vals=loss(np.array([boundary-1e-8,boundary,boundary+1e-8]),tau,k)
        assert np.max(abs(np.diff(vals)))<1e-8

@pytest.mark.parametrize('tau',[.2,.5,.8])
def test_events_match_warm_and_zero_sum(tau):
    rng=np.random.default_rng(0);X=rng.normal(size=(40,9));y=1.3+X[:,0]-X[:,2]+rng.standard_t(3,size=40)*.4
    lams=np.geomspace(.5,.01,14)
    a=quantile_path(X,y,lams,tau,.8,method='warm');b=quantile_path(X,y,lams,tau,.8,method='events')
    np.testing.assert_allclose(a['coefficients'],b['coefficients'],atol=2e-5)
    assert np.max(abs(b['coefficients'][:,1:].sum(1)))<1e-9
    assert max(b['kkt'])<2e-6

@pytest.mark.parametrize('kwargs',[dict(tau=0),dict(k=-1),dict(divisor=0),dict(method='bad')])
def test_rejects_invalid_options(kwargs):
    with pytest.raises(ValueError):quantile_path(np.eye(3),np.ones(3),[.1],**kwargs)

def test_zero_lambda_constant_response_and_full_constraint():
    fit=quantile_path(np.eye(3),np.full(3,2.),[.1,0.],C=np.eye(3))
    np.testing.assert_allclose(fit['coefficients'],[[2,0,0,0],[2,0,0,0]],atol=1e-9)

@pytest.mark.parametrize('tau',[.2,.5,.8])
@pytest.mark.parametrize('intercept',[False,True])
def test_Q1_compact_absolute_relative_and_weights(tau,intercept):
    from classo import Classo,pathlasso
    rng=np.random.default_rng(21);X=rng.normal(size=(25,5));C=np.ones((1,5));y=X[:,0]-X[:,1]+rng.normal(size=25)
    w=np.array([1.,2.,.7,1.3,1.])
    kw=dict(typ='Q1',tau=tau,quantile_k=.8,intercept=intercept,w=w,meth='SLSQP')
    maximum,b=Classo((X,C,y),.15,get_lambdamax=True,**kw)
    absolute=Classo((X,C,y),.15*maximum,true_lam=True,**kw)
    np.testing.assert_allclose(b,absolute,atol=1e-8)
    B,L=pathlasso((X,C,y),lambdas=[1.,.15,0.],**kw)
    np.testing.assert_allclose(B[1],b,atol=3e-5)
    np.testing.assert_allclose(np.array(L),np.array([1.,.15,0.])*maximum)
    np.testing.assert_allclose(B[:,int(intercept):].sum(1),0.,atol=1e-8)
    np.testing.assert_allclose(B[0,int(intercept):],0.,atol=1e-7)
    direct=quantile_path(X/w,y,[.15*maximum],tau,.8,C/w,intercept,method='warm')['coefficients'][0]
    np.testing.assert_allclose(b,direct/(np.r_[1.,w] if intercept else w),atol=1e-8)


def test_Q1_zero_maximum_and_unsupported_budget():
    from classo import Classo,pathlasso
    X=np.eye(3);C=np.eye(3);y=np.full(3,2.)
    maximum,b=Classo((X,C,y),.2,typ='Q1',intercept=True,get_lambdamax=True,true_lam=True)
    assert maximum==0.
    np.testing.assert_allclose(b,[2,0,0,0],atol=1e-8)
    with pytest.raises(ValueError):pathlasso((X,C,y),typ='Q1',meth='DR',n_active=2)

@pytest.mark.parametrize('tau',[.2,.5,.8])
def test_DR_matches_convex_warm(tau):
    rng=np.random.default_rng(31);X=rng.normal(size=(35,8));y=.7+X[:,0]-X[:,1]+rng.normal(size=35)*.4
    dr=quantile_path(X,y,[.2,.05,.01],tau,.9,method='DR')
    warm=quantile_path(X,y,[.2,.05,.01],tau,.9,method='warm')
    Z=np.c_[np.ones(len(y)),X]
    np.testing.assert_allclose(Z@dr['coefficients'].T,Z@warm['coefficients'].T,atol=5e-5)
    assert max(dr['kkt'])<2e-6
    assert dr['iterations']>0


def test_event_path_budget_starts_at_zero_without_warm_optimizer(monkeypatch):
    import classo.solve_Q1 as module
    monkeypatch.setattr(module,'minimize',lambda *a,**k:(_ for _ in ()).throw(AssertionError('warm solver called')))
    rng=np.random.default_rng(14);X=rng.normal(size=(60,200));y=X[:,0]-X[:,1]+rng.normal(size=60)*.5
    fit=quantile_path(X,y,[1.,0.],.5,.8,method='events',n_active=6)
    assert fit['stop_reason']=='n_active'
    assert fit['evaluated_lambdas'][-1]>0
    assert max(np.count_nonzero(abs(b[1:])>1e-9) for b in fit['coefficients'])<=6
    assert fit['iterations']==0
    assert max(fit['kkt'])<2e-6


def test_large_lambda_shortcut_avoids_all_optimizers(monkeypatch):
    import classo.solve_Q1 as module
    monkeypatch.setattr(module,'minimize',lambda *a,**k:(_ for _ in ()).throw(AssertionError('solver called')))
    X=np.eye(5);y=np.arange(5.)
    for method in ['events','DR']:
        fit=quantile_path(X,y,[100.],method=method)
        assert fit['iterations']==0 and fit['events']==[]
        np.testing.assert_allclose(fit['coefficients'][0,1:],0.)


def test_no_artificial_residual_zero_knots():
    X=np.ones((2,1));y=np.array([.2,1.])
    fit=quantile_path(X,y,[0.],.5,10.,C=np.empty((0,1)),intercept=False,method='events')
    assert len(fit['events'])==1
    np.testing.assert_allclose(fit['coefficients'],[[.6]],atol=1e-10)


def test_dense_asymmetric_degenerate_optimal_face_regression():
    rng=np.random.default_rng(7);X=rng.normal(size=(24,30));y=1.3+X[:,0]-X[:,2]+rng.standard_t(3,size=24)*.4
    fit=quantile_path(X,y,np.geomspace(.5,.01,14),.2,.8,np.ones((1,30)),False,method='events')
    warm=quantile_path(X,y,[.01],.2,.8,np.ones((1,30)),False,method='warm')
    f=lambda b:loss(y-X@b,.2,.8).mean()+.01*abs(b).sum()
    assert abs(f(fit['coefficients'][-1])-f(warm['coefficients'][0]))<1e-8
    assert max(fit['kkt'])<1e-8
    assert any(e['kind']=='flat-pivot' for e in fit['events'])


def test_DR_reuses_R1_loop_and_matrix_free_resolvents(monkeypatch):
    import classo._q1_dr as adapter
    from classo.solve_R1 import QQ
    original=adapter.Classo_R1;calls=[]
    def wrapped(pb,lam):
        assert pb.matrix_free and pb.AtA is None
        calls.append(pb)
        return original(pb,lam)
    monkeypatch.setattr(adapter,'Classo_R1',wrapped)
    rng=np.random.default_rng(93);X=rng.normal(size=(25,6));y=X[:,0]-X[:,1]+rng.normal(size=25)*.4
    quantile_path(X,y,[.1,.05],.2,.8,method='DR')
    assert len(calls)==2 and calls[0] is calls[1]
    assert hasattr(calls[0],'dr_resolvents')
    A=rng.normal(size=(5,8));v=rng.normal(size=8)
    dense=QQ(.7,A);operator=QQ(.7,A,matrix_free=True)
    np.testing.assert_allclose(operator[0]@A,dense[0]@A,atol=1e-12)
    np.testing.assert_allclose(operator[1]@v,dense[1]@v,atol=1e-12)


@pytest.mark.parametrize('method',['P-PDS','PF-PDS'])
@pytest.mark.parametrize('tau',[.2,.5,.8])
@pytest.mark.parametrize('intercept',[False,True])
def test_PDS_matches_warm_with_general_constraints_and_zero_lambda(method,tau,intercept):
    rng=np.random.default_rng(107);X=rng.normal(size=(28,6));y=.4+X[:,0]-X[:,1]+rng.normal(size=28)*.4
    C=np.array([[1.,1.,1.,1.,1.,1.],[0.,1.,-1.,0.,0.,0.]])
    lams=[.15,.04,0.]
    fit=quantile_path(X,y,lams,tau,.9,C,intercept,method=method)
    reference=quantile_path(X,y,lams,tau,.9,C,intercept,method='warm')
    Z=np.c_[np.ones(len(y)),X] if intercept else X
    for b,r,lam in zip(fit['coefficients'],reference['coefficients'],lams):
        obj=lambda v:loss(y-Z@v,tau,.9).mean()+lam*abs(v[int(intercept):]).sum()
        assert abs(obj(b)-obj(r))<1e-7
        np.testing.assert_allclose(C@b[int(intercept):],0.,atol=1e-7)
    assert max(fit['kkt'])<=1e-8
    assert fit['iterations']>0


@pytest.mark.parametrize('method',['P-PDS','PF-PDS'])
def test_PDS_compact_interfaces_weights_and_R1_reuse(method,monkeypatch):
    from classo import Classo,pathlasso
    import classo._q1_dr as adapter
    original=adapter.Classo_R1;calls=[]
    def wrapped(pb,lam):
        assert pb.type==method and pb.AtA.shape[0]==12+6+1
        assert not isinstance(pb.AtA,np.ndarray)
        calls.append(pb)
        return original(pb,lam)
    monkeypatch.setattr(adapter,'Classo_R1',wrapped)
    rng=np.random.default_rng(49);X=rng.normal(size=(12,6));y=.8+X[:,0]-X[:,1]+rng.normal(size=12)*.3
    C=np.ones((1,6));w=np.array([1.,1.3,.8,1.1,.7,1.2])
    options=dict(typ='Q1',meth=method,tau=.2,quantile_k=.8,intercept=True,w=w)
    maximum,b=Classo((X,C,y),.3,get_lambdamax=True,**options)
    B,L=pathlasso((X,C,y),lambdas=[1.,.3,.1],**options)
    np.testing.assert_allclose(B[1],b,atol=2e-5)
    np.testing.assert_allclose(L,[maximum,.3*maximum,.1*maximum])
    expected=Classo((X,C,y),.3*maximum,typ='Q1',meth='SLSQP',tau=.2,quantile_k=.8,intercept=True,w=w,true_lam=True)
    np.testing.assert_allclose(b,expected,atol=2e-5)
    assert len(calls)>=3 and calls[-1] is calls[-2] and calls[0] is not calls[-1]


@pytest.mark.parametrize('method',['P-PDS','PF-PDS'])
def test_PDS_empty_constraints_limit_and_support_budget(method):
    rng=np.random.default_rng(61);X=rng.normal(size=(20,5));y=X[:,0]+rng.normal(size=20)*.2
    fit=quantile_path(X,y,[.08],.8,.7,C=np.empty((0,5)),intercept=False,method=method)
    assert max(fit['kkt'])<=1e-8
    with pytest.raises(ValueError,match='did not converge'):
        quantile_path(X,y,[.08],.8,.7,method=method,pds_max_iter=1)
    with pytest.raises(ValueError,match='support stopping'):
        quantile_path(X,y,[.08],method=method,n_active=2)
