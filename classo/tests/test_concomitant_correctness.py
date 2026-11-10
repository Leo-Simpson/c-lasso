"""Concomitant objectives, scale units, boundary cases and path dispatch."""
import numpy as np
import pytest
from scipy.optimize import minimize,LinearConstraint
from classo import Classo,pathlasso
from classo.solve_R3 import root_2,prox_phi_1
from classo.solve_R4 import find_sigmax,prox_phi_i,problem_R4,pathlasso_R4


def data(constrained=True):
    rng=np.random.default_rng(5);X=rng.normal(size=(12,3))
    y=3+X[:,0]-X[:,1]+rng.normal(size=12);y[0]+=4
    C=np.ones((1,3)) if constrained else np.empty((0,3))
    return X,C,y


def reference(X,C,y,lam,typ,e,rho,intercept):
    free=int(intercept);Z=np.c_[np.ones(len(y)),X] if free else X
    q,p=Z.shape[1],X.shape[1];selection=np.c_[np.zeros((p,free)),np.eye(p)]
    # Variables [coefficients, sigma (R4 only), absolute coefficients].
    scale_dim=int(typ=='R4');dim=q+scale_dim
    epigraph=np.vstack([np.c_[selection,np.zeros((p,scale_dim)),-np.eye(p)],
                       np.c_[-selection,np.zeros((p,scale_dim)),-np.eye(p)]])
    constraints=[LinearConstraint(epigraph,-np.inf,0.)]
    if len(C):constraints.append(LinearConstraint(np.c_[np.zeros((len(C),free)),C,np.zeros((len(C),scale_dim+p))],0.,0.))
    def objective(v):
        r=Z@v[:q]-y
        if typ=='R3':
            norm=np.linalg.norm(r);loss=2*np.sqrt(e)*norm
            grad=2*np.sqrt(e)*Z.T@r/norm if norm else np.zeros(q)
        else:
            sigma=v[q]
            if sigma>0:
                z=r/sigma;clip=np.clip(z,-rho,rho)
                loss=sigma*np.where(abs(z)<=rho,z*z,2*rho*abs(z)-rho*rho).sum()+e*sigma
            else:
                clip=rho*np.sign(r);loss=2*rho*abs(r).sum()
            grad=np.r_[2*Z.T@clip,e-clip@clip]
        return (loss+lam*v[dim:].sum())/len(y),np.r_[grad,np.full(p,lam)]/len(y)
    start=np.zeros(dim+p)
    if scale_dim:start[q]=max(1.,np.std(y))
    bounds=[(None,None)]*q+([(0.,None)] if scale_dim else [])+[(0.,None)]*p
    fit=minimize(objective,start,jac=True,bounds=bounds,method='SLSQP',constraints=constraints,
                 options=dict(ftol=1e-12,maxiter=3000))
    assert fit.success,fit.message
    return fit.fun*len(y)


def objective(X,y,beta,sigma,typ,e,rho,intercept,lam):
    b=beta[int(intercept):];r=X@b+(beta[0] if intercept else 0.)-y
    if typ=='R3':
        np.testing.assert_allclose(sigma,np.linalg.norm(r)/np.sqrt(e),atol=2e-4,rtol=2e-4)
        loss=2*np.sqrt(e)*np.linalg.norm(r)
    elif sigma>1e-8:
        z=r/sigma;loss=sigma*np.where(abs(z)<=rho,z*z,2*rho*abs(z)-rho*rho).sum()+e*sigma
        assert abs(e-np.clip(z,-rho,rho)@np.clip(z,-rho,rho))<3e-3*max(1.,e)
    else:loss=2*rho*abs(r).sum()
    assert sigma>=-1e-8
    return loss+lam*abs(b).sum()


@pytest.mark.parametrize('typ',['R3','R4'])
@pytest.mark.parametrize('method',['Path-Alg','DR'])
@pytest.mark.parametrize('intercept',[False,True])
@pytest.mark.parametrize('constrained',[False,True])
@pytest.mark.parametrize('custom',[False,True])
def test_joint_objective_against_independent_optimizer(typ,method,intercept,constrained,custom):
    X,C,y=data(constrained);e=9. if custom else (6. if typ=='R3' else 12.)
    lam=1.2;rho=1.345
    beta,sigma=Classo((X,C,y),lam,typ=typ,meth=method,intercept=intercept,true_lam=True,e=e,rho=rho)
    actual=objective(X,y,beta,sigma,typ,e,rho,intercept,lam)
    expected=reference(X,C,y,lam,typ,e,rho,intercept)
    assert actual==pytest.approx(expected,abs=3e-4,rel=3e-5)
    np.testing.assert_allclose(C@beta[int(intercept):],0.,atol=2e-4)


@pytest.mark.parametrize('typ',['R3','R4'])
@pytest.mark.parametrize('method',['Path-Alg','DR'])
def test_path_scale_matches_single_fits_in_original_units(typ,method):
    X,C,y=data();fractions=[1.,.3,.1];e=9.
    B,L,S=pathlasso((X,C,y),lambdas=fractions,typ=typ,meth=method,e=e,intercept=True,return_sigm=True)
    for b,lam,sigma in zip(B,L,S):
        b1,s1=Classo((X,C,y),lam,typ=typ,meth=method,e=e,intercept=True,true_lam=True)
        np.testing.assert_allclose(b,b1,atol=3e-4,rtol=3e-4)
        np.testing.assert_allclose(sigma,s1,atol=3e-4,rtol=3e-4)


@pytest.mark.parametrize('typ',['R3','R4'])
@pytest.mark.parametrize('intercept',[False,True])
def test_zero_response_and_zero_scale(typ,intercept):
    X,C,_=data();y=np.full(len(X),2.) if intercept else np.zeros(len(X))
    beta,sigma=Classo((X,C,y),.3,typ=typ,meth='Path-Alg',intercept=intercept)
    np.testing.assert_allclose(beta,np.r_[2.,np.zeros(3)] if intercept else np.zeros(3),atol=1e-12)
    assert sigma==0.
    B,L,S=pathlasso((X,C,y),lambdas=[1.,.2,0.],true_lam=True,typ=typ,meth='Path-Alg',intercept=intercept,return_sigm=True)
    np.testing.assert_allclose(B,np.tile(beta,(3,1)),atol=1e-12)
    np.testing.assert_allclose(S,0.)
    np.testing.assert_allclose(L,[1.,.2,0.])


def test_linear_interpolation_root_has_correct_sign():
    assert root_2(0.,1.,-1.)==pytest.approx(.5)


def test_zero_residual_prox_preserves_positive_scale():
    sigma,r,_=prox_phi_1(3.,np.zeros(2),1.,0.)
    assert sigma==2.5;np.testing.assert_array_equal(r,0.)
    assert prox_phi_i(3.,0.,1.,0.,1.345)[:2]==(2.5,0.)


def test_scale_initializer_treats_negative_outliers_symmetrically():
    y=np.array([-100.,1.,1.]);s=find_sigmax(y,1.345,3.)
    assert s==pytest.approx(find_sigmax(-y,1.345,3.))
    assert np.sum(np.clip(y/s,-1.345,1.345)**2)==pytest.approx(3.)


@pytest.mark.parametrize('rho',[.8,1.])
def test_R4_closed_zero_scale_boundary(rho):
    X=np.zeros((3,1));y=np.array([0.,1.,10.]);C=np.empty((0,1))
    b,sigma=Classo((X,C,y),.3,typ='R4',meth='Path-Alg',rho=rho,intercept=True)
    np.testing.assert_allclose(b,[1.,0.],atol=1e-12)
    assert sigma==0.


def test_R3_exact_fit_without_a_ridge():
    X=np.array([[.01],[.02]]);y=np.array([1.,2.]);C=np.empty((0,1))
    b,sigma=Classo((X,C,y),.5,typ='R3',meth='Path-Alg')
    np.testing.assert_allclose(b,[100.],atol=1e-8)
    assert sigma<1e-10


def test_R4_path_respects_requested_method():
    X,C,y=data();pb=problem_R4((X,C,y),'Path-Alg',1.345,intercept=True)
    pathlasso_R4(pb,[1.,.3])
    assert pb.type=='Path-Alg'


def test_cubic_root_reaches_requested_precision():
    from classo.solve_R3 import calc_Newton
    for a,b in [(1.,2.),(-3.,.01),(0.,1e-8),(100.,1.)]:
        root=calc_Newton(a,b,0.)
        assert abs(root**3+a*root-b)<1e-12


@pytest.mark.parametrize('typ',['R3','R4'])
@pytest.mark.parametrize('method',['Path-Alg','DR'])
def test_redundant_constraints(typ,method):
    X,C,y=data()
    redundant=np.vstack([C,2*C,np.zeros_like(C)])
    b,s=Classo((X,redundant,y),.2,typ=typ,meth=method)
    assert np.linalg.norm(C@b)<2e-4
    assert s>=0.
