"""Independent objective checks for fixed-ridge paths and perspective fits."""
import numpy as np
import pytest
from classo import Classo, pathlasso
from classo.elastic_net import problem_elastic_net
from classo._quadratic_homotopy import solve_path_quadratic


def data(seed=37):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(24,5)); X[:,4] = X[:,3]  # intentional collinearity
    y = X@np.array([1.8,-1.8,.5,-.5,0])+rng.normal(size=24)
    return X,np.ones((1,5)),y


def reference(X,C,y,typ,lam,ridge,w,intercept=True,rho=1.345,lower=-1.,e=None):
    cp = pytest.importorskip('cvxpy')
    n,p = X.shape
    multi = typ.startswith('MC')
    K = len(np.unique(y)) if multi else 1
    B = cp.Variable((p,K)); b = cp.Variable((1,K)) if intercept else np.zeros((1,K))
    score = X@B+np.ones((n,1))@b
    constraints = [C@B == 0] if len(C) else []
    if typ.startswith('R'):
        residual = score[:,0]-y
        if typ == 'R1': loss = cp.sum_squares(residual)
        elif typ == 'R2': loss = cp.sum(cp.huber(residual,rho))
        else:
            sigma = cp.Variable(nonneg=True); t = cp.Variable(nonneg=True)
            if typ == 'R4':
                u=cp.Variable(n); core=residual-u; tail=2*rho*cp.norm1(u)
            else: core=residual; tail=0
            scale_weight=e if e is not None else n/2 if typ=='R3' else n
            augmented_core=cp.hstack([2*core/np.sqrt(scale_weight),
                                      np.sqrt(2*ridge/scale_weight)*B[:,0],t-sigma])
            constraints += [cp.SOC(t+sigma,augmented_core)]
            loss=(e if e is not None else n/2 if typ == 'R3' else n)*(t+sigma)+tail
    else:
        T = 2*np.eye(K)[np.unique(y,return_inverse=True)[1]]-1 if multi else y[:,None]
        margin=cp.multiply(T,score)
        if typ in ('C1','MC1'): loss=cp.sum_squares(cp.pos(1-margin))
        else:
            # h² times Huber(pos(1-margin)/h) equals the Huberized hinge.
            h=1-lower
            loss=cp.sum(cp.huber(cp.pos(1-margin),h))
    penalty=cp.sum(cp.multiply(w,cp.max(cp.abs(B),axis=1))) if multi else w@cp.abs(B[:,0])
    outside_ridge=0 if typ in ('R3','R4') else .5*ridge*cp.sum_squares(B)
    problem=cp.Problem(cp.Minimize(loss+lam*penalty+outside_ridge),constraints)
    problem.solve(solver='CLARABEL',tol_gap_abs=1e-10,tol_gap_rel=1e-10,tol_feas=1e-10)
    if problem.status != 'optimal':
        problem.solve(solver='SCS',eps=1e-8,max_iters=200000)
    assert problem.status == 'optimal'
    out=np.vstack([b.value,B.value]) if intercept else B.value
    return out if multi else out[:,0]


@pytest.mark.parametrize('typ',['R1','R2','C1','C2','R3','R4'])
def test_independent_constrained_optimum(typ):
    pytest.importorskip('cvxpy')
    X,C,y=data(); w=np.array([.4,1.,2.,.7,1.3])
    if typ.startswith('C'): y=np.where(y>0,1,-1)
    if typ.startswith('MC'): y=np.digitize(y,[-1,1])
    ridge=2.3
    maximum,beta,*scale=Classo((X,C,y),.25,typ=typ,ridge=ridge,w=w,intercept=True,get_lambdamax=True)
    expected=reference(X,C,y,typ,.25*maximum,ridge,w)
    np.testing.assert_allclose(beta,expected,atol=8e-5,rtol=8e-5)
    np.testing.assert_allclose(C@beta[1:],0,atol=1e-7)
    if typ in ('R3','R4'): assert scale[0]>=0


@pytest.mark.parametrize('typ',['R1','R2','C1','C2'])
def test_segment_interiors_are_exact(typ):
    X,C,y=data(8)
    if typ.startswith('C'): y=np.where(y>0,1,-1)
    if typ.startswith('MC'): y=np.digitize(y,[-1,1])
    pb=problem_elastic_net((X,C,y),typ,ridge=3.,intercept=True)
    B,L=solve_path_quadratic(pb,.12)
    assert pb.path_info['completed']
    assert max(pb.path_info['residuals'],default=0)<1e-5
    # Independently solve interiors, not merely the knots used by the engine.
    indices=np.unique(np.linspace(0,len(L)-2,min(4,len(L)-1)).astype(int))
    for i in indices:
        mid=(L[i]+L[i+1])/2
        expected=reference(X,C,y,typ,mid*pb.lambdamax,3.,np.ones(X.shape[1]))
        actual=(B[i]+B[i+1])/2
        if not typ.startswith('MC'): actual=actual[:,0]
        np.testing.assert_allclose(actual,expected,atol=8e-5,rtol=8e-5)


def test_zero_ridge_legacy_unchanged():
    X,C,y=data()
    for typ in ('R1','R2','C1','C2','R3','R4'):
        target=np.where(y>0,1,-1) if typ.startswith('C') else y
        kw=dict(typ=typ,meth='Path-Alg',intercept=False)
        old=Classo((X,C,target),.5,**kw)
        new=Classo((X,C,target),.5,ridge=0.,**kw)
        for a,b in zip(old,new) if isinstance(old,tuple) else [(old,new)]:
            np.testing.assert_array_equal(a,b)


@pytest.mark.parametrize('bad',[-1,np.nan,np.inf,[1]])
def test_invalid_ridge(bad):
    X,C,y=data()
    with pytest.raises(ValueError,match='ridge'):
        Classo((X,C,y),.5,ridge=bad)
    with pytest.raises(ValueError,match='ridge'):
        pathlasso((X,C,y),ridge=bad)


def test_absolute_lambda_and_unpenalized_intercept():
    X,C,y=data();y=y+7
    maximum,b=Classo((X,C,y),.4,ridge=5.,intercept=True,get_lambdamax=True)
    absolute=Classo((X,C,y),.4*maximum,ridge=5.,intercept=True,true_lam=True)
    np.testing.assert_allclose(b,absolute,atol=1e-9)
    B,L=pathlasso((X,C,y),lambdas=[1.,.4],ridge=5.,intercept=True)
    np.testing.assert_allclose(B[-1],b,atol=1e-9)
    assert abs(B[0,0]-np.mean(y))<1e-8
    assert np.max(abs(B[0,1:]))<1e-8


def test_high_level_cv_and_fixed_lambda():
    from classo import classo_problem
    X,C,y=data();pb=classo_problem(X,y,C=C)
    pb.formulation.concomitant=False;pb.formulation.intercept=True;pb.formulation.ridge=2.
    pb.model_selection.StabSel=False;pb.model_selection.PATH=True
    pb.model_selection.PATHparameters.lambdas=np.array([1.,.6,.3])
    pb.model_selection.CV=True;pb.model_selection.CVparameters.Nsubset=3
    pb.model_selection.CVparameters.lambdas=np.array([1.,.6,.3])
    pb.model_selection.LAMfixed=True;pb.model_selection.LAMfixedparameters.lam=.3
    pb.solve()
    expected=Classo((X,C,y),.3,typ='R1',ridge=2.,intercept=True)
    np.testing.assert_allclose(pb.solution.LAMfixed.beta,expected)
    np.testing.assert_allclose(pb.solution.PATH.BETAS[-1],expected)
    assert np.isfinite(pb.solution.CV.yGraph).all()


def test_zero_scale_perspective():
    pytest.importorskip('cvxpy')
    X,C,y=data();y=np.zeros_like(y)
    for typ in ('R3','R4'):
        b,s=Classo((X,C,y),.5,typ=typ,ridge=1.,intercept=True)
        np.testing.assert_allclose(b,0.,atol=1e-6)
        assert s<1e-6




@pytest.mark.parametrize('typ',['R3','R4'])
@pytest.mark.parametrize('intercept',[False,True])
def test_perspective_custom_scale_weight_and_absolute_path(typ,intercept):
    pytest.importorskip('cvxpy')
    X,C,y=data(13); y[0]+=20
    e=8.;rho=1.8
    maximum,b,s=Classo((X,C,y),.3,typ=typ,ridge=4.,intercept=intercept,
                       get_lambdamax=True,e=e,rho=rho)
    expected=reference(X,C,y,typ,.3*maximum,4.,np.ones(5),intercept,rho=rho,e=e)
    np.testing.assert_allclose(b,expected,atol=1e-4)
    B,L,S=pathlasso((X,C,y),lambdas=[maximum*1.1,.3*maximum,0.],
        typ=typ,ridge=4.,intercept=intercept,e=e,rho=rho,true_lam=True,return_sigm=True)
    np.testing.assert_allclose(B[1],b,atol=1e-4)
    assert np.max(abs(B[0,int(intercept):]))<1e-5
    assert np.min(S)>=0


def test_r4_closed_zero_scale_baseline_and_positive_scale_fit():
    pytest.importorskip('cvxpy')
    X,C,y=data();rho=.5;e=30.
    maximum,b,s=Classo((X,C,y),.4,typ='R4',ridge=2.,intercept=True,rho=rho,e=e,get_lambdamax=True)
    expected=reference(X,C,y,'R4',.4*maximum,2.,np.ones(5),rho=rho,e=e)
    np.testing.assert_allclose(b,expected,atol=1e-4)
    # Inside-perspective ridge forces positive scale for nonzero beta.
    assert s>0
    beta=b[1:];residual=X@beta+b[0]-y
    assert np.minimum((residual/s)**2,rho**2).sum()+np.dot(beta,beta)/(s*s)==pytest.approx(e,abs=1e-6)
    null,zero_scale=Classo((X,C,y),1.1*maximum,typ='R4',ridge=2.,
        intercept=True,rho=rho,e=e,true_lam=True)
    np.testing.assert_allclose(null[1:],0,atol=1e-10)
    assert zero_scale==0




@pytest.mark.parametrize('method',['first','max','lam'])
def test_stability_propagates_ridge(method):
    from classo import stability
    X,C,y=data()
    result=stability((X,C,y),StabSelmethod=method,formulation='R1',
                     ridge=2.,Nlam=17,q=1,B=2,intercept=True)
    distribution=result[0] if method=='first' else result
    assert np.shape(distribution)==(6,)
    assert np.all((distribution>=0)&(distribution<=1))
