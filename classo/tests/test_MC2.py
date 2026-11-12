"""MC2 remains usable without importing or exposing the ridge feature."""
import numpy as np
import pytest
from classo import Classo, pathlasso
from classo.solve_MC2 import problem_MC2
from classo.tests.test_MC1_binary import reference_MC1


@pytest.mark.parametrize('intercept', [False, True])
@pytest.mark.parametrize('lower', [-1., 0.])
def test_MC2_independent_epigraph_optimum(intercept, lower):
    rng = np.random.default_rng(62)
    X = rng.normal(size=(24, 4))
    y = np.arange(24) % 3
    C = np.ones((1, 4))
    pb = problem_MC2((X, C, y), intercept=intercept, rho_classification=lower)
    B = Classo((X,C,y), .3, typ='MC2', meth='Path-Alg', intercept=intercept,
               rho_classification=lower)
    expected = reference_MC1(pb, .3*pb.lambdamax)
    # Verify the stated Huberized hinge independently of the problem's gradient.
    def objective(beta):
        margin = pb.targets * (pb.matrix[0] @ beta)
        h = 1-lower
        loss = np.where(margin < lower, h*h+2*h*(lower-margin),
                        np.maximum(1-margin, 0)**2).sum()
        return loss + .3*pb.lambdamax*pb.weights@np.max(abs(beta),axis=1)
    assert objective(B) == pytest.approx(objective(expected), rel=1e-8, abs=1e-8)
    np.testing.assert_allclose(pb.matrix[0]@B,pb.matrix[0]@expected,atol=2e-5)
    np.testing.assert_allclose(C@B[int(intercept):],0,atol=1e-8)


@pytest.mark.parametrize('intercept', [False, True])
def test_MC2_binary_matches_current_C2(intercept):
    rng=np.random.default_rng(18);X=rng.normal(size=(24,4))
    y=np.where(X[:,0]-X[:,1]+.8*rng.normal(size=24)>0,1,-1)
    C=np.ones((1,4))
    maximum,b=Classo((X,C,y),.4,typ='C2',meth='Path-Alg',intercept=intercept,
                     get_lambdamax=True)
    B=Classo((X,C,y),.8*maximum,typ='MC2',meth='Path-Alg',intercept=intercept,true_lam=True)
    np.testing.assert_allclose(B,np.c_[-b,b],atol=2e-7)
    values,_=pathlasso((X,C,y),[2*maximum,.8*maximum],typ='MC2',
                       intercept=intercept,true_lam=True)
    np.testing.assert_allclose(values[-1],B,atol=2e-7)
