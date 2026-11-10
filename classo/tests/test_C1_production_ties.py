"""Full requested paths for the production reports supplied 2026-11-10."""
from pathlib import Path
import numpy as np
import pytest
from scipy.optimize import linprog
from classo import classo_problem
from classo.path_alg import h_lambdamax


CASES = ['1_segment_certificate_lp', '2_iteration_limit',
         '3_outgoing_face', '4_regions_cycled']


@pytest.mark.parametrize('case', CASES)
def test_full_production_path_and_independent_stationarity(case):
    fixture=Path(__file__).parent/'fixtures/c1_production'/(case+'.npz')
    with np.load(fixture) as data:
        X,C,y,fractions=(data[key] for key in ('X','C','y','fractions'))
    problem=classo_problem(X,y,C=C)
    problem.formulation.classification=True
    problem.formulation.concomitant=False
    problem.formulation.huber=False
    problem.formulation.intercept=True
    problem.model_selection.PATH=True
    problem.model_selection.CV=False
    problem.model_selection.StabSel=False
    problem.model_selection.LAMfixed=False
    problem.model_selection.PATHparameters.lambdas=fractions
    problem.model_selection.PATHparameters.n_active=len(y)
    problem.solve()
    betas=np.asarray(problem.solution.PATH.BETAS)
    assert betas.shape==(len(fractions),X.shape[1]+1)
    assert np.isfinite(betas).all()
    maximum=h_lambdamax((X,C,y),0,typ='C1',intercept=True)
    for beta,fraction in zip(betas,fractions):
        lam=fraction*maximum
        hinge=np.maximum(1-y*(X@beta[1:]+beta[0]),0.)
        gradient=-2*X.T@(y*hinge)
        scale=max(1.,lam,np.max(abs(gradient)))
        active=abs(beta[1:])>1e-7
        lower=np.where(active,gradient+lam*np.sign(beta[1:]),gradient-lam)
        upper=np.where(active,gradient+lam*np.sign(beta[1:]),gradient+lam)
        # Independently minimize stationarity error over equality multipliers.
        fit=linprog(np.r_[np.zeros(len(C)),1.],
                    A_ub=np.vstack([np.c_[C.T,-np.ones(X.shape[1])],
                                    np.c_[-C.T,-np.ones(X.shape[1])]])/scale,
                    b_ub=np.r_[-lower,upper]/scale,
                    bounds=[(None,None)]*len(C)+[(0,None)],method='highs')
        assert fit.success
        assert fit.fun/scale<1e-6
        assert abs(2*y@hinge)<1e-6*scale
        assert np.max(abs(C@beta[1:]),initial=0.)<1e-7*max(1.,np.max(abs(C))*np.linalg.norm(beta[1:],1))
