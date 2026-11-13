"""Original production paths and independent KKT checks on every requested fit."""
from pathlib import Path
import numpy as np
import pytest
from scipy.optimize import linprog
from classo import classo_problem
from classo.path_alg import h_lambdamax

CASES=['error1_kkt_check','error2_zero_length_segment','error3_pivot_lp_infeasible']


def load_case(case):
    p=Path(__file__).parent/'fixtures'/'c1_production_nov13'/(case+'.npz')
    with np.load(p) as data:
        return tuple(data[k] for k in ('X','C','y','lambdas'))


@pytest.mark.parametrize('case',CASES)
def test_full_path_independent_original_objective_stationarity(case):
    X,C,y,fractions=load_case(case)
    pb=classo_problem(X,y,C=C)
    pb.formulation.classification=True;pb.formulation.intercept=True
    pb.formulation.huber=False;pb.formulation.concomitant=False
    ms=pb.model_selection;ms.PATH=True;ms.CV=False;ms.StabSel=False;ms.LAMfixed=False
    ms.PATHparameters.lambdas=fractions
    pb.solve()
    betas=np.asarray(pb.solution.PATH.BETAS)
    assert betas.shape==(len(fractions),X.shape[1]+1)
    assert np.isfinite(betas).all()
    maximum=h_lambdamax((X,C,y),0,typ='C1',intercept=True)
    for beta,fraction in zip(betas,fractions):
        lam=maximum*fraction
        hinge=np.maximum(1-y*(X@beta[1:]+beta[0]),0)
        gradient=-2*X.T@(y*hinge)
        scale=max(1.,lam,np.max(abs(gradient)))
        active=abs(beta[1:])>1e-7
        lower=np.where(active,gradient+lam*np.sign(beta[1:]),gradient-lam)
        upper=np.where(active,gradient+lam*np.sign(beta[1:]),gradient+lam)
        fit=linprog(np.r_[np.zeros(len(C)),1.],
            A_ub=np.vstack([np.c_[C.T,-np.ones(X.shape[1])],
                            np.c_[-C.T,-np.ones(X.shape[1])]])/scale,
            b_ub=np.r_[-lower,upper]/scale,
            bounds=[(None,None)]*len(C)+[(0,None)],method='highs')
        assert fit.success
        assert fit.fun/scale < 1e-6
        assert abs(2*y@hinge)<1e-6*scale
        assert np.max(abs(C@beta[1:]),initial=0)<1e-7*max(1.,np.max(abs(C))*np.linalg.norm(beta[1:],1))
