"""Fast paths must preserve the independently validated general homotopy."""
import numpy as np
import pytest
from classo.solve_MC1 import problem_MC1
from classo.mc1_path_engine import solve_path_MC1


def interpolate(B, knots, lam):
    values=np.asarray(B).reshape(len(B),-1)
    return np.array([np.interp(lam, knots[::-1], v[::-1]) for v in values.T])


@pytest.mark.parametrize('classes,seed', [(2,8),(3,12),(4,19)])
def test_linear_fast_paths_match_general_solver(classes,seed):
    rng=np.random.default_rng(seed)
    X=rng.normal(size=(24,5));y=np.arange(24)%classes
    paths=[];infos=[]
    for fast in [False,True]:
        pb=problem_MC1((X,np.ones((1,5)),y),intercept=True)
        pb._mc1_fast=fast
        B,L=solve_path_MC1(pb,.2)
        paths.append((B,np.asarray(L)));infos.append(pb.path_info)
    for lam in np.linspace(.2,1,41):
        np.testing.assert_allclose(interpolate(*paths[0],lam),interpolate(*paths[1],lam),atol=2e-7,rtol=2e-7)
    assert infos[0]['direction_linear']==infos[0]['certificate_linear']==0
    assert infos[1]['direction_linear']>0
    assert infos[1]['certificate_linear']>0
    assert max(infos[1]['residuals'])<1e-7
