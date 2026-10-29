import numpy as np
import pytest

from classo import Classo, pathlasso
from classo.solve_MC1 import problem_MC1, Classo_MC1, pathlasso_MC1
from classo.mc1_path_engine import solve_path_MC1
from classo.tests.test_MC1_binary import reference_MC1


def objective(pb, B, lam):
    return pb.loss_gradient(B)[0]+lam*pb.weights.dot(np.max(np.abs(B), axis=1))


def test_identity_full_path_to_zero():
    pb = problem_MC1((np.eye(3), None, [0, 1, 2]))
    BETA, knots = solve_path_MC1(pb, 0.)
    assert len(knots) == 2
    np.testing.assert_allclose(BETA[-1], pb.targets, atol=1e-12)
    B = pathlasso_MC1(pb, [1., .75, .25, 0.])
    np.testing.assert_allclose(B, [(1-lam)*pb.targets for lam in [1., .75, .25, 0.]], atol=1e-12)
    assert pb.path_info['completed']


@pytest.mark.parametrize('K', [2, 3])
@pytest.mark.parametrize('constrained', [False, True])
def test_every_segment_against_independent_optimizer(K, constrained):
    rng = np.random.default_rng(22)
    X = rng.normal(size=(18, 4))
    y = np.arange(18) % K
    C = np.ones((1, 4)) if constrained else np.empty((0, 4))
    pb = problem_MC1((X, C, y))
    BETA, knots = solve_path_MC1(pb, .05)
    assert pb.path_info['completed']
    assert np.all(np.diff(knots) < 0)
    assert knots[-1] == pytest.approx(.05)
    for i in range(len(knots)-1):
        for alpha in (.2, .7):
            relative = (1-alpha)*knots[i]+alpha*knots[i+1]
            B = (1-alpha)*BETA[i]+alpha*BETA[i+1]
            ref = reference_MC1(pb, relative*pb.lambdamax)
            assert objective(pb, B, relative*pb.lambdamax) == pytest.approx(
                objective(pb, ref, relative*pb.lambdamax), rel=1e-9, abs=1e-8)
            np.testing.assert_allclose(X.dot(B), X.dot(ref), atol=3e-5)
            np.testing.assert_allclose(C.dot(B), 0., atol=1e-9)


@pytest.mark.parametrize('constrained', [False, True])
def test_actual_binary_path_matches_C1(constrained):
    rng = np.random.default_rng(18)
    X = rng.normal(size=(24, 4))
    y = np.where(X[:, 0]-X[:, 1]+.8*rng.normal(size=24) > 0, 1, -1)
    C = np.ones((1, 4)) if constrained else np.empty((0, 4))
    pb = problem_MC1((X, C, y))
    for relative, B in zip([.8, .4, .15], pathlasso_MC1(pb, [.8, .4, .15])):
        beta = Classo((X, C, y), relative*pb.lambdamax/2, typ='C1', true_lam=True)
        np.testing.assert_allclose(B, np.column_stack([-beta, beta]), atol=1e-4, rtol=1e-4)


def test_weights_equivalent_to_column_rescaling_and_compact_dispatch():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(15, 3))
    C, y, w = np.ones((1, 3)), np.arange(15) % 3, np.array([.5, 1., 2.])
    pb = problem_MC1((X, C, y), w=w)
    raw = Classo_MC1(pb, .2)
    rescaled = Classo_MC1(problem_MC1((X/w, C/w, y)), .2)/w[:, None]
    np.testing.assert_allclose(raw, rescaled, atol=1e-8)
    public = Classo((X, C, y), .2, typ='MC1', meth='Path-Alg', w=w)
    np.testing.assert_allclose(public, raw, atol=1e-9)
    B, values = pathlasso((X, C, y), [.8, .2], typ='MC1', w=w)
    np.testing.assert_allclose(B[-1], raw, atol=1e-9)
    np.testing.assert_allclose(values, pb.lambdamax*np.array([.8, .2]))


def test_invalid_requests_and_event_limit():
    pb = problem_MC1((np.eye(3), None, [0, 1, 2]))
    with pytest.raises(ValueError):
        pathlasso_MC1(pb, [.1, .8])
    assert Classo((np.eye(3), None, [0, 1, 2]), .5, typ='MC1',
                  meth='Path-Alg', intercept=True).shape == (4, 3)
    rng = np.random.default_rng(18)
    pb = problem_MC1((rng.normal(size=(18, 4)), None, np.arange(18) % 3))
    with pytest.raises(RuntimeError, match='max_events'):
        solve_path_MC1(pb, .01, max_events=1)
    assert not pb.path_info['completed']


@pytest.mark.parametrize('seed', range(20))
def test_stress_paths_and_interior_certification(seed):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(20, 5))
    C = np.ones((1, 5)) if seed % 2 else None
    pb = problem_MC1((X, C, np.arange(20) % 3))
    BETA, knots = solve_path_MC1(pb, .02)
    assert pb.path_info['completed']
    assert knots[-1] == pytest.approx(.02)
    assert max(pb.path_info['residuals']) < 1e-7
    # Check every affine piece rather than just the requested output grid.
    for i in range(len(knots)-1):
        B = .63*BETA[i]+.37*BETA[i+1]
        lam = (.63*knots[i]+.37*knots[i+1])*pb.lambdamax
        ref = reference_MC1(pb, lam)
        assert objective(pb, B, lam) == pytest.approx(objective(pb, ref, lam), rel=1e-9, abs=1e-7)


def test_redundant_equalities_and_zero_threshold():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(15, 3))
    y = np.arange(15) % 3
    first = Classo_MC1(problem_MC1((X, np.ones((1, 3)), y)), .2)
    redundant = Classo_MC1(problem_MC1((X, np.ones((2, 3)), y)), .2)
    np.testing.assert_allclose(first, redundant, atol=1e-8)
    pb = problem_MC1((X, np.eye(3), y))
    assert pb.lambdamax < 1e-12
    np.testing.assert_allclose(Classo_MC1(pb, 0), 0, atol=1e-12)


def test_nonunique_direction_does_not_silently_add_ridge():
    pb = problem_MC1((np.array([[1., 1.], [-1., -1.]]), None, [0, 1]))
    with pytest.raises(np.linalg.LinAlgError, match='Nonunique'):
        solve_path_MC1(pb, .2)
    assert not pb.path_info['completed']
