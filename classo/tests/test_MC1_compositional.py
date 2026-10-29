import numpy as np
import pytest

from classo import Classo, pathlasso
from classo.mc1_path_engine import solve_path_MC1
from classo.solve_MC1 import problem_MC1, Classo_MC1
from classo.tests.test_MC1_binary import reference_MC1


def test_intercepts_match_independent_reference():
    rng = np.random.default_rng(12)
    X = rng.normal(size=(18, 3))
    pb = problem_MC1((X, np.ones((1, 3)), np.arange(18) % 3), intercept=True)
    B = Classo_MC1(pb, .2)
    ref = reference_MC1(pb, .2*pb.lambdamax)
    np.testing.assert_allclose(B, ref, atol=2e-6)
    np.testing.assert_allclose(B[1:].sum(axis=0), 0, atol=1e-10)


def test_unpenalized_intercept_at_zero_features_and_above_threshold():
    X = np.zeros((9, 3))
    y = np.array([0]*4+[1]*3+[2]*2)
    B = Classo((X, np.ones((1, 3)), y), 100., typ='MC1', meth='Path-Alg',
               true_lam=True, intercept=True)
    np.testing.assert_allclose(B[0], 2*np.bincount(y)/len(y)-1)
    np.testing.assert_array_equal(B[1:], 0.)


def test_sample_rescaling_and_clr_invariance():
    rng = np.random.default_rng(8)
    X = rng.normal(size=(24, 4))  # log abundances
    y, C = np.arange(24) % 3, np.ones((1, 4))
    shifted = X + rng.normal(size=(24, 1))
    pb = problem_MC1((X, C, y), intercept=True)
    other = problem_MC1((shifted, C, y), intercept=True)
    assert pb.lambdamax == pytest.approx(other.lambdamax)
    B, shifted_B = Classo_MC1(pb, .3), Classo_MC1(other, .3)
    np.testing.assert_allclose(B, shifted_B, atol=2e-8)
    np.testing.assert_allclose(np.c_[np.ones(len(X)), X].dot(B),
                               np.c_[np.ones(len(X)), shifted].dot(shifted_B), atol=1e-8)


def test_sparsity_stopping_preserves_actual_lambda_values():
    rng = np.random.default_rng(12)
    X, y, C = rng.normal(size=(18, 3)), np.arange(18) % 3, np.ones((1, 3))
    pb = problem_MC1((X, C, y), intercept=True)
    BETA, knots = solve_path_MC1(pb, .01, n_active=2)
    assert pb.path_info['stop_reason'] == 'n_active'
    assert not pb.path_info['completed']
    assert knots[-1] > .01
    assert all(np.count_nonzero(np.max(np.abs(B[1:]), axis=1) > 1e-7) <= 2 for B in BETA)
    B, values = pathlasso((X, C, y), [.9, .01], typ='MC1', intercept=True, n_active=2)
    assert values[-1] == pytest.approx(knots[-1]*pb.lambdamax)
    assert len(B) == len(values)
    ref = reference_MC1(pb, values[-1])
    np.testing.assert_allclose(B[-1], ref, atol=2e-6)


def test_simultaneous_entry_is_not_split_to_meet_budget():
    pb = problem_MC1((np.eye(3), np.ones((1, 3)), [0, 1, 2]), intercept=True)
    BETA, knots = solve_path_MC1(pb, .1, n_active=1)
    assert pb.path_info['stop_reason'] == 'n_active'
    assert knots == [1.]
    np.testing.assert_array_equal(BETA[0][1:], 0.)


@pytest.mark.parametrize('limit', [-1, 1.5])
def test_invalid_sparsity_limit(limit):
    pb = problem_MC1((np.eye(3), None, [0, 1, 2]))
    with pytest.raises(ValueError):
        solve_path_MC1(pb, .1, n_active=limit)
