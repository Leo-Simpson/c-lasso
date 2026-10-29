import numpy as np
import pytest
from classo.multiclass_path import initialize_path, face_direction


def test_unconstrained_initialization():
    result = initialize_path(np.eye(2), np.array([[3., -2.], [1., 0.]]))
    assert result['lambda_max'] == 5
    np.testing.assert_array_equal(result['candidates'], [0])
    assert result['active'].size == 0
    assert not result['B'].any()


def test_constrained_initialization():
    C = np.ones((1, 2))
    result = initialize_path(np.eye(2), np.diag([3., 2.]), C)
    assert result['lambda_max'] == pytest.approx(2.5)
    assert np.abs(result['adjusted_correlations']).sum(axis=1).max() <= 2.5+1e-8
    assert initialize_path(np.eye(2), np.diag([3., 2.]), np.eye(2))['lambda_max'] < 1e-8


def test_tied_face_direction_matches_closed_form():
    # For one row, Y=(3,2), lambda in (1,5): B=(t,t), t=(5-lambda)/2.
    dB, dt = face_direction(np.ones((1, 1)), None, [0], [[(0, 1), (1, 1)]], 2)
    np.testing.assert_allclose(dB, [[-.5, -.5]])
    np.testing.assert_allclose(dt, [-.5])


def test_free_coordinate_is_not_forced_to_tie():
    # Below lambda=1 the second coordinate stays at its unpenalized fit, 2.
    dB, dt = face_direction(np.ones((1, 1)), None, [0], [[(0, 1)]], 2)
    np.testing.assert_allclose(dB, [[-1., 0.]])


def test_tree_constraint_and_redundant_equalities():
    C = np.ones((2, 2))
    dB, dt = face_direction(np.eye(2), C, [0, 1],
                            [[(0, 1), (1, -1)], [(0, -1), (1, 1)]], 2)
    np.testing.assert_allclose(C @ dB, 0., atol=1e-12)
    np.testing.assert_allclose(dB, [[-.5, .5], [.5, -.5]], atol=1e-12)


def test_nonunique_direction_rejected():
    with pytest.raises(np.linalg.LinAlgError):
        face_direction(np.ones((1, 2)), None, [0, 1], [[(0, 1)], [(0, 1)]], 1)
