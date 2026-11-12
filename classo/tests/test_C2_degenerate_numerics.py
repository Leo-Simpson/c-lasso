"""Regressions for redundant dual equations and weak tree-design curvature."""
import numpy as np
import pytest
from classo._classification_path import _dual_segment, _solve_design


def test_active_stationarity_roundoff_is_not_an_inactive_entry_event():
    # Active rows satisfy the equalities within roundoff. Rechecking their
    # saturated bounds as inequalities used to create negative event times.
    C = np.ones((1, 3))
    signs = np.array([1., -1., 0.])
    gradient = np.array([-1.+1e-10, 1.+1e-10, .2])
    gdot = np.array([1.+1e-10, -1.+2e-10, 0.])
    active = np.array([0, 1])
    delta, dual = _dual_segment(C, gradient, gdot, 1., .4, active, signs, 0)
    assert delta == pytest.approx(.4)
    adjusted = gradient+delta*gdot+C.T@dual
    np.testing.assert_allclose(adjusted[active], -(1.-delta)*signs[active], atol=1e-9)
    assert abs(adjusted[2]) < 1.-delta


def test_reduced_dual_event_matches_exact_inactive_entry_time():
    C = np.ones((1, 3))
    delta, dual = _dual_segment(C, np.array([-1., 1., .2]),
                               np.array([1., -1., 1.]), 1., .8,
                               np.array([0, 1]), np.array([1., -1., 0.]), 0)
    assert delta == pytest.approx(.4)
    np.testing.assert_allclose(dual, 0., atol=1e-14)


def test_reduced_dual_retains_nonunique_equality_multipliers():
    C = np.array([[1., 1., 0.], [0., 0., 1.]])
    delta, dual = _dual_segment(C, np.array([-1., 1., .2]),
                               np.array([1., -1., 2.]), 1., .8,
                               np.array([0, 1]), np.array([1., -1., 0.]), 0)
    assert delta == pytest.approx(.8)
    adjusted = np.array([-1., 1., .2])+delta*np.array([1., -1., 2.])+C.T@dual
    assert np.max(abs(adjusted)) <= 1.-delta+1e-9


def test_inconsistent_active_direction_still_fails():
    with pytest.raises(RuntimeError, match='not stationary'):
        _dual_segment(np.empty((0, 2)), np.array([-1., .2]),
                      np.array([.5, 0.]), 1., .4,
                      np.array([0]), np.array([1., 0.]), 0)


def test_design_svd_retains_weak_curvature_and_true_nullspace():
    Z = np.diag([1., 1e-6, 0.])
    rhs = np.array([2., 2e-6, 0.])
    direction = _solve_design(Z, rhs, np.array([0., 0., 7.]))
    np.testing.assert_allclose(direction, [1., 1e6, 7.], rtol=1e-12)
    np.testing.assert_allclose(2*Z.T@(Z@direction), rhs, atol=1e-14)


def test_inconsistent_true_nullspace_still_fails():
    with pytest.raises(np.linalg.LinAlgError, match='another outgoing face'):
        _solve_design(np.diag([1., 0.]), np.array([1., 1.]))
