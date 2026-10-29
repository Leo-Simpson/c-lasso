import numpy as np
import pytest

from classo.solve_MC1 import problem_MC1


def test_class_encoding_and_initial_state():
    X = np.eye(3)
    pb = problem_MC1((X, None, ["b", "a", "c"]))
    np.testing.assert_array_equal(pb.classes, ["a", "b", "c"])
    np.testing.assert_array_equal(pb.targets, [[-1, 1, -1], [1, -1, -1], [-1, -1, 1]])
    assert pb.dim == (3, 3, 0)
    assert pb.lambdamax == pytest.approx(6)
    assert not pb.init.any() and not pb.active.size
    assert pb.loss_gradient(pb.init)[0] == 9
    expected = pb.loss_gradient(np.eye(3))[0]
    X[:] = 0
    assert pb.loss_gradient(np.eye(3))[0] == expected


def test_gradient_and_active_hessians_by_finite_difference():
    rng = np.random.default_rng(31)
    X = rng.normal(size=(9, 4))
    pb = problem_MC1((X, np.ones((1, 4)), np.arange(9) % 3))
    B = rng.normal(size=(4, 3)) * .4
    delta = rng.normal(size=B.shape)
    loss, grad = pb.loss_gradient(B)
    eps = 1e-6
    lp, gp = pb.loss_gradient(B + eps * delta)
    lm, gm = pb.loss_gradient(B - eps * delta)
    assert (lp - lm) / (2 * eps) == pytest.approx(np.sum(grad * delta), rel=1e-7)
    H, masks = pb.active_hessians(B, np.arange(4))
    for k in range(3):
        np.testing.assert_allclose((gp-gm)[:, k] / (2*eps), H[k].dot(delta[:, k]), atol=1e-7)
    restricted, _ = pb.active_hessians(B, [3, 1])
    np.testing.assert_allclose(restricted, H[:, [3, 1]][:, :, [3, 1]])


def test_constrained_zero_optimality_certificate():
    X, C = np.eye(3), np.ones((1, 3))
    pb = problem_MC1((X, C, [0, 1, 2]))
    _, grad = pb.loss_gradient(pb.init)
    adjusted = -grad - C.T.dot(pb.dual)
    np.testing.assert_allclose(adjusted, pb.adjusted_correlations)
    assert np.abs(adjusted).sum(axis=1).max() == pytest.approx(pb.lambdamax)
    # U=(-2,-2,-2) gives a tight row L1 norm of 4, improving the raw norm 6.
    assert pb.lambdamax == pytest.approx(4)


def test_margin_boundary_and_empty_active_set():
    pb = problem_MC1((np.eye(2), None, [0, 1]))
    B = pb.targets.copy()
    loss, grad = pb.loss_gradient(B)
    assert loss == 0
    np.testing.assert_array_equal(grad, 0)
    H, masks = pb.active_hessians(B, [])
    assert H.shape == (2, 0, 0)
    assert not masks.any()


@pytest.mark.parametrize("data", [
    (np.eye(2), None, [1, 1]),
    (np.eye(2), None, [[1, 0], [0, 1]]),
    (np.eye(2), np.ones((1, 3)), [0, 1]),
    (np.eye(2), None, [0, np.nan]),
])
def test_invalid_data(data):
    with pytest.raises(ValueError):
        problem_MC1(data)


def test_invalid_coefficients_and_active_indices():
    pb = problem_MC1((np.eye(2), None, [0, 1]))
    with pytest.raises(ValueError):
        pb.loss_gradient(np.zeros((2, 1)))
    for active in ([0, 0], [-1], [2], [0.5]):
        with pytest.raises(ValueError):
            pb.active_hessians(pb.init, active)
