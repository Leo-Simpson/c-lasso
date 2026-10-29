import numpy as np
import pytest

from classo.solve_MC1 import problem_MC1, Classo_MC1
from classo.trac_MC1 import prepare_trac_MC1


def test_tree_path_preserves_leaf_predictions_and_log_contrasts():
    rng = np.random.default_rng(9)
    X = rng.normal(size=(24, 4))
    X -= X.mean(axis=1, keepdims=True)
    # Four tips, two internal sibling clades, one root.
    A = np.c_[np.eye(4), [1, 1, 0, 0], [0, 0, 1, 1], np.ones(4)]
    y = np.arange(24) % 3
    pb, M = prepare_trac_MC1((X, None, y), A)
    B = Classo_MC1(pb, .3)
    leaf_B = M.dot(B)
    np.testing.assert_allclose(pb.matrix[0].dot(B), X.dot(leaf_B), atol=1e-10)
    np.testing.assert_allclose(leaf_B.sum(axis=0), 0., atol=1e-9)
    # Adding a per-sample constant to log features leaves log-contrast scores invariant.
    shifted = X + rng.normal(size=(24, 1))
    np.testing.assert_allclose(shifted.dot(leaf_B), X.dot(leaf_B), atol=1e-9)
    raw = problem_MC1((X.dot(A), np.ones((1, 4)).dot(A), y))
    raw_B = Classo_MC1(raw, .3)
    assert pb.lambdamax == pytest.approx(raw.lambdamax)
    # Node coefficients may be nonunique; compare identifiable scores and objectives.
    np.testing.assert_allclose(X.dot(A).dot(raw_B), X.dot(leaf_B), atol=1e-7)
    normalized_penalty = pb.weights.dot(np.max(np.abs(B), axis=1))
    raw_penalty = np.max(np.abs(raw_B), axis=1).sum()
    assert normalized_penalty == pytest.approx(raw_penalty, abs=1e-8)


def test_invalid_tree_inputs():
    X, y = np.eye(3), [0, 1, 2]
    for A in (np.ones((2, 2)), np.zeros((3, 1)), np.full((3, 2), .5)):
        with pytest.raises(ValueError):
            prepare_trac_MC1((X, None, y), A)
