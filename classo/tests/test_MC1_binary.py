"""Independent small convex reference problems, not an MC1 path solver."""
import numpy as np
import pytest
from scipy.optimize import minimize

from classo import Classo
from classo.solve_MC1 import problem_MC1


def reference_MC1(pb, lam):
    """Solve a small epigraph problem independently of the homotopy code."""
    X, C, _ = pb.matrix
    p, K = X.shape[1], pb.n_classes
    size = p*K

    def objective(z):
        loss, grad = pb.loss_gradient(z[:size].reshape(p, K))
        # Scaling improves SLSQP's line search without changing the minimizer.
        scale = len(X)
        return ((loss + lam * pb.weights.dot(z[size:])) / scale,
                np.r_[grad.ravel(), lam*pb.weights] / scale)

    # t_j +/- B_jk >= 0, with row-major coefficient ordering.
    T = np.repeat(np.eye(p), K, axis=0)
    inequalities = np.vstack([np.c_[-np.eye(size), T], np.c_[np.eye(size), T]])
    constraints = [dict(type="ineq", fun=lambda z: inequalities.dot(z),
                        jac=lambda z: inequalities)]
    if len(C):
        E = np.c_[np.kron(C, np.eye(K)), np.zeros((len(C)*K, p))]
        constraints.append(dict(type="eq", fun=lambda z: E.dot(z), jac=lambda z: E))
    # Start at the feasible nuisance-only fit. A zero epigraph for free bias
    # rows can make SLSQP stop at a nonstationary all-zero intercept.
    initial = np.r_[pb.init.ravel(), np.max(abs(pb.init),axis=1)]
    result = minimize(objective, initial, jac=True, method="SLSQP",
                      bounds=[(None, None)]*size + [(0, None)]*p,
                      constraints=constraints, options=dict(ftol=1e-12, maxiter=2000))
    assert result.success, result.message
    assert np.min(inequalities.dot(result.x)) >= -1e-8
    return result.x[:size].reshape(p, K)


@pytest.mark.parametrize("constrained", [False, True])
@pytest.mark.parametrize("fraction", [.15, .6])
def test_binary_MC1_matches_C1(constrained, fraction):
    rng = np.random.default_rng(18)
    X = rng.normal(size=(24, 4))
    y = np.where(X[:, 0] - X[:, 1] + .8*rng.normal(size=len(X)) > 0, 1, -1)
    C = np.ones((1, 4)) if constrained else np.empty((0, 4))
    pb = problem_MC1((X, C, y))
    # Absolute lambda makes the factor-of-two loss normalization explicit.
    lam_C1 = fraction * pb.lambdamax / 2
    beta = Classo((X, C, y), lam_C1, typ="C1", meth="Path-Alg", true_lam=True)
    B = reference_MC1(pb, 2*lam_C1)
    expected = np.column_stack([-beta, beta])
    # Existing C1 code snaps sample-margin events within 1e-4 (path_alg.py).
    # Keep that comparison distinct from the tighter objective check below.
    np.testing.assert_allclose(B, expected, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(C.dot(B), 0, atol=1e-8)
    np.testing.assert_array_equal(np.argmax(X.dot(B), axis=1), (X.dot(beta) > 0).astype(int))
    binary_loss = np.maximum(1-y*X.dot(beta), 0).dot(np.maximum(1-y*X.dot(beta), 0))
    binary_objective = binary_loss + lam_C1*np.abs(beta).sum()
    mc_objective = pb.loss_gradient(B)[0] + 2*lam_C1*np.max(np.abs(B), axis=1).sum()
    assert mc_objective == pytest.approx(2*binary_objective, rel=1e-8, abs=1e-8)


def test_unconstrained_binary_lambda_max_scaling():
    X = np.array([[1., 2.], [-2., 1.], [1., -1.]])
    y = np.array([1, -1, 1])
    C = np.empty((0, 2))
    lam_C1, _ = Classo((X, C, y), .5, typ="C1", get_lambdamax=True)
    assert problem_MC1((X, C, y)).lambdamax == pytest.approx(2*lam_C1)
