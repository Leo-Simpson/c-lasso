"""Scalar C1 regression cases; no external data or optional solvers needed."""
import numpy as np
import pytest
from scipy.optimize import minimize, LinearConstraint
from classo.path_alg import solve_path, parameters_for_update, up_cl, check_C1_certificate
from classo._classification_path import direction, segment


def tree_like(seed):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(18, 8))
    X = np.c_[A, A[:, :4].mean(1), A[:, 4:].mean(1),
              A[:, :2].mean(1), A[:, 2:4].mean(1)]
    C = np.ones((1, X.shape[1]))
    y = np.where(A[:, 0]-A[:, 1]+rng.normal(size=18) > 0, 1, -1)
    return X, C, y


def reference(X, C, y, lam):
    """Independent squared-hinge epigraph QP in original scalar coordinates."""
    p = X.shape[1]
    Z = np.c_[np.ones(len(X)), X]
    q = p+1
    selection = np.c_[np.zeros(p), np.eye(p)]
    epigraph = np.vstack([np.c_[-selection, np.eye(p)],
                          np.c_[selection, np.eye(p)]])
    constraints = [LinearConstraint(epigraph, 0, np.inf),
                   LinearConstraint(np.c_[np.zeros(len(C)), C, np.zeros((len(C), p))], 0, 0)]
    def objective(v):
        h = np.maximum(1-y*(Z@v[:q]), 0)
        value = (h@h+lam*v[q:].sum())/len(y)
        gradient = np.r_[-2*Z.T@(y*h), np.full(p, lam)]/len(y)
        return value, gradient
    fit = minimize(objective, np.zeros(q+p), jac=True, method='SLSQP',
                   constraints=constraints, options=dict(ftol=1e-12, maxiter=2000))
    assert fit.success, fit.message
    return fit.fun*len(y)


@pytest.mark.parametrize('seed', [1, 5, 6, 31])
def test_dependent_columns_release_incompatible_zero_faces(seed):
    X, C, y = tree_like(seed)
    intercepts, B, L = solve_path((X, C, y), .005, 0, 0, 'C1', intercept=True)
    assert L[-1] == .005
    maximum = parameters_for_update((X, C, y), .005, 0, 'C1', intercept=True).lambdamax
    for i in np.unique(np.linspace(0, len(L)-2, 3).astype(int)):
        b = (np.asarray(B[i])+B[i+1])/2
        bias = (intercepts[i]+intercepts[i+1])/2
        lam = (L[i]+L[i+1])/2*maximum
        h = np.maximum(1-y*(X@b+bias), 0)
        objective = h@h+lam*abs(b).sum()
        assert objective == pytest.approx(reference(X, C, y, lam), rel=2e-7, abs=2e-7)
    state = parameters_for_update((X, C, y), .005, 0, 'C1', intercept=True)
    for _ in range(300):
        check_C1_certificate(state)
        margin = y*(X@state.beta+state.beta0)
        assert not np.any(state.F & (margin > 1+1e-8))
        assert not np.any(~state.F & (margin < 1-1e-8))
        if state.lam <= .005:
            break
        up_cl(state)
    else:
        pytest.fail('C1 did not reach the requested endpoint')


def test_small_positive_coordinate_keeps_its_signed_face():
    # A nonzero coefficient retains its sign even below a generic event band.
    X = np.array([[1.]])
    beta = np.array([5e-9])
    # A free intercept is not needed: choose lambda from this actual gradient.
    certificate = -(-2*X.T@(1-X@beta))
    lam = float(certificate[0])
    D, active, quadratic = direction(X, np.empty((0, 1)), np.ones(1), beta, lam, certificate)
    assert D[0] > 0
    assert active.tolist() == [0]


def test_unit_interval_certificate_handles_tiny_event_distance():
    X = np.array([[1.]])
    C = np.empty((0, 1))
    y = np.ones(1)
    beta = np.array([.5])
    D = np.array([.5])
    delta = segment(X, C, y, beta, 1., D, np.array([0]), np.ones(1, bool), 1e-9)
    assert delta == pytest.approx(1e-9, rel=1e-8, abs=1e-16)
    assert beta[0]+delta*D[0] > beta[0]
