import numpy as np
import pytest

from classo.solve_MC1 import problem_MC1
from classo.path_alg_MC1 import derivatives_MC1, margin_events_MC1


def test_three_class_affine_segment_and_tied_margin_events():
    # Identity design: all three signed coefficients in each row are tied.
    # On 0<lambda<6, t=1-lambda/6. This is an exact analytic MC1 path segment.
    pb = problem_MC1((np.eye(3), None, [0, 1, 2]))
    lam = 3.
    B = .5 * pb.targets
    faces = [[(k, int(pb.targets[j, k])) for k in range(3)] for j in range(3)]
    dB, dt, masks = derivatives_MC1(pb, B, [0, 1, 2], faces)
    np.testing.assert_allclose(dB, -pb.targets/6, atol=1e-12)
    np.testing.assert_allclose(dt, -np.ones(3)/6, atol=1e-12)
    assert masks.all()
    # Every interior point satisfies the full objective's row stationarity.
    for delta in (.25, 1., 2.):
        next_B, next_lam = B - delta*dB, lam-delta
        _, grad = pb.loss_gradient(next_B)
        np.testing.assert_allclose(grad + next_lam*pb.targets/3, 0, atol=1e-12)
    events = margin_events_MC1(pb, B, dB, lam+1e-10)
    assert len(events) == 9
    np.testing.assert_allclose([event[0] for event in events], 3., atol=1e-12)
    assert not margin_events_MC1(pb, B, np.zeros_like(B), lam)


def test_class_specific_hessians_and_boundary_rejection():
    pb = problem_MC1((np.eye(3), None, [0, 1, 2]))
    B = .5*pb.targets
    B[0, 0] = 2.
    faces = [[(0, 1)], [(0, -1), (1, 1), (2, -1)],
             [(0, -1), (1, -1), (2, 1)]]
    # Row 0's only penalized maximum is in a zero-curvature loss region.
    # Its outgoing stationary direction cannot be solved on this face.
    with pytest.raises(np.linalg.LinAlgError):
        derivatives_MC1(pb, B, [0, 1, 2], faces)
    with pytest.raises(ValueError, match="margin boundary"):
        derivatives_MC1(pb, pb.targets, [0, 1, 2], faces)
    with pytest.raises(ValueError, match="margin-boundary"):
        margin_events_MC1(pb, pb.targets, B, 1.)
