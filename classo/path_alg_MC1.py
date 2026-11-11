import numpy as np

from .multiclass_path import face_direction


"""Local MC1 calculations retained alongside the event-driven path engine.

Full knot traversal is implemented by mc1_path_engine.solve_path_MC1.

Use increasing-lambda derivatives, consistently with face_direction. A step
delta towards lower regularization updates B to B - delta * beta_dot.
No ridge is added, and singular fixed-face systems raise explicitly.
"""


def derivatives_MC1(pb, B, active, faces):
    """Compute a local fixed-face direction with class-specific hinge pieces.

    The caller must supply an already selected, valid outgoing face. This
    function does not decide releases at weakly active faces. In particular,
    at margin=1 an outgoing-region decision is required before calling this
    strict-region helper. Full certificate propagation remains future work.
    Returns active-row coefficient derivatives, row-maximum derivatives and
    the current quadratic sample masks.
    """
    B = pb._coefficient_matrix(B)
    margins = pb.targets * pb.matrix[0].dot(B)
    if np.any(np.abs(margins - 1.0) <= pb.tol):
        raise ValueError("A sample is on a margin boundary; select the outgoing region first")
    H, quadratic = pb.active_hessians(B, active)
    beta_dot, t_dot = face_direction(pb.matrix[0], pb.matrix[1], active, faces,
                                    pb.n_classes, tol=pb.tol, hessians=H,
                                    weights=pb.weights[np.asarray(active, dtype=int)],
                                    hessian_designs=lambda: [np.vstack([
                                        np.sqrt(2)*pb.matrix[0][quadratic[:,k]][:,active],
                                        np.diag(np.sqrt(pb.ridge_diagonal[active]))])
                                        for k in range(pb.n_classes)])
    return beta_dot, t_dot, quadratic


def margin_events_MC1(pb, B, beta_dot, lam):
    """List positive exact sample-margin event distances below lambda=lam.

    beta_dot is a full (p,K) increasing-lambda derivative. Returns sorted
    (delta, sample, class) tuples; retains simultaneous events. This is one
    event family, not the next global step: row/face and dual-certificate
    events must also be included before taking any step.
    """
    B = pb._coefficient_matrix(B)
    beta_dot = pb._coefficient_matrix(beta_dot)
    if not np.isfinite(lam) or lam < 0:
        raise ValueError("lam must be finite and nonnegative")
    X = pb.matrix[0]
    margins = pb.targets * X.dot(B)
    rate = pb.targets * X.dot(beta_dot)
    if np.any(np.abs(margins - 1.0) <= pb.tol):
        raise ValueError("Resolve current margin-boundary events before propagation")
    delta = np.full(margins.shape, np.inf)
    np.divide(margins - 1.0, rate, out=delta, where=rate != 0)
    rows, columns = np.nonzero((delta > 0) & (delta <= lam))
    return sorted((float(delta[i, k]), int(i), int(k)) for i, k in zip(rows, columns))
