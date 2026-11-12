"""Evaluate achieved affine homotopy segments without extrapolation."""
import numpy as np

def evaluate_path(pb, path, n_active=0):
    """Evaluate the knot path at descending relative lambda values.

    n_active limits nonzero feature rows, excluding the intercept. The path
    stops before an outgoing segment would exceed this limit; simultaneous
    entries are not split. On a sparse stop, return only achieved points,
    including the stop knot. Their relative lambdas are available through
    pb.path_info['evaluated_lambdas']; never extrapolate a stopped solution.
    """
    from ._quadratic_homotopy import solve_path_quadratic

    path = np.asarray(path, float)
    if (path.ndim != 1 or not len(path) or not np.isfinite(path).all()
            or np.any(path < 0) or np.any(path > 1) or np.any(np.diff(path) > 0)):
        raise ValueError("path must be a nonempty descending sequence in [0,1]")
    BETA, knots = solve_path_quadratic(pb, path[-1], n_active=n_active)
    if pb.path_info["stop_reason"] == "n_active":
        path = path[path >= knots[-1]]
        if not len(path) or path[-1] > knots[-1]:
            path = np.r_[path, knots[-1]]
    pb.path_info["evaluated_lambdas"] = path.copy()
    if len(knots) == 1:
        return [BETA[0].copy() for lam in path]
    out = []
    for lam in path:
        i = min(max(np.searchsorted(-np.asarray(knots), -lam, side="right")-1, 0), len(knots)-2)
        fraction = (knots[i]-lam)/(knots[i]-knots[i+1])
        out.append((1-fraction)*BETA[i]+fraction*BETA[i+1])
    return out
