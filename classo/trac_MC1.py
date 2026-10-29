"""Numerical trac preparation for MC1, independent of tree-library objects."""
import numpy as np

from .solve_MC1 import problem_MC1


def prepare_trac_MC1(data, A, w=None):
    """Return an MC1 problem and the leaf-to-node coefficient map M.

    data=(X,C,y), with X ALREADY log/CLR transformed and columns aligned to
    A's leaves. A is a leaf-by-node binary descendant-membership matrix;
    this helper does not infer or validate a biological tree. With n_j leaves
    in node j, M=A/n_j, X_node=X M, C_node=C M, and weights=w/n_j.
    C=None selects the leaf log-contrast constraint sum(beta)=0. w defaults
    to one per node. This matches the numerical aggregation/weight scaling
    in the inspected q2-classo add_taxa source. No count replacement is done.

    For node coefficients Gamma, leaf coefficients are M Gamma. Thus the
    log-contrast remains valid for every class, rather than merely imposing
    a tree-shaped equality with no feature transformation.
    """
    X, C, y = data
    X, A = np.asarray(X, float), np.asarray(A, float)
    if (X.ndim != 2 or A.ndim != 2 or X.shape[1] != A.shape[0]
            or not A.shape[1] or not np.isfinite(A).all()
            or not np.all((A == 0) | (A == 1))):
        raise ValueError("A must be an aligned, finite binary leaf-by-node matrix")
    counts = A.sum(axis=0)
    if np.any(counts == 0):
        raise ValueError("Every node must contain at least one leaf")
    C = np.ones((1, X.shape[1])) if C is None else np.asarray(C, float)
    if C.ndim != 2 or C.shape[1] != X.shape[1] or not np.isfinite(C).all():
        raise ValueError("C must have one finite column per leaf")
    w = np.ones(A.shape[1]) if w is None else np.asarray(w, float)
    if w.shape != (A.shape[1],) or not np.isfinite(w).all() or np.any(w <= 0):
        raise ValueError("w must have one positive finite value per node")
    M = A/counts
    return problem_MC1((X.dot(M), C.dot(M), y), w=w/counts), M
