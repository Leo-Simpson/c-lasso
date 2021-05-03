"""Utility functions for implementing and testing out ALO for c-lasso.
"""

import numpy as np
import scipy.linalg



def solve_cls(X, y, C):
    """Solve the constrained least-squares problem.

    This currently uses a very naive method based on explicit inversion.
    A better method would use a Cholesky decomposition or similar.

    Parameters
    ----------
    X : np.array
        Design matrix
    y : np.array
        Observation vector
    C : np.array
        Constraint matrix
    """
    K = X.T @ X
    K_inv = np.linalg.inv(K)
    P = K_inv - K_inv @ C.T @ np.linalg.inv(C @ K_inv @ C.T) @ C @ K_inv
    return P @ (X.T @ y)


def alo_cls_h_naive(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Computes the ALO leverages for the CLS (constrained least-squares).

    Note that just like for the OLS, the CLS is a linear smoother, and hence
    the ALO leverages are exact LOO leverages.

    This is the reference implementation which uses "obvious" linear algebra.
    See `alo_cls_h` for a better implementation.

    Parameters
    ----------
    X : np.ndarray
        A numpy array of size [n, p] containing the design matrix.
    C : np.ndarray
        A numpy array of size [d, p] containing the constraints.

    Returns
    -------
    np.ndarray
        A 1-dimensional array of size n, representing the computed leverage of
        each observation.
    """
    K = X.T @ X
    K_inv = np.linalg.inv(K)
    P = K_inv - K_inv @ C.T @ np.linalg.inv(C @ K_inv @ C.T) @ C @ K_inv
    return np.diag(X @ P @ X.T)


def alo_cls_h(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Computes the ALO leverages for the CLS.

    Note that just like for the OLS, the CLS is a linear smoother, and hence
    the ALO leverages are exact LOO leverages.

    See `alo_cls_h_naive` for the mathematically convenient expression. This
    function implements the computation in a much more efficient manner by
    relying extensively on the cholesky decomposition.
    """
    K = X.T @ X
    K_cho, _ = scipy.linalg.cho_factor(
        K, overwrite_a=True, lower=True, check_finite=False
    )
    K_inv_2_C = scipy.linalg.solve_triangular(
        K_cho, C.T, lower=True, check_finite=False
    )
    K_inv_2_Xt = scipy.linalg.solve_triangular(
        K_cho, X.T, lower=True, check_finite=False
    )

    C_Ki_C = K_inv_2_C.T @ K_inv_2_C

    CKC_cho, _ = scipy.linalg.cho_factor(
        C_Ki_C, overwrite_a=True, lower=True, check_finite=False
    )
    F = scipy.linalg.solve_triangular(
        CKC_cho, K_inv_2_C.T, lower=True, check_finite=False
    )
    return (K_inv_2_Xt ** 2).sum(axis=0) - ((F @ K_inv_2_Xt) ** 2).sum(axis=0)


def alo_h(
    X: np.ndarray, beta: np.ndarray, y: np.ndarray, C: np.ndarray
):
    """Computes the ALO leverage and residual for the c-lasso.

    Due to its L1 structure, the ALO for the constrained lasso corresponds
    to the ALO of the CLS reduced to the equi-correlation set. This function directly
    extracts the equi-correlation set and delegates to `alo_cls_h` for computing
    the ALO leverage.

    Parameters
    ----------
    X : np.ndarray
        A numpy array of size [n, p] representing the design matrix.
    beta : np.ndarray
        A numpy array of size [p] representing the solution at which to
        compute the ALO risk.
    y : np.ndarray
        A numpy array of size [n] representing the observations.
    C : np.ndarray
        A numpy array of size [d, p] representing the constraint matrix.

    Returns
    -------
    alo_residual : np.ndarray
        A numpy array of size [n] representing the estimated ALO residuals
    h : np.ndarray
        A numpy array of size [n] representing the ALO leverage at each observation.
    """
    E = np.flatnonzero(beta)

    if len(E) == 0:
        return (y - X @ beta), np.zeros(X.shape[0])

    X_E = X[:, E]
    C_E = C[:, E]

    h = alo_cls_h(X_E, C_E)
    return (y - X @ beta) / (1 - h), h


def alo_classo_risk(
    X: np.ndarray, C: np.ndarray, y: np.ndarray, betas: np.ndarray
):
    """Computes the ALO risk for the c-lasso at the given estimates.

    Parameters
    ----------
    X : np.ndarray
        A numpy array of size [n, p] representing the design matrix.
    C : np.ndarray
        A numpy array of size [d, p] representing the constraint matrix.
    y : np.ndarray
        A numpy array of size [n] representing the observations.
    betas : np.ndarray
        A numpy array of size [m, p], where ``m`` denotes the number of solutions
        in the path, representing the solution at each point in the path.

    Returns
    -------
    mse : np.ndarray
        A numpy array of size [m], representing the ALO estimate of the mean squared error
        at each solution along the path.
    df : np.ndarray
        A numpy array of size [m], representing the estimated normalized degrees of freedom
        at each solution along the path.
    """
    mse = np.empty(len(betas))
    df = np.empty(len(betas))

    for i, beta in enumerate(betas):
        res, h = alo_h(X, beta, y, C)
        df[i] = np.mean(h)
        mse[i] = np.mean(np.square(res))

    return mse, df



"""
Not used for now.
import functools
from typing import Tuple

import multiprocessing
import numpy as np
import scipy.linalg
import tqdm
import sklearn.linear_model

from classo import classo_problem
from classo.solve_R1 import pathlasso_R1, problem_R1




def generate_data(n, p, k, d, sigma=1, seed=None):
    ""Generate random c-lasso problem.

    Parameters
    ----------
    n : int
        Number of observations
    p : int
        Number of parameters
    k : int
        Number of ground truth non-zero parameters.
    d : int
        Number of constraints
    sigma : float
        Standard deviation of additive noise.
    seed : int, optional
        Optional integer used to seed the random number generator
        for reproducibility.
    ""
    rng = np.random.Generator(np.random.Philox(seed))

    X = rng.normal(scale=1 / np.sqrt(k), size=(n, p))
    C = rng.normal(size=(d, p))
    beta_nz = np.ones(k)
    C_k = C[:, :k]

    # ensure that beta verifies the constraint by projecting.
    beta_nz = beta_nz - C_k.T @ scipy.linalg.lstsq(C_k.T, beta_nz)[0]
    beta_nz /= np.mean(beta_nz ** 2)
    beta = np.concatenate((beta_nz, np.zeros(p - k)))

    eps = rng.normal(scale=sigma, size=(n,))

    y = X @ beta + eps
    return (X, C, y), beta


def solve_standard(X, C, y, lambdas=None):
    ""Utility function to solve standard c-lasso formulation.""
    problem = problem_R1((X, C, y), "Path-Alg")
    problem.tol = 1e-6

    if lambdas is None:
        lambdas = np.logspace(0, 1, num=80, base=1e-3)
    else:
        lambdas = lambdas / problem.lambdamax

    if lambdas[0] < lambdas[-1]:
        lambdas = lambdas[::-1]

    beta = pathlasso_R1(problem, lambdas)
    return np.array(beta), lambdas * problem.lambdamax

def solve_loo(X, C, y):
    ""Solves the leave-one-out problem for each observation.

    This function makes use of python multi-processing in order
    to accelerate the computation across all the cores.
    ""
    _, lambdas = solve_standard(X, C, y)

    ctx = multiprocessing.get_context("spawn")

    with ctx.Pool(initializer=_set_sequential_mkl) as pool:
        result = pool.imap(
            functools.partial(_solve_loo_i_beta, X=X, C=C, y=y, lambdas=lambdas),
            range(X.shape[0]),
        )

        result = list(result)

    return np.stack(result, axis=0), lambdas


def solve_loo_i(X, C, y, i, lambdas):
    X = np.concatenate((X[:i], X[i + 1 :]))
    y = np.concatenate((y[:i], y[i + 1 :]))
    return solve_standard(X, C, y, lambdas)


def _solve_loo_i_beta(i, X, C, y, lambdas):
    return solve_loo_i(X, C, y, i, lambdas)[0]


def _set_sequential_mkl():
    import os

    try:
        import mkl

        mkl.set_num_threads(1)
    except ImportError:
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OMP_NUM_THREADS"] = "1"


"""


"""
# The functions below are simply helper functions which implement the same functionality for the LASSO (not the C-LASSO)
# They are mostly intended for debugging and do not need to be integrated.


def solve_lasso(X, y, lambdas=None):
    lambdas, betas, _ = sklearn.linear_model.lasso_path(
        X, y, intercept=False, lambdas=lambdas
    )
    return lambdas, betas.T


def alo_lasso_h(X, y, beta, tol=1e-4):
    E = np.abs(beta) > tol
    if E.sum() == 0:
        return y - X @ beta, np.zeros(X.shape[0])

    X = X[:, E]

    K = X.T @ X
    H = X @ scipy.linalg.solve(K, X.T, assume_a="pos")

    h = np.diag(H)
    return (y - X @ beta[E]) / (1 - h), h


def alo_lasso_risk(X, y, betas):
    mse = np.empty(len(betas))
    df = np.empty(len(betas))

    for i, beta in enumerate(betas):
        res, h = alo_lasso_h(X, y, beta)
        df[i] = np.mean(h)
        mse[i] = np.mean(np.square(res))

    return mse, df


def _lasso_loo(i, X, y, lambdas):
    X_i = np.concatenate((X[:i], X[i + 1 :]))
    y_i = np.concatenate((y[:i], y[i + 1 :]))
    return solve_lasso(X_i, y_i, lambdas)[1]


def solve_lasso_loo(X, y, lambdas=None, progress=False):
    if lambdas is None:
        lambdas, _ = solve_lasso(X, y)

    with multiprocessing.Pool(initializer=_set_sequential_mkl) as pool:
        result = pool.imap(
            functools.partial(_lasso_loo, X=X, y=y, lambdas=lambdas), range(X.shape[0])
        )
        if progress:
            result = tqdm.tqdm(result, total=X.shape[0])
        result = list(result)

    return lambdas, np.stack(result, axis=0)
"""