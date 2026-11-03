import numpy as np

from .solve_R1 import problem_R1, Classo_R1, pathlasso_R1
from .solve_R2 import problem_R2, Classo_R2, pathlasso_R2
from .solve_R3 import problem_R3, Classo_R3, pathlasso_R3
from .solve_R4 import problem_R4, Classo_R4, pathlasso_R4
from .path_alg import solve_path, pathalgo_general, h_lambdamax


"""
Classo and pathlasso are the main functions,
they can call every algorithm acording
to the method and formulation required
"""

# can be 'Path-Alg', 'P-PDS' , 'PF-PDS' or 'DR'


def _divide_by_weights(matrix, w):
    """Rescale X and C by the weights, so that the problem has unit weights."""
    if w is None:
        return matrix
    return (matrix[0] / w, matrix[1] / w, matrix[2])


def _centered(matrices):
    """Center X and y.

    For R1 and R3 the intercept is simply beta0 = ybar - Xbar . beta, so by
    replacing X by X - Xbar and y by y - ybar we can solve the standard problem.
    """
    X, C, y = matrices
    Xbar, ybar = np.mean(X, axis=0), np.mean(y)
    return (X - Xbar, C, y - ybar), Xbar, ybar


def _scaled(lam, lambdamax, true_lam):
    """Express lam as a fraction of lambda_max."""
    return lam / lambdamax if true_lam else lam


def _problem_R3(matrices, meth, e):
    """Problem R3, rescaled when e differs from its default n / 2."""
    n = len(matrices[0])
    if e is None or e == n / 2:
        return problem_R3(matrices, meth), 1.0
    r = np.sqrt(2 * e / n)
    return problem_R3((matrices[0] * r, matrices[1], matrices[2] * r), meth), r


def _problem_R4(matrices, meth, rho, e, intercept):
    """Problem R4, rescaled when e differs from its default n."""
    n = len(matrices[0])
    if e is None or e == n:
        return problem_R4(matrices, meth, rho, intercept=intercept), 1.0
    r = np.sqrt(e / n)
    rescaled = (matrices[0] * r, matrices[1], matrices[2] * r)
    return problem_R4(rescaled, meth, rho / r, intercept=intercept), r


# Solvers of Classo: each returns (lambdamax, beta, sigma or None).


def _classo_R1(matrices, lam, meth, true_lam, intercept, **_):
    if intercept:
        matrices, Xbar, ybar = _centered(matrices)
    if meth not in ["Path-Alg", "P-PDS", "PF-PDS", "DR"]:
        meth = "DR"
    pb = problem_R1(matrices, meth)
    beta = Classo_R1(pb, _scaled(lam, pb.lambdamax, true_lam))
    if intercept:
        beta = np.array([ybar - np.vdot(Xbar, beta)] + list(beta))
    return pb.lambdamax, beta, None


def _classo_R2(matrices, lam, meth, true_lam, intercept, rho, **_):
    if meth not in ["Path-Alg", "P-PDS", "PF-PDS", "DR"]:
        meth = "ODE"
    pb = problem_R2(matrices, meth, rho, intercept=intercept)
    return pb.lambdamax, Classo_R2(pb, _scaled(lam, pb.lambdamax, true_lam)), None


def _classo_R3(matrices, lam, meth, true_lam, intercept, e, **_):
    if intercept:
        matrices, Xbar, ybar = _centered(matrices)
    if meth not in ["Path-Alg", "DR"]:
        meth = "DR"
    pb, _r = _problem_R3(matrices, meth, e)
    beta, s = Classo_R3(pb, _scaled(lam, pb.lambdamax, true_lam))
    if intercept:
        beta = np.array([ybar - np.vdot(Xbar, beta)] + list(beta))
    return pb.lambdamax, beta, s


def _classo_R4(matrices, lam, meth, true_lam, intercept, rho, e, **_):
    if meth not in ["Path-Alg", "DR"]:
        meth = "DR"
    pb, _r = _problem_R4(matrices, meth, rho, e, intercept)
    beta, s = Classo_R4(pb, _scaled(lam, pb.lambdamax, true_lam))
    return pb.lambdamax, beta, s


def _classo_classification(typ, rho, matrices, lam, true_lam, intercept):
    """C1 (rho = 0) and C2: the last point of the path algorithm."""
    assert set(matrices[2]).issubset({1, -1})
    lambdamax = h_lambdamax(matrices, rho, typ=typ, intercept=intercept)
    out = solve_path(
        matrices,
        _scaled(lam, lambdamax, true_lam),
        False,
        rho,
        typ,
        intercept=intercept,
    )
    if intercept:
        beta = np.array([out[0][-1]] + list(out[1][-1]))
    else:
        beta = out[0][-1]
    return lambdamax, beta, None


def _classo_C1(matrices, lam, true_lam, intercept, **_):
    return _classo_classification("C1", 0, matrices, lam, true_lam, intercept)


def _classo_C2(matrices, lam, true_lam, intercept, rho_classification, **_):
    return _classo_classification(
        "C2", rho_classification, matrices, lam, true_lam, intercept
    )


_CLASSO_SOLVERS = {
    "R1": _classo_R1,
    "R2": _classo_R2,
    "R3": _classo_R3,
    "R4": _classo_R4,
    "C1": _classo_C1,
    "C2": _classo_C2,
}


def Classo(
    matrix,
    lam,
    typ="R1",
    meth="DR",
    rho=1.345,
    get_lambdamax=False,
    true_lam=False,
    e=None,
    rho_classification=-1.0,
    w=None,
    intercept=False,
    return_sigm=True,
):
    """Solve the constrained Lasso problem of formulation ``typ`` at a single lambda.

    Args:
        matrix (tuple) : the data ``(X, C, y)``.
        lam (float) : the penalty, as a fraction of lambda_max (unless ``true_lam`` is True).
        typ (str, optional) : formulation, one of "R1", "R2", "R3", "R4", "C1", "C2".
        meth (str, optional) : numerical method, "Path-Alg", "DR", "P-PDS" or "PF-PDS".
        get_lambdamax (bool, optional) : if True, lambda_max is returned first.
        intercept (bool, optional) : if True, an unpenalized intercept is fitted and returned first in beta.

    Returns:
        ndarray : the solution beta, preceded by lambda_max if ``get_lambdamax``,
        and followed by sigma for the concomitant formulations "R3" and "R4".

    Example:
        >>> import numpy as np
        >>> from classo import random_data, Classo
        >>> (X, C, y), sol = random_data(50, 10, 3, 1, 0.5, zerosum=True, seed=4)
        >>> beta = Classo((X, C, y), 0.1, meth="Path-Alg")
        >>> np.nonzero(np.abs(beta) > 1e-3)[0]
        array([1, 2, 5])
        >>> bool(abs(C @ beta).max() < 1e-8)
        True
    """
    solver = _CLASSO_SOLVERS.get(typ, _classo_R1)  # any other typ is solved as R1
    lambdamax, beta, s = solver(
        _divide_by_weights(matrix, w),
        lam,
        meth=meth,
        true_lam=true_lam,
        intercept=intercept,
        rho=rho,
        e=e,
        rho_classification=rho_classification,
    )

    if w is not None:
        if intercept:
            beta[1:] = beta[1:] / w
        else:
            beta = beta / w

    if typ in ["R3", "R4"] and return_sigm:
        return (lambdamax, beta, s) if get_lambdamax else (beta, s)
    return (lambdamax, beta) if get_lambdamax else beta


# Solvers of pathlasso: each returns (lambdamax, lambdas, BETA, S or None), where
# lambdas are the input lambdas expressed as fractions of lambda_max.


def _rescaled_path(lambdass, lambdamax, true_lam):
    return [lamb / lambdamax for lamb in lambdass] if true_lam else lambdass


def _path_R1(matrices, lambdass, meth, true_lam, intercept, n_active, **_):
    if intercept:
        matrices, Xbar, ybar = _centered(matrices)
    pb = problem_R1(matrices, meth)
    lambdass = _rescaled_path(lambdass, pb.lambdamax, true_lam)
    BETA = pathlasso_R1(pb, lambdass, n_active=n_active)
    if intercept:
        BETA = np.array([[ybar - Xbar.dot(beta)] + list(beta) for beta in BETA])
    return pb.lambdamax, lambdass, BETA, None


def _path_R2(matrices, lambdass, meth, true_lam, intercept, rho, Nactive, **_):
    pb = problem_R2(matrices, meth, rho, intercept=intercept)
    lambdass = _rescaled_path(lambdass, pb.lambdamax, true_lam)
    return pb.lambdamax, lambdass, pathlasso_R2(pb, lambdass, n_active=Nactive), None


def _path_R3(matrices, lambdass, meth, true_lam, intercept, e, Nactive, **_):
    if intercept:
        matrices, Xbar, ybar = _centered(matrices)
    pb, r = _problem_R3(matrices, meth, e)
    lambdass = _rescaled_path(lambdass, pb.lambdamax, true_lam)
    BETA, S = pathlasso_R3(pb, lambdass, n_active=Nactive)
    S = np.array(S) / r ** 2
    BETA = np.array(BETA)
    if intercept:
        BETA = np.array([[ybar - Xbar.dot(beta)] + list(beta) for beta in BETA])
    return pb.lambdamax, lambdass, BETA, S


def _path_R4(matrices, lambdass, meth, true_lam, intercept, rho, e, Nactive, **_):
    pb, r = _problem_R4(matrices, meth, rho, e, intercept)
    lambdass = _rescaled_path(lambdass, pb.lambdamax, true_lam)
    BETA, S = pathlasso_R4(pb, lambdass, n_active=Nactive)
    return pb.lambdamax, lambdass, np.array(BETA), np.array(S) / r ** 2


def _path_classification(typ, rho, matrices, lambdass, true_lam, intercept, Nactive):
    """C1 (no rho) and C2 with the general path algorithm."""
    assert set(matrices[2]).issubset({1, -1})
    lambdamax = h_lambdamax(matrices, rho, typ=typ, intercept=intercept)
    lambdass = _rescaled_path(lambdass, lambdamax, true_lam)
    options = {"n_active": Nactive, "intercept": intercept}
    if typ == "C2":
        options["rho"] = rho
    BETA = pathalgo_general(matrices, lambdass, typ, **options)
    return lambdamax, lambdass, BETA, None


def _path_C1(matrices, lambdass, true_lam, intercept, Nactive, **_):
    return _path_classification(
        "C1", 0, matrices, lambdass, true_lam, intercept, Nactive
    )


def _path_C2(matrices, lambdass, true_lam, intercept, rho_classification, Nactive, **_):
    return _path_classification(
        "C2", rho_classification, matrices, lambdass, true_lam, intercept, Nactive
    )


_PATH_SOLVERS = {
    "R1": _path_R1,
    "R2": _path_R2,
    "R3": _path_R3,
    "R4": _path_R4,
    "C1": _path_C1,
    "C2": _path_C2,
}


def pathlasso(
    matrix,
    lambdas=False,
    n_active=0,
    lamin=1e-2,
    typ="R1",
    meth="Path-Alg",
    rho=1.345,
    true_lam=False,
    e=None,
    return_sigm=False,
    rho_classification=-1.0,
    w=None,
    intercept=False,
):
    """Compute the solutions of formulation ``typ`` along a path of lambdas.

    Args:
        matrix (tuple) : the data ``(X, C, y)``.
        lambdas (ndarray, optional) : lambdas as fractions of lambda_max; by default 100 values from 1 to ``lamin``.
        n_active (int, optional) : if positive, stop once this many coefficients are active.
        typ (str, optional) : formulation, one of "R1", "R2", "R3", "R4", "C1", "C2".
        meth (str, optional) : numerical method, "Path-Alg", "DR", "P-PDS" or "PF-PDS".
        return_sigm (bool, optional) : for "R3" and "R4", also return the sigmas.
        intercept (bool, optional) : if True, an unpenalized intercept is fitted and returned first in each beta.

    Returns:
        tuple : ``(BETAS, lambdas)`` with one beta per row of BETAS and the lambdas
        rescaled by lambda_max, plus the sigmas when ``return_sigm`` applies.

    Example:
        >>> import numpy as np
        >>> from classo import random_data, pathlasso
        >>> (X, C, y), sol = random_data(50, 10, 3, 1, 0.5, zerosum=True, seed=4)
        >>> BETAS, lambdas = pathlasso((X, C, y), lambdas=np.linspace(1.0, 0.1, 5))
        >>> BETAS.shape
        (5, 10)
        >>> np.count_nonzero(np.abs(BETAS) > 1e-3, axis=1)
        array([0, 0, 2, 3, 3])
        >>> bool(np.abs(BETAS @ C.T).max() < 1e-8)
        True
    """
    if type(lambdas) is bool:
        lambdas = lamin ** (np.linspace(0.0, 1, 100))
    # decreasing order
    lambdass = list(lambdas[::-1]) if lambdas[0] < lambdas[-1] else list(lambdas)

    solver = _PATH_SOLVERS.get(typ, _path_R1)  # any other typ is solved as R1
    lambdamax, lambdass, BETA, S = solver(
        _divide_by_weights(matrix, w),
        lambdass,
        meth=meth,
        true_lam=true_lam,
        intercept=intercept,
        rho=rho,
        e=e,
        rho_classification=rho_classification,
        n_active=n_active,
        Nactive=n_active if n_active != 0 else False,
    )

    real_path = [lam * lambdamax for lam in lambdass]

    if w is not None:
        ww = np.array([1] + list(w)) if intercept else w
        BETA = np.array([beta / ww for beta in BETA])

    if typ in ["R3", "R4"] and return_sigm:
        return (np.array(BETA), real_path, S)
    return (np.array(BETA), real_path)
