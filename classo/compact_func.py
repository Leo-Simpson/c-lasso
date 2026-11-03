import numpy as np
import numpy.linalg as LA

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

    if w is not None:
        matrices = (matrix[0] / w, matrix[1] / w, matrix[2])
    else:
        matrices = matrix

    X, C, y = matrices

    if typ == "R3":
        if intercept:
            # here we use the fact that for R1 and R3,
            # the intercept is simple beta0 = ybar-Xbar .vdot(beta)
            # so by changing the X to X-Xbar and y to y-ybar
            #  we can solve standard problem
            Xbar, ybar = np.mean(X, axis=0), np.mean(y)
            matrices = (X - Xbar, C, y - ybar)

        if meth not in ["Path-Alg", "DR"]:
            meth = "DR"
        if e is None or e == len(matrices[0]) / 2:
            r = 1.0
            pb = problem_R3(matrices, meth)
            e = len(matrices[0]) / 2
        else:
            r = np.sqrt(2 * e / len(matrices[0]))
            pb = problem_R3((matrices[0] * r, matrices[1], matrices[2] * r), meth)
        lambdamax = pb.lambdamax
        if true_lam:
            beta, s = Classo_R3(pb, lam / lambdamax)
        else:
            beta, s = Classo_R3(pb, lam)

        if intercept:
            betaO = ybar - np.vdot(Xbar, beta)
            beta = np.array([betaO] + list(beta))

    elif typ == "R4":

        if meth not in ["Path-Alg", "DR"]:
            meth = "DR"
        if e is None or e == len(matrices[0]):
            r = 1.0
            pb = problem_R4(matrices, meth, rho, intercept=intercept)
            e = len(matrices[0])
        else:
            r = np.sqrt(e / len(matrices[0]))
            pb = problem_R4(
                (matrices[0] * r, matrices[1], matrices[2] * r),
                meth,
                rho / r,
                intercept=intercept,
            )

        lambdamax = pb.lambdamax
        if true_lam:
            beta, s = Classo_R4(pb, lam / lambdamax)
        else:
            beta, s = Classo_R4(pb, lam)

    elif typ == "R2":

        if meth not in ["Path-Alg", "P-PDS", "PF-PDS", "DR"]:
            meth = "ODE"
        pb = problem_R2(matrices, meth, rho, intercept=intercept)
        lambdamax = pb.lambdamax
        if true_lam:
            beta = Classo_R2(pb, lam / lambdamax)
        else:
            beta = Classo_R2(pb, lam)

    elif typ == "C2":

        assert set(matrices[2]).issubset({1, -1})

        lambdamax = h_lambdamax(
            matrices, rho_classification, typ="C2", intercept=intercept
        )
        if true_lam:
            out = solve_path(
                matrices,
                lam / lambdamax,
                False,
                rho_classification,
                "C2",
                intercept=intercept,
            )
        else:
            out = solve_path(
                matrices, lam, False, rho_classification, "C2", intercept=intercept
            )
        if intercept:
            beta0, beta = out[0][-1], out[1][-1]
            beta = np.array([beta0] + list(beta))
        else:
            beta = out[0][-1]

    elif typ == "C1":

        assert set(matrices[2]).issubset({1, -1})

        lambdamax = h_lambdamax(matrices, 0, typ="C1", intercept=intercept)
        if true_lam:
            out = solve_path(
                matrices, lam / lambdamax, False, 0, "C1", intercept=intercept
            )
        else:
            out = solve_path(matrices, lam, False, 0, "C1", intercept=intercept)
        if intercept:
            beta0, beta = out[0][-1], out[1][-1]
            beta = np.array([beta0] + list(beta))
        else:
            beta = out[0][-1]

    else:  # LS
        if intercept:
            # here we use the fact that for R1 and R3,
            #  the intercept is simple beta0 = ybar-Xbar .vdot(beta)
            #  so by changing the X to X-Xbar and y to y-ybar
            #  we can solve standard problem
            Xbar, ybar = np.mean(X, axis=0), np.mean(y)
            matrices = (X - Xbar, C, y - ybar)

        if meth not in ["Path-Alg", "P-PDS", "PF-PDS", "DR"]:
            meth = "DR"
        pb = problem_R1(matrices, meth)
        lambdamax = pb.lambdamax
        if true_lam:
            beta = Classo_R1(pb, lam / lambdamax)
        else:
            beta = Classo_R1(pb, lam)

        if intercept:
            betaO = ybar - np.vdot(Xbar, beta)
            beta = np.array([betaO] + list(beta))

    if w is not None:
        if intercept:
            beta[1:] = beta[1:] / w
        else:
            beta = beta / w

    if typ in ["R3", "R4"] and return_sigm:
        if get_lambdamax:
            return (lambdamax, beta, s)
        else:
            return (beta, s)
    if get_lambdamax:
        return (lambdamax, beta)
    else:
        return beta


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

    Nactive = n_active
    if Nactive == 0:
        Nactive = False
    if type(lambdas) is bool:
        lambdas = lamin ** (np.linspace(0.0, 1, 100))

    if lambdas[0] < lambdas[-1]:
        lambdass = [
            lambdas[i] for i in range(len(lambdas) - 1, -1, -1)
        ]  # reverse the list if needed
    else:
        lambdass = [lambdas[i] for i in range(len(lambdas))]

    if w is not None:
        matrices = (matrix[0] / w, matrix[1] / w, matrix[2])
    else:
        matrices = matrix

    X, C, y = matrices

    if typ == "R2":

        pb = problem_R2(matrices, meth, rho, intercept=intercept)
        lambdamax = pb.lambdamax
        if true_lam:
            lambdass = [lamb / lambdamax for lamb in lambdass]
        BETA = pathlasso_R2(pb, lambdass, n_active=Nactive)

    elif typ == "R3":
        if intercept:
            # here we use the fact that for R1 and R3, the intercept is simple beta0 = ybar-Xbar .vdot(beta) so by changing the X to X-Xbar and y to y-ybar we can solve standard problem
            Xbar, ybar = np.mean(X, axis=0), np.mean(y)
            matrices = (X - Xbar, C, y - ybar)
        if e is None or e == len(matrices[0]) / 2:
            r = 1.0
            pb = problem_R3(matrices, meth)
        else:
            r = np.sqrt(2 * e / len(matrices[0]))
            pb = problem_R3((matrices[0] * r, matrices[1], matrices[2] * r), meth)
        lambdamax = pb.lambdamax
        if true_lam:
            lambdass = [lamb / lambdamax for lamb in lambdass]
        BETA, S = pathlasso_R3(pb, lambdass, n_active=Nactive)
        S = np.array(S) / r ** 2
        BETA = np.array(BETA)
        if intercept:
            BETA = np.array([[ybar - Xbar.dot(beta)] + list(beta) for beta in BETA])

    elif typ == "R4":

        if e is None or e == len(matrices[0]):
            r = 1.0
            pb = problem_R4(matrices, meth, rho, intercept=intercept)
        else:
            r = np.sqrt(e / len(matrices[0]))
            pb = problem_R4(
                (matrices[0] * r, matrices[1], matrices[2] * r),
                meth,
                rho / r,
                intercept=intercept,
            )

        lambdamax = pb.lambdamax
        if true_lam:
            lambdass = [lamb / lambdamax for lamb in lambdass]
        BETA, S = pathlasso_R4(pb, lambdass, n_active=Nactive)
        S = np.array(S) / r ** 2
        BETA = np.array(BETA)

    elif typ == "C2":

        assert set(matrices[2]).issubset({1, -1})

        lambdamax = h_lambdamax(
            matrices, rho_classification, typ="C2", intercept=intercept
        )
        if true_lam:
            lambdass = [lamb / lambdamax for lamb in lambdass]
        BETA = pathalgo_general(
            matrices,
            lambdass,
            "C2",
            n_active=Nactive,
            rho=rho_classification,
            intercept=intercept,
        )

    elif typ == "C1":

        assert set(matrices[2]).issubset({1, -1})

        lambdamax = h_lambdamax(matrices, 0, typ="C1", intercept=intercept)
        if true_lam:
            lambdass = [lamb / lambdamax for lamb in lambdass]
        BETA = pathalgo_general(
            matrices, lambdass, "C1", n_active=Nactive, intercept=intercept
        )

    else:  # R1
        if intercept:
            # here we use the fact that for R1 and R3,
            #  the intercept is simple beta0 = ybar-Xbar .vdot(beta)
            #  so by changing the X to X-Xbar and y to y-ybar
            #  we can solve standard problem
            Xbar, ybar = np.mean(X, axis=0), np.mean(y)
            matrices = (X - Xbar, C, y - ybar)
        pb = problem_R1(matrices, meth)
        lambdamax = pb.lambdamax
        if true_lam:
            lambdass = [lamb / lambdamax for lamb in lambdass]
        BETA = pathlasso_R1(pb, lambdass, n_active=n_active)

        if intercept:
            BETA = np.array([[ybar - Xbar.dot(beta)] + list(beta) for beta in BETA])

    real_path = [lam * lambdamax for lam in lambdass]

    if w is not None:
        if intercept:
            ww = np.array([1] + list(w))
        else:
            ww = w

        BETA = np.array([beta / ww for beta in BETA])

    if typ in ["R3", "R4"] and return_sigm:
        return (np.array(BETA), real_path, S)
    return (np.array(BETA), real_path)
