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
    covariates=None,
    ridge=0.0,
):

    from .elastic_net import validate_ridge, ridge_path
    ridge = validate_ridge(ridge)
    if ridge > 0 or typ == "MC2":
        if not np.isfinite(lam) or lam < 0:
            raise ValueError("lambda must be finite and nonnegative")
        B, _, S, maximum = ridge_path(matrix, [lam], typ, ridge, w, intercept,
            rho, rho_classification, covariates, true_lam, e)
        result = (B[0], S[0]) if S is not None and return_sigm else (B[0],)
        if get_lambdamax:
            return (maximum,)+result
        return result if len(result)>1 else result[0]

    if typ == "MC1":
        from .solve_MC1 import problem_MC1, Classo_MC1

        if meth != "Path-Alg":
            raise ValueError("MC1 currently supports meth='Path-Alg' only")
        pb = problem_MC1(matrix, meth, w=w, intercept=intercept, covariates=covariates)
        if not np.isfinite(lam) or lam < 0:
            raise ValueError("lambda must be finite and nonnegative")
        relative = (min(lam/pb.lambdamax, 1.) if pb.lambdamax else 1.) if true_lam else lam
        beta = Classo_MC1(pb, relative)
        return (pb.lambdamax, beta) if get_lambdamax else beta

    if covariates is not None:
        raise ValueError("covariates currently requires typ='MC1'; binary C1 objectives can use two-class MC1")
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
                (lam / lambdamax if lambdamax else 1.),
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
                matrices, (lam / lambdamax if lambdamax else 1.), False, 0, "C1", intercept=intercept
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

    if covariates is not None:
        raise ValueError("covariates currently requires typ='MC1'; binary C1 objectives can use two-class MC1")
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
    covariates=None,
    ridge=0.0,
):

    from .elastic_net import validate_ridge, ridge_path
    ridge = validate_ridge(ridge)
    if ridge > 0 or typ == "MC2":
        if type(lambdas) is bool:
            if not np.isfinite(lamin) or not 0 <= lamin <= 1:
                raise ValueError("lamin must lie in [0,1]")
            values = np.linspace(1.,0.,100) if lamin == 0 else np.geomspace(1.,lamin,100)
        else:
            values = lambdas
        B, L, S, _ = ridge_path(matrix, values, typ, ridge, w, intercept,
            rho, rho_classification, covariates, true_lam, e, n_active)
        return (B, list(L), S) if S is not None and return_sigm else (B, list(L))

    if typ == "MC1":
        from .solve_MC1 import problem_MC1, pathlasso_MC1

        if meth != "Path-Alg":
            raise ValueError("MC1 currently supports meth='Path-Alg' only")
        pb = problem_MC1(matrix, meth, w=w, intercept=intercept, covariates=covariates)
        if type(lambdas) is bool and (not np.isfinite(lamin) or not 0 <= lamin <= 1):
            raise ValueError("lamin must lie in [0,1]")
        values = ((np.linspace(1., 0., 100) if lamin == 0 else np.geomspace(1., lamin, 100)) if type(lambdas) is bool
                  else np.asarray(lambdas, float))
        if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError("lambdas must be a finite nonnegative vector")
        values = np.sort(values)[::-1]
        if true_lam:
            relative = np.minimum(values/pb.lambdamax, 1.) if pb.lambdamax else np.ones_like(values)
            real_path = values
        else:
            relative, real_path = values, values*pb.lambdamax
        BETA = pathlasso_MC1(pb, relative, n_active=n_active)
        if pb.path_info["stop_reason"] == "n_active":
            real_path = pb.path_info["evaluated_lambdas"]*pb.lambdamax
        return np.asarray(BETA), list(real_path)

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

    if covariates is not None:
        raise ValueError("covariates currently requires typ='MC1'; binary C1 objectives can use two-class MC1")
    if w is not None:
        matrices = (matrix[0] / w, matrix[1] / w, matrix[2])
    else:
        matrices = matrix

    raw_lambdass=list(lambdass)
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
            lambdass = [lamb / lambdamax if lambdamax else 1. for lamb in lambdass]
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
            lambdass = [lamb / lambdamax if lambdamax else 1. for lamb in lambdass]
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

    real_path = (raw_lambdass if true_lam and typ in ["C1","C2"] and lambdamax==0
                 else [lam * lambdamax for lam in lambdass])

    if covariates is not None:
        raise ValueError("covariates currently requires typ='MC1'; binary C1 objectives can use two-class MC1")
    if w is not None:
        if intercept:
            ww = np.array([1] + list(w))
        else:
            ww = w

        BETA = np.array([beta / ww for beta in BETA])

    if typ in ["R3", "R4"] and return_sigm:
        return (np.array(BETA), real_path, S)
    return (np.array(BETA), real_path)
