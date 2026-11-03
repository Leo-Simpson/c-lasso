"""
The equation we are solving is:

2N.T.dot(N)  beta_dot      C.T v_dot = s     on active variables (beta_dot elsewhere)

        C    beta_dot      0   v_dot = 0


    and
    (lambda s)_dot  = N.T.N beta_dot   + C.T v_dot              on inactive variables




N = A[F,:]
if intercept :
    N = A[F,:]-mean(A[F,:],axis=0)

"""

import numpy as np
import numpy.linalg as LA

N = 10000
N_frac = 100


class parameters_for_update:
    """Object that has parameters needed at each breaking point during path algorithm

    Attributes :
        number_act  (int) : number of active parameter
        idr         : saves the independant rows of the matrix C resctricted to the actives parameters
        Xt          :  inverse of M
        activity    :  list of boolean, activity[i] is True when variable i is active
        beta        : current solution beta
        s           : current subgradient
        lam         : current lam
        lambdamax   : lambdamax
        M           : matrix to invert
        y           : output
        r           : residual
        F           : F is the set where r<1 and if C1, and it is the set where rho<r<1 for C2, and r<rho for R2
        rho         : only use when doing huber path algo


    """

    def __init__(self, matrices, lamin, rho, typ, eps_L2=1e-3, intercept=False):
        if typ == "C2" and rho > 1:
            raise ValueError(
                "For huberized hinge, rho has to be smaller than 1, but here it is :",
                rho,
            )

        (self.A, self.C, self.y) = matrices
        self.lamin = lamin
        self.rho = rho
        self.formulation = typ
        self.intercept = intercept
        n, d, k = len(self.A), len(self.A[0]), len(self.C)
        self.number_act = 0
        # C1 follows the unregularized squared-hinge objective. A hidden
        # ridge changes directions and can change which margin event is next.
        if typ == "C1":
            eps_L2 = 0.
        self.eps_L2 = eps_L2
        self.idr = [False] * k
        self.activity = [False] * d
        self.beta = np.zeros(d)

        if typ in ["C1", "C2"]:
            r_func = lambda b0, y: y * b0
            dr = self.y
        else:
            r_func = lambda b0, y: b0 - y
            dr = 1.0

        if intercept:
            self.beta0 = find_beta0(r_func, dr, self.y, rho, typ)
            self.Abar = np.mean(self.A, axis=0)
        else:
            self.beta0 = 0.0

        self.F = find_F(r_func(self.beta0, self.y), rho, typ)
        P = self.A[self.F]
        if intercept:
            P = P - np.mean(P, axis=0)
        AtA = P.T.dot(P) + eps_L2 * np.eye(d)

        self.r = r_func(self.beta0, self.y)
        s = -2 * self.A.T.dot(dr * h_prime(rho, typ)(self.r))
        self.lambdamax = LA.norm(s, np.inf)
        self.lamin = lamin
        self.s = s / self.lambdamax
        self.lam = 1.0

        for i in range(d):
            if self.s[i] == 1.0 or self.s[i] == -1.0:
                self.activity[i] = True
                self.number_act += 1
                if k > 0:
                    to_ad = next_idr1(self.idr, self.C[:, self.activity])
                    if type(to_ad) == int:
                        self.idr[to_ad] = True

        if k == 0:
            self.M = 2 * AtA
        else:
            self.M = np.concatenate(
                (
                    np.concatenate((2 * AtA, self.C.T), axis=1),
                    np.concatenate((self.C, np.zeros((k, k))), axis=1),
                ),
                axis=0,
            )

        N = self.M[self.activity + self.idr, :][:, self.activity + self.idr]
        if typ == "C1":
            self.Xt = _C1_inverse(N)
        else:
            try:
                self.Xt = LA.inv(N)
            except LA.LinAlgError:
                self.Xt = LA.inv(N + self.eps_L2 * np.eye(len(N)))


# iteration of the function up to solve the path at each breaking points.
def solve_path(matrices, lamin, n_active, rho, typ, intercept=False):
    """
    This functions will compute the path for all the breaking points :
    beta is a piecewise linear function of lambda, and only value on the breaking points
    is computed here

    Args :
        matrices : A (aka X),C,y the matrices of the problem
        lamin : fraction of lambda that gives one criteria to stop : continue while lambda > lamin * lambda_max
        n_active : another criteria to stop
        rho : only useful for huber-classification
        typ : can be 'R1', 'R3', 'R2','C2' or 'C1'

    Return :
        BETA : list of beta(lambda) for lambda in LAMBDA
        LAMBDA : list of breaking points
    """

    d = len(matrices[0][0])
    param = parameters_for_update(matrices, lamin, rho, typ, intercept=intercept)
    if intercept:
        BETA0 = [param.beta0]
    BETA, LAM = [param.beta], [param.lam]
    if typ == "C1":
        check_C1_certificate(param)
    if param.lam < lamin:
        if intercept:
            return BETA0, BETA, LAM
        else:
            return BETA, LAM
    for i in range(d * N_frac):

        still_F = np.any(param.F)
        too_active = n_active > 0 and param.number_act >= n_active

        if not still_F or too_active or param.lam == lamin:
            if intercept:
                return BETA0, BETA, LAM
            else:
                return BETA, LAM

        # elif not np.any(param.F):
        #    print(param.r)
        #    raise ValueError("The problem looks infeasible because the set of active sample became zero, "
        #                     "at the iteration {} "
        #                      "for formulation {} "
        #                      "with intercept ? {} "
        #                      "with rho equal to {} ".format(i, typ, intercept, rho ))

        up(param)
        if typ == "C1":
            check_C1_certificate(param)
        BETA.append(param.beta), LAM.append(param.lam)

        # print("inside : ", param.r[ param.F], np.nonzero(param.F)[0] )
        # print(" outside : ", param.r[~param.F], np.nonzero(~param.F)[0])

        if intercept:
            BETA0.append(param.beta0)

    raise ValueError(
        "The path algorithm did not finsh after %i iterations " % N,
        "and with lamin=%f" % lamin,
        " and with n_active =" + str(n_active),
    )


def _C1_inverse(matrix):
    """Symmetric inverse, with a minimum-norm policy for tied directions.

    Rank-deficient active sets (e.g. duplicated columns) use a minimum-norm
    inverse in equilibrated coordinates, not a ridge-perturbed objective. Returned knots must still pass
    the actual primal/dual certificate; inconsistent directions are rejected.
    """
    # Symmetric equilibration prevents large feature scales from causing
    # genuine constraint directions to be mistaken for numerical nullspace.
    scaling = np.ones(len(matrix))
    balanced = matrix.copy()
    for _ in range(5):
        norm = np.max(np.abs(balanced), axis=1, initial=0.)
        factor = np.ones_like(norm)
        factor[norm > 0] = 1./np.sqrt(norm[norm > 0])
        scaling *= factor
        balanced = factor[:, None]*balanced*factor[None, :]
    eigenvalues, vectors = LA.eigh(balanced)
    cutoff = 1e-12*np.max(np.abs(eigenvalues), initial=0.)
    nonzero = np.abs(eigenvalues) > cutoff
    inverse = np.zeros_like(eigenvalues)
    inverse[nonzero] = 1./eigenvalues[nonzero]
    return scaling[:, None]*(vectors*inverse).dot(vectors.T)*scaling[None, :]


def check_C1_certificate(param, tol=1e-7):
    """Check the actual squared-hinge KKT conditions at a returned C1 knot.

    First try the small active stationarity multiplier solve. If its choice
    of nonunique multipliers fails, minimize the residual over all multipliers
    with a small LP. Never confuse a failed multiplier candidate with failure
    of the primal solution. No coefficient refit or numerical ridge is used.
    """
    A, C, y = param.A, param.C, param.y
    margin = y*(A.dot(param.beta)+param.beta0)
    v = 2*y*np.minimum(margin-1., 0.)
    gradient = A.T.dot(v)
    lam = param.lam*param.lambdamax
    scale = max(1., lam, np.max(np.abs(gradient), initial=0.))
    allowance = tol*scale
    active = np.abs(param.beta) > 1e-9
    rhs = -gradient[active]-lam*np.sign(param.beta[active])
    if len(C) and np.any(active):
        dual = LA.lstsq(C[:, active].T, rhs, rcond=None)[0]
    else:
        dual = np.zeros(len(C))

    def stationarity_residual(dual):
        adjusted = gradient+C.T.dot(dual)
        return max(np.max(np.maximum(np.abs(adjusted)-lam, 0.), initial=0.),
                   np.max(np.abs(adjusted[active]+lam*np.sign(param.beta[active])), initial=0.))

    residual = stationarity_residual(dual)
    if residual > allowance and len(C):
        from scipy.optimize import linprog
        # Minimize the infinity-norm KKT error over all equality multipliers.
        blocks = [C.T, -C.T, C[:, active].T, -C[:, active].T]
        M = np.vstack(blocks)
        fit = linprog(np.r_[np.zeros(len(C)), 1.],
                      A_ub=np.c_[M, -np.ones(len(M))],
                      b_ub=np.r_[lam-gradient, lam+gradient, rhs, -rhs],
                      bounds=[(None, None)]*len(C)+[(0., None)], method="highs")
        if not fit.success:
            raise RuntimeError("C1 certificate multiplier LP failed: " + fit.message)
        dual = fit.x[:-1]
        residual = stationarity_residual(dual)
    margin_error = np.max(np.abs(margin-param.r), initial=0.)
    equality_error = np.max(np.abs(C.dot(param.beta)), initial=0.)
    intercept_error = abs(v.sum()) if param.intercept else 0.
    if (residual > allowance or intercept_error > allowance
            or equality_error > tol*max(1., np.max(np.abs(param.beta), initial=0.))
            or margin_error > tol*max(1., np.max(np.abs(margin), initial=0.))):
        raise RuntimeError(
            "C1 path failed its KKT certificate at lambda={:.8g}: "
            "stationarity={:.3g}, intercept={:.3g}, equality={:.3g}, margins={:.3g}"
            .format(lam, residual, intercept_error, equality_error, margin_error))
    param.certificate = dict(dual=dual, stationarity=residual,
                             intercept=intercept_error, equality=equality_error,
                             margins=margin_error)
    return param.certificate


def solve_path_Conc(matrices, stop, n_active=False, lassopath=True, true_lam=False):
    """
    This functions will compute the path for all the breaking points :
    beta is a piecewise linear function of lambda, and only value on the breaking points
    is computed here

    Args :
        matrices : A (aka X),C,y the matrices of the problem
        stop : fraction of lambda_R3 that gives one criteria to stop :
            continue while lambda_R3 > stop * lambda_R3_max
            but this is the lambda of R3, which live in another space..
        n_active : another criteria to stop

    Return :
        BETA : list of beta(lambda) for lambda in LAMBDA
        LAM : list of breaking points
        R : list of residuals rescaled
    """

    (A, C, y) = matrices
    n, d, k = len(A), len(A[0]), len(C)
    # to compute r = (A beta - y)/||y|| more efficientely ; and we set reduclam=lam/stop to 2 so that if stop = 0, the condition reduclam < ||r|| is never furfilled
    A_over_NORMy, y_over_NORMy, reduclam = (
        A / (LA.norm(y)),
        y / (LA.norm(y)),
        2.0,
    )
    if lassopath:
        lamin, R = 0, [-y_over_NORMy]
    else:
        lamin, beta_old, reduclam_old, r_old = (
            0,
            np.zeros(d),
            1.0,
            -y_over_NORMy,
        )

    param = parameters_for_update(matrices, lamin, 0, "R3")
    BETA, LAM = [param.beta], [param.lam]
    for i in range(d * N_frac):

        up(param)
        BETA.append(param.beta), LAM.append(param.lam)
        param.r = A_over_NORMy.dot(param.beta) - y_over_NORMy
        if stop != 0:
            reduclam = param.lam / stop
        if lassopath:
            R.append(param.r)
            if (
                (reduclam <= LA.norm(param.r))
                or (param.number_act >= n - k)
                or (n_active > 0 and param.number_act >= n_active)
            ):
                return (BETA, LAM, R)
        else:
            if reduclam <= LA.norm(param.r):
                return (
                    (beta_old, param.beta),
                    (reduclam_old, reduclam),
                    (r_old, param.r),
                )
            beta_old, reduclam_old, r_old = param.beta, reduclam, param.r

    raise ValueError(
        "The concomitant path algorithm did not finsh after %i iterations " % N,
        "and with lamin=" + str(stop),
        " and with n_active =" + str(n_active),
    )


def pathalgo_general(matrix, path, typ, n_active=False, rho=0, intercept=False):
    """
    This function is only to interpolate the solution path between the breaking points
    """
    BETA, BETA0, i = [], [], 0
    if intercept:
        B0, B, sp_path = solve_path(
            matrix, path[-1], n_active, rho, typ, intercept=intercept
        )
        B0.append(B0[-1])
    else:
        B, sp_path = solve_path(
            matrix, path[-1], n_active, rho, typ, intercept=intercept
        )

    sp_path.append(path[-1]), B.append(B[-1])
    for lam in path:
        while lam < sp_path[i + 1]:
            i += 1
        teta = (sp_path[i] - lam) / (sp_path[i] - sp_path[i + 1])
        BETA.append(B[i] * (1 - teta) + B[i + 1] * teta)
        if intercept:
            BETA0.append(B0[i] * (1 - teta) + B0[i + 1] * teta)

    if intercept:
        BETA = np.array([[BETA0[i]] + list(BETA[i]) for i in range(len(BETA0))])

    return BETA


def up(param):
    """
    Function to call to go from a breaking point to the next one
    """
    formulation = param.formulation
    if formulation in ["R1", "R3"]:
        up_LS(param)
    elif formulation == "R2":
        up_huber(param)
    elif formulation == "C1":
        up_cl(param)
    elif formulation == "C2":
        up_huber_cl(param)
    else:
        raise ValueError(
            "Unknown formulation {}, please enter one of those : R1, R2, R3, C1, C2".format(formulation)
        )


def _active_event_product(bi, di):
    """An active coefficient reaches zero (threshold on beta_i * dbeta_i)."""
    return abs(bi * di) > 1e-10 and bi * di > 0


def _active_event_cl(bi, di):
    """An active coefficient reaches zero.

    A small coefficient can still cross zero before any other event, and a
    product threshold is not invariant to X scaling.
    """
    return abs(bi) > 1e-12 and bi * di > 0


def _skip_inactive_ls(e, s0):
    return abs(e - s0) < 1e-10


def _skip_inactive_huber(e, s0):
    return abs(e + s0) < 1e-10 or abs(s0) > 1


def _skip_inactive_huber_cl(e, s0):
    return abs(e - s0) < 1e-10 or abs(s0) > 1


def _coefficient_events(
    lam, lambdamax, activity, beta, beta_dot, lam_s_dot, s, active_event, skip_inactive
):
    """For each coefficient, the decrease of lambda at which its activity changes.

    Active coefficients leave when they reach zero; inactive ones enter when their
    subgradient reaches +-1. Coefficients without an event keep ``lam``.
    """
    L = [lam] * len(activity)
    for i in range(len(activity)):
        bi, di, e, s0 = beta[i], beta_dot[i], lam_s_dot[i], s[i]
        if activity[i]:
            if active_event(bi, di):
                L[i] = bi / (di * lambdamax)
        elif not skip_inactive(e, s0):
            dl = (1 + s0) / (1 + e) if e > s0 else (1 - s0) / (1 - e)
            L[i] = dl * lam
    return L


def _update_activity(L, dlamb, activity, number_act, idr, constraints, has_constraints):
    """Switch the coefficients whose event happens at dlamb, and the rows of C kept in idr.

    ``activity`` and ``idr`` are updated in place; the new number of active
    coefficients is returned.
    """
    for i in range(len(activity)):
        if not L[i] < dlamb + 1e-10:  # (also skips NaN, as before)
            continue
        activity[i] = not activity[i]
        number_act += 1 if activity[i] else -1
        if has_constraints:
            next_idr = next_idr1 if activity[i] else next_idr2
            to_ad = next_idr(idr, constraints[:, activity])
            if type(to_ad) == int:
                idr[to_ad] = activity[i]
    return number_act


def _advance(beta, beta_dot, s, lam_s_dot, lam, dlamb, lambdamax):
    """Move beta and the subgradient s from lam to lam - dlamb."""
    beta = beta - lambdamax * beta_dot * dlamb
    if dlamb < lam:
        s = lam_s_dot + lam / (lam - dlamb) * (s - lam_s_dot)
    return beta, s, lam - dlamb


def _next_beta0(param, A, F, beta_dot, dlamb, lambdamax):
    """Move the intercept along with beta, using the current quadratic sample set F."""
    beta0_dot = -np.vdot(np.mean(A[F], axis=0), beta_dot)
    return param.beta0 - lambdamax * beta0_dot * dlamb


def _refresh_quadratic_block(M, A, F, d, eps_L2, intercept):
    """Recompute the X^T X block of M for the samples in F."""
    if intercept:
        P = A[F] - np.mean(A[F], axis=0)
        M[:d, :][:, :d] = 2 * P.T.dot(P) + eps_L2 * np.eye(d)
    else:
        M[:d, :][:, :d] = 2 * A[F].T.dot(A[F]) + eps_L2 * np.eye(d)


def _regularised_inverse(N, eps_L2):
    try:
        return LA.inv(N)
    except LA.LinAlgError:
        return LA.inv(N + eps_L2 * np.eye(len(N)))


def _store(param, **values):
    for name, value in values.items():
        setattr(param, name, value)


def _huber_sample_event(r, ADl, rho, dlamb):
    """First sample whose residual crosses +-rho before dlamb, if any.

    Looks for 0 < dl < dlamb such that F[j] and |r[j] + ADl[j] * dl| > rho,
    or not F[j] and |r[j] + ADl[j] * dl| < rho.
    """
    huber_up, j_switch = False, None
    for j in range(len(r)):
        if abs(ADl[j]) < 1e-6 or abs(r[j]) < 1e-6 or abs(abs(r[j]) - rho) < 1e-6:
            continue
        c = -r[j] / ADl[j]
        dl = (1 - rho / abs(r[j])) * c
        if dl < 0:
            dl = (1 + rho / abs(r[j])) * c if c > 0 else dlamb
        if dl < dlamb:
            huber_up, j_switch, dlamb = True, j, dl
    return huber_up, j_switch, dlamb


def _hinge_sample_event(r, yADl, dlamb):
    """Smallest 0 < dl < dlamb at which a margin r[j] + yADl[j] * dl reaches 1.

    The boundary itself is skipped, not distinct nearby margin crossings:
    a wider band can leave a sample on the wrong loss piece.
    """
    for j in range(len(r)):
        if abs(r[j] - 1) < 1e-10:
            continue
        dl = (1 - r[j]) / yADl[j] if yADl[j] != 0.0 else -1
        if 0 < dl < dlamb:
            dlamb = dl
    return dlamb


def _huber_hinge_sample_event(r, yADl, rho, dlamb):
    """First sample whose margin reaches 1 or rho before dlamb, if any."""
    max_up, j_switch = False, None
    for j in range(len(r)):
        if yADl[j] != 0.0:
            dlmax = (1 - r[j]) / yADl[j]
            dlhuber = (rho - r[j]) / yADl[j]
        else:
            dlmax, dlhuber = dlamb + 1, dlamb + 1
        if dlmax <= 0.0:
            dlmax = dlamb + 1
        if dlhuber <= 0.0:
            dlhuber = dlamb + 1
        dl = min(dlhuber, dlmax)
        if dl < dlamb:
            max_up, j_switch, dlamb = True, j, dl
    return max_up, j_switch, dlamb


def up_LS(param):
    """
    Function to call to go from a breaking point to the next one
    """
    lambdamax, M, activity, s, lam = (
        param.lambdamax,
        param.M,
        param.activity,
        param.s,
        param.lam,
    )
    number_act, idr = param.number_act, param.idr

    d = len(activity)
    beta_dot, lam_s_dot = derivatives(
        activity, s, M[:d, :d], M[d:, :d], param.Xt, idr, number_act
    )
    L = _coefficient_events(
        lam, lambdamax, activity, param.beta, beta_dot, lam_s_dot, s,
        _active_event_product, _skip_inactive_ls,
    )
    dlamb = min(min(L), lam - param.lamin)
    # Update matrix inverse, list of rows in C and activity
    number_act = _update_activity(
        L, dlamb, activity, number_act, idr, param.C, len(M) > d
    )
    N = M[activity + idr, :][:, activity + idr]
    Xt = _regularised_inverse(N, param.eps_L2)
    beta, s, lam = _advance(param.beta, beta_dot, s, lam_s_dot, lam, dlamb, lambdamax)

    _store(param, number_act=number_act, idr=idr, Xt=Xt, activity=activity,
           beta=beta, s=s, lam=lam)


def up_huber(param):
    """
    Function to call to go from a breaking point to the next one
    """
    lambdamax, A, rho, eps_L2 = param.lambdamax, param.A, param.rho, param.eps_L2
    activity, F, s, lam, M, r = (
        param.activity,
        param.F,
        param.s,
        param.lam,
        param.M,
        param.r,
    )
    number_act, idr = param.number_act, param.idr

    d = len(activity)
    beta_dot, lam_s_dot = derivatives(
        activity, s, M[:d, :d], M[d:, :d], param.Xt, idr, number_act
    )
    L = _coefficient_events(
        lam, lambdamax, activity, param.beta, beta_dot, lam_s_dot, s,
        _active_event_product, _skip_inactive_huber,
    )
    dlamb = min(min(L), lam, lam - param.lamin)

    ADl = -A.dot(beta_dot) * lambdamax
    if param.intercept:
        ADl += np.vdot(param.Abar, beta_dot) * lambdamax
    huber_up, j_switch, dlamb = _huber_sample_event(r, ADl, rho, dlamb)

    beta, s, lam_next = _advance(
        param.beta, beta_dot, s, lam_s_dot, lam, dlamb, lambdamax
    )
    r = r + ADl * dlamb
    if param.intercept:
        param.beta0 = _next_beta0(param, A, F, beta_dot, dlamb, lambdamax)

    if huber_up:
        F[j_switch] = not F[j_switch]
        _refresh_quadratic_block(M, A, F, d, eps_L2, param.intercept)
    else:
        # Update matrix inverse, list of rows in C and activity
        number_act = _update_activity(
            L, dlamb, activity, number_act, idr, M[d:, :d], len(M) > d
        )

    N = M[activity + idr, :][:, activity + idr]
    Xt = _regularised_inverse(N, eps_L2)

    _store(param, number_act=number_act, idr=idr, Xt=Xt, activity=activity, F=F,
           beta=beta, s=s, lam=lam_next, M=M, r=r)


def up_cl(param):
    """
    Function to call to go from a breaking point to the next one
    """
    lambdamax, A, y = param.lambdamax, param.A, param.y
    activity, F, lam, M, r = param.activity, param.F, param.lam, param.M, param.r
    number_act, idr = param.number_act, param.idr
    # Roundoff at a saturated dual coordinate must not suppress its event.
    s = np.where(np.abs(np.abs(param.s) - 1.0) < 1e-10, np.sign(param.s), param.s)

    d = len(activity)
    beta_dot, lam_s_dot = derivatives(
        activity, s, M[:d, :d], M[d:, :d], param.Xt, idr, number_act
    )
    L = _coefficient_events(
        lam, lambdamax, activity, param.beta, beta_dot, lam_s_dot, s,
        _active_event_cl, _skip_inactive_huber,
    )
    dlamb = min(min(L), lam, lam - param.lamin)

    yADl = -y * A.dot(beta_dot) * lambdamax
    if param.intercept:
        # The intercept derivative uses the CURRENT quadratic sample set.
        # Using the original full-sample mean makes tracked margins drift.
        yADl += y * np.vdot(np.mean(A[F], axis=0), beta_dot) * lambdamax
    dlamb = _hinge_sample_event(r, yADl, dlamb)

    # Several samples can reach the same boundary at exactly one knot.
    # Switch all of them, including ties with a coefficient event.
    sample_switches = np.flatnonzero(
        (np.abs(r + yADl * dlamb - 1.0) <= 1e-10) & (np.abs(r - 1.0) > 1e-10)
    )

    beta, s, lam_next = _advance(
        param.beta, beta_dot, s, lam_s_dot, lam, dlamb, lambdamax
    )
    r = r + yADl * dlamb
    if param.intercept:
        param.beta0 = _next_beta0(param, A, F, beta_dot, dlamb, lambdamax)

    if len(sample_switches) > 0:
        F[sample_switches] = ~F[sample_switches]
        _refresh_quadratic_block(M, A, F, d, param.eps_L2, param.intercept)
    # Update matrix inverse, list of rows in C and activity
    number_act = _update_activity(
        L, dlamb, activity, number_act, idr, M[d:, :d], len(M) > d
    )
    N = M[activity + idr, :][:, activity + idr]
    Xt = _C1_inverse(N)

    _store(param, number_act=number_act, idr=idr, Xt=Xt, activity=activity, F=F,
           beta=beta, s=s, lam=lam_next, M=M, r=r)


def up_huber_cl(param):
    """
    Function to call to go from a breaking point to the next one
    """
    lambdamax, A, y, rho, eps_L2 = (
        param.lambdamax,
        param.A,
        param.y,
        param.rho,
        param.eps_L2,
    )
    activity, F, s, lam, M, r = (
        param.activity,
        param.F,
        param.s,
        param.lam,
        param.M,
        param.r,
    )
    number_act, idr = param.number_act, param.idr

    d = len(activity)
    beta_dot, lam_s_dot = derivatives(
        activity, s, M[:d, :d], M[d:, :d], param.Xt, idr, number_act
    )
    L = _coefficient_events(
        lam, lambdamax, activity, param.beta, beta_dot, lam_s_dot, s,
        _active_event_product, _skip_inactive_huber_cl,
    )
    dlamb = min(min(L), lam, lam - param.lamin)

    yADl = -y * A.dot(beta_dot) * lambdamax
    if param.intercept:
        yADl += y * np.vdot(param.Abar, beta_dot) * lambdamax
    max_up, j_switch, dlamb = _huber_hinge_sample_event(r, yADl, rho, dlamb)

    beta, s, lam_next = _advance(
        param.beta, beta_dot, s, lam_s_dot, lam, dlamb, lambdamax
    )
    r = r + yADl * dlamb
    if param.intercept:
        param.beta0 = _next_beta0(param, A, F, beta_dot, dlamb, lambdamax)

    if max_up:
        F[j_switch] = not F[j_switch]
        if np.any(F):
            _refresh_quadratic_block(M, A, F, d, eps_L2, param.intercept)
    else:
        # Update matrix inverse, list of rows in C and activity
        number_act = _update_activity(
            L, dlamb, activity, number_act, idr, M[d:, :d], len(M) > d
        )

    N = M[activity + idr, :][:, activity + idr]
    Xt = _regularised_inverse(N, eps_L2)

    _store(param, number_act=number_act, idr=idr, Xt=Xt, activity=activity, F=F,
           beta=beta, s=s, lam=lam_next, M=M, r=r)


# Compute the derivatives of the solution Beta and the derivative of lambda*subgradient thanks to the ODE
def derivatives(activity, s, Mat, C, Inv, idr, number_act):
    """
    Compute the derivatives of the solution Beta and the derivative of lambda*subgradient
    thanks to the following equation :

    2A[active].T A[active] beta_dot + C[active].T v_dot = s       on active variables
    C[active]   beta_dot                               = 0

     (lambda s)_dot  = 2A[inactive].T A[active] beta_dot         on inactive variables

     Then : Mat = A[inactive].T A[active]  and Inv is the inverse of a matrix of the first equaction.

    """
    beta_dot = np.zeros(len(activity))
    beta_dot[activity] = -Inv[:number_act, :number_act].dot(s[activity])
    lam_s_dot = -Mat.dot(beta_dot)

    if len(C) > 0:
        v_dot = np.zeros(len(C))
        v_dot[idr] = Inv[number_act:, :number_act].dot(s[activity])
        lam_s_dot += C.T.dot(v_dot)

    return (beta_dot, lam_s_dot)


# Upddate a list of constraints which are independant if we restrict the matrix C to the acrive set (C_A has to have independant rows)
# When we ad an active parameter
def next_idr1(liste, mat):
    # function to know which indice one should add
    if sum(liste) == len(mat):
        return False
    if sum(liste) == 0:
        for i in range(len(mat)):
            for j in range(len(mat[0])):
                if not (mat[i, j] == 0):
                    return i
        return False
    Q = LA.qr(mat[liste, :].T)[0]
    for j in range(len(mat)):
        if (not liste[j]) and (
            LA.norm(mat[j] - LA.multi_dot([Q, Q.T, mat[j]])) > 1e-10
        ):
            return j
    return False


# When we remove an active parameter
def next_idr2(liste, mat):
    if sum(liste) == 0:
        return False
    R = LA.qr(mat[liste, :].T)[1]
    for i in range(len(R)):
        if abs(R[i, i]) < 1e-10:  # looking for the i-th True element of liste
            j, somme = 0, liste[0]
            while somme <= i:
                j, somme = j + 1, somme + liste[j + 1]
            return j
    return False


def h_lambdamax(matrices, rho, typ="R1", intercept=False):
    param = parameters_for_update(matrices, 0.0, rho, typ, intercept=intercept)
    return param.lambdamax


# Compute the derivative of the huber function, particulary useful for the computing of lambdamax
def h_prime(rho, typ):
    if rho > 0 and typ == "R2":
        # huber regress
        # where grad(h)(r) is :
        # r if -rho<r<rho   ; -rho if r < -rho  ; rho if r > rho
        return lambda r: np.minimum(r, rho) + np.maximum(r, -rho) - r

    elif typ == "C2":
        # huber classification
        # where grad(h)(r) is :
        # (r-1) if 1>r>rho   ;  (rho-1) if r<rho  ;  0  if r >1
        # NF*beta0 = (1-rho)*Nminus + np.sum(y[F])
        return lambda r: np.minimum(np.maximum(r, rho) - 1.0, 0.0)

    elif typ == "C1":
        # classification
        # grad(h)(r) is :
        # (r-1) if 1>r  ; 0  if r >1
        return lambda r: np.minimum(r - 1, 0.0)

    else:
        # R1 or R3
        # grad(h)(r) is:
        # r
        return lambda r: r


def find_F(y, rho, typ):
    """
    Find initial value of F, and beta0 if intercept
    F is the set of indices for which objective is quadratic
    """
    if rho > 0 and typ == "R2":
        # huber regress
        return (y >= -rho) & (y <= rho)
    elif typ == "C2":
        # huber classify
        return (y >= rho) & (y <= 1)
    elif typ == "C1":
        return y <= 1
    else:
        # no huber
        return np.ones(len(y), dtype=bool)


def find_beta0(r, dbeta0, y, rho, typ):
    gradh = lambda b0: np.sum(dbeta0 * h_prime(rho, typ)(r(b0, y)))
    beta0 = binary_search(gradh, min(y), max(y))
    return beta0


def binary_search(f, a, b, tol=1e-8):
    c = (a + b) / 2
    # if f(a) * f(b) > 0:
    #    print("gradh(min(y)) = ", f(a))
    #    print("gradh(max(y)) = ", f(b))
    #    raise ValueError("Error in binary search for initial intercept")
    while abs(f(c)) > tol:
        if f(c) * f(a) < 0:
            b = c
        else:
            a = c
        c = (a + b) / 2
    return c


"""
Function that solve the concomitant problem for every lambda thanks to the previous function:
we firstly compute all the non concomitant Least square problems,
then we use it to find sigma and so the solution, using the equation for sigma:

sigma = || A*B(lambda*sigma) - y ||_2       (where B(lambda) is found thanks to solve_path)

            teta = (sp_path[i]-lam)/(sp_path[i]-sp_path[i+1])
"""


"""
# Update the invers of a matrix whom we add a line, which is useul to compute the derivatives
def next_inv(Xt, B, al, ligne):
    n = len(Xt)
    Yt = np.zeros((n + 1, n + 1))
    alpha = 1 / al
    B = np.array([B])
    b1 = Xt[:ligne, :][:, :ligne] + alpha * B[:, :ligne].T.dot(B[:, :ligne])
    b2 = Xt[ligne:, :][:, :ligne] + alpha * B[:, ligne:].T.dot(B[:, :ligne])
    b4 = Xt[ligne:, :][:, ligne:] + alpha * B[:, ligne:].T.dot(B[:, ligne:])
    col1 = np.concatenate((b1, -alpha * B[:, :ligne], b2), axis=0)
    col2 = np.concatenate((b2.T, -alpha * B[:, ligne:], b4), axis=0)
    col = np.concatenate(
        (-alpha * B[0, :ligne], [alpha], -alpha * B[0, ligne:]), axis=0
    )
    return np.concatenate((col1, np.array([col]).T, col2), axis=1)



def pathalgo_huber_cl(matrix, path, rho, n_active = False, intercept = False):
    return pathalgo_general(
        matrix, path, "C2", n_active=n_active, rho=rho, intercept=intercept
    )


def pathalgo_cl(matrix, path, n_active = False, intercept = False):
    return pathalgo_general(matrix, path, "C1", n_active, intercept=intercept)



"""
