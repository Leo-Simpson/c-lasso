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
        F           : F is the set where r<1 and if huber, then it is the set where rho<r<1
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
        AtA = 2 * P.T.dot(P) + eps_L2 * np.eye(d)

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
    if param.lam < lamin:
        if intercept:
            return BETA0, BETA, LAM
        else:
            return BETA, LAM
    for i in range(d * N_frac):
        up(param)
        BETA.append(param.beta), LAM.append(param.lam)
        if intercept:
            BETA0.append(param.beta0)
        if (n_active > 0 and param.number_act >= n_active) or param.lam == lamin:
            if intercept:
                return BETA0, BETA, LAM
            else:
                return BETA, LAM

    raise ValueError(
        "The path algorithm did not finsh after %i iterations " % N,
        "and with lamin=%f" % lamin,
        " and with n_active =" + str(n_active),
    )


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


def pathalgo_huber_cl(matrix, path, rho, n_active=False, intercept=False):
    return pathalgo_general(
        matrix, path, "C2", n_active=n_active, rho=rho, intercept=intercept
    )


def pathalgo_cl(matrix, path, n_active=False, intercept=False):
    return pathalgo_general(matrix, path, "C1", n_active, intercept=intercept)


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
            "Unknown formulation {}, please enter one of those : R1, R2, R3, C1, C2".format()
        )


def up_LS(param):
    """
    Function to call to go from a breaking point to the next one
    """
    # parameters that does not change
    lambdamax = param.lambdamax
    lamin = param.lamin
    M = param.M
    C = param.C
    eps_L2 = param.eps_L2

    # parameters to be updated
    number_act = param.number_act
    idr = param.idr
    Xt = param.Xt
    activity = param.activity
    beta = param.beta
    s = param.s
    lam = param.lam

    d = len(activity)
    L = [lam] * d
    Mat = M[:d, :d]
    beta_dot, lam_s_dot = derivatives(activity, s, Mat, M[d:, :d], Xt, idr, number_act)
    for i in range(d):
        bi, di, e, s0 = beta[i], beta_dot[i], lam_s_dot[i], s[i]
        if activity[i]:
            if abs(bi * di) > 1e-10 and bi * di > 0:
                L[i] = bi / (di * lambdamax)
        else:
            if abs(e - s0) < 1e-10:
                continue
            if e > s0:
                dl = (1 + s0) / (1 + e)
            else:
                dl = (1 - s0) / (1 - e)
            L[i] = dl * lam
    dlamb = min(min(L), lam - lamin)
    # Update matrix inverse, list of rows in C and activity
    for i in range(d):
        if L[i] < dlamb + 1e-10:
            if activity[i]:
                activity[i], number_act = False, number_act - 1
                if len(M) > d:
                    to_ad = next_idr2(idr, C[:, activity])
                    if type(to_ad) == int:
                        idr[to_ad] = False
            else:
                # x = M[:, activity + idr][i]
                # al = M[i, i] - np.vdot(x, Xt.dot(x))
                # if (abs(al) < 1e-10): break
                activity[i], number_act = True, number_act + 1
                if len(M) > d:
                    to_ad = next_idr1(idr, C[:, activity])
                    if type(to_ad) == int:
                        idr[to_ad] = True

    N = M[activity + idr, :][:, activity + idr]

    try:
        Xt = LA.inv(N)
    except LA.LinAlgError:
        Xt = LA.inv(N + eps_L2 * np.eye(len(N)))

    beta = beta - lambdamax * beta_dot * dlamb
    if not (lam == dlamb):
        s = lam_s_dot + lam / (lam - dlamb) * (s - lam_s_dot)
    lam -= dlamb

    param.number_act = number_act
    param.idr = idr
    param.Xt = Xt
    param.activity = activity
    param.beta = beta
    param.s = s
    param.lam = lam


def up_huber(param):
    """
    Function to call to go from a breaking point to the next one
    """

    # parameters that does not change
    lambdamax = param.lambdamax
    lamin = param.lamin
    A = param.A
    y = param.y
    C = param.C
    rho = param.rho
    eps_L2 = param.eps_L2

    # parameters to be updated
    number_act = param.number_act
    idr = param.idr
    Xt = param.Xt
    activity = param.activity
    F = param.F
    beta = param.beta
    s = param.s
    lam = param.lam
    M = param.M
    r = param.r

    d = len(activity)
    L = [lam] * d
    Mat = M[:d, :d]
    beta_dot, lam_s_dot = derivatives(activity, s, Mat, M[d:, :d], Xt, idr, number_act)
    for i in range(d):
        bi, di, e, s0 = beta[i], beta_dot[i], lam_s_dot[i], s[i]
        if activity[i]:
            if abs(bi * di) > 1e-10 and bi * di > 0:
                L[i] = bi / (di * lambdamax)
        else:
            if abs(e + s0) < 1e-10 or abs(s0) > 1:
                continue
            if e > s0:
                dl = (1 + s0) / (1 + e)
            else:
                dl = (1 - s0) / (1 - e)
            L[i] = dl * lam

    dlamb = min(min(L), lam, lam - lamin)
    huber_up = False
    ADl = -A.dot(beta_dot) * lambdamax
    if param.intercept:
        ADl += np.vdot(param.Abar, beta_dot) * lambdamax

    huber_up = False
    j_switch = None
    for j in range(len(r)):
        #     find if there is a  0< dl < dlamb such that F[j] and |r[j]+ADl[j]*dl|>rho
        # or  find if there is a  0< dl < dlamb such that not F[j] |r[j]+ADl[j]*dl|<rho
        if abs(ADl[j]) < 1e-6 or abs(r[j]) < 1e-6 or abs(abs(r[j]) - rho) < 1e-6:
            continue
        c = -r[j] / ADl[j]
        dl = (1 - rho / abs(r[j])) * c
        if dl < 0:
            if c > 0:
                dl = (1 + rho / abs(r[j])) * c
            else:
                dl = dlamb
        if dl < dlamb:
            huber_up, j_switch, dlamb = True, j, dl
    beta = beta - lambdamax * beta_dot * dlamb
    s = lam_s_dot + lam / (lam - dlamb) * (s - lam_s_dot)
    r = r + ADl * dlamb
    lam = lam - dlamb
    if param.intercept:
        beta0_dot = -np.vdot(np.mean(A[F], axis=0), beta_dot)
        beta0 = param.beta0 - lambdamax * beta0_dot * dlamb

    if huber_up:
        F[j_switch] = not F[j_switch]
        if param.intercept:
            P = A[F] - np.mean(A[F], axis=0)
            M[:d, :][:, :d] = 2 * P.T.dot(P) + eps_L2 * np.eye(d)
        else:
            M[:d, :][:, :d] = 2 * A[F].T.dot(A[F]) + eps_L2 * np.eye(d)
    else:
        # Update matrix inverse, list of rows in C and activity
        for i in range(d):
            if L[i] < dlamb + 1e-10:
                if activity[i]:
                    activity[i], number_act = False, number_act - 1
                    if len(M) > d:
                        to_ad = next_idr2(idr, M[d:, :d][:, activity])
                        if type(to_ad) == int:
                            idr[to_ad] = False
                else:
                    activity[i], number_act = True, number_act + 1
                    if len(M) > d:
                        to_ad = next_idr1(idr, M[d:, :d][:, activity])
                        if type(to_ad) == int:
                            idr[to_ad] = True

    N = M[activity + idr, :][:, activity + idr]

    try:
        Xt = LA.inv(N)
    except LA.LinAlgError:
        Xt = LA.inv(N + eps_L2 * np.eye(len(N)))

    param.number_act = number_act
    param.idr = idr
    param.Xt = Xt
    param.activity = activity
    param.F = F
    param.beta = beta
    param.s = s
    param.lam = lam
    param.M = M
    param.r = r
    if param.intercept:
        param.beta0 = beta0


def up_cl(param):
    """
    Function to call to go from a breaking point to the next one
    """

    lambdamax = param.lambdamax
    lamin = param.lamin
    A = param.A
    y = param.y
    eps_L2 = param.eps_L2

    number_act = param.number_act
    idr = param.idr
    Xt = param.Xt
    activity = param.activity
    F = param.F
    beta = param.beta
    s = param.s
    lam = param.lam
    M = param.M
    r = param.r

    d = len(activity)
    L = [lam] * d
    Mat = M[:d, :d]
    beta_dot, lam_s_dot = derivatives(activity, s, Mat, M[d:, :d], Xt, idr, number_act)
    for i in range(d):
        bi, di, e, s0 = beta[i], beta_dot[i], lam_s_dot[i], s[i]
        if activity[i]:
            if abs(bi * di) > 1e-10 and bi * di > 0:
                L[i] = bi / (di * lambdamax)
        else:
            if abs(e + s0) < 1e-10 or abs(s0) > 1:
                continue
            if e > s0:
                dl = (1 + s0) / (1 + e)
            else:
                dl = (1 - s0) / (1 - e)
            L[i] = dl * lam

    dlamb = min(min(L), lam, lam - lamin)
    max_up = False
    j_switch = None
    yADl = -y * A.dot(beta_dot) * lambdamax
    if param.intercept:
        yADl += y * np.vdot(param.Abar, beta_dot) * lambdamax
    for j in range(len(r)):
        #     find if there is a  0< dl < dlamb such that F[j] and r[j]+yADl[j]*dl > 1
        # or  find if there is a  0< dl < dlamb such that not F[j] r[j]+yADl[j]*dl < 1
        if abs(r[j] - 1) < 1e-4:
            continue
        if yADl[j] != 0.0:
            dl = (1 - r[j]) / yADl[j]
        else:
            dl = -1
        if dl < dlamb and dl > 0:
            max_up, j_switch, dlamb = True, j, dl

    beta = beta - lambdamax * beta_dot * dlamb
    s = lam_s_dot + lam / (lam - dlamb) * (s - lam_s_dot)
    r = r + yADl * dlamb
    lam = lam - dlamb
    if param.intercept:
        AbarF = np.mean(A[F], axis=0)
        beta0_dot = -np.vdot(AbarF, beta_dot)
        beta0 = param.beta0 - lambdamax * beta0_dot * dlamb

    if max_up:
        F[j_switch] = not F[j_switch]
        if param.intercept:
            P = A[F] - np.mean(A[F], axis=0)
            M[:d, :][:, :d] = 2 * P.T.dot(P) + eps_L2 * np.eye(d)
        else:
            M[:d, :][:, :d] = 2 * A[F].T.dot(A[F]) + eps_L2 * np.eye(d)
    else:
        # Update matrix inverse, list of rows in C and activity
        for i in range(d):
            if L[i] < dlamb + 1e-10:
                if activity[i]:
                    activity[i], number_act = False, number_act - 1
                    if len(M) > d:
                        to_ad = next_idr2(idr, M[d:, :d][:, activity])
                        if type(to_ad) == int:
                            idr[to_ad] = False
                else:
                    # x = M[:,activity+idr][i]
                    # al = M[i,i]-np.vdot(x,Xt.dot(x))
                    # if (abs(al)<1e-10): break
                    activity[i], number_act = True, number_act + 1
                    if len(M) > d:
                        to_ad = next_idr1(idr, M[d:, :d][:, activity])
                        if type(to_ad) == int:
                            idr[to_ad] = True

    N = M[activity + idr, :][:, activity + idr]

    try:
        Xt = LA.inv(N)
    except LA.LinAlgError:
        Xt = LA.inv(N + eps_L2 * np.eye(len(N)))

    param.number_act = number_act
    param.idr = idr
    param.Xt = Xt
    param.activity = activity
    param.F = F
    param.beta = beta
    param.s = s
    param.lam = lam
    param.M = M
    param.r = r
    if param.intercept:
        param.beta0 = beta0


def up_huber_cl(param):
    """
    Function to call to go from a breaking point to the next one
    """

    # parameters that does not change
    lambdamax = param.lambdamax
    lamin = param.lamin
    A = param.A
    y = param.y
    rho = param.rho
    eps_L2 = param.eps_L2

    # parameters to be updated
    number_act = param.number_act
    idr = param.idr
    Xt = param.Xt
    activity = param.activity
    F = param.F
    beta = param.beta
    s = param.s
    lam = param.lam
    M = param.M
    r = param.r

    d = len(activity)
    L = [lam] * d
    Mat = M[:d, :d]
    beta_dot, lam_s_dot = derivatives(
        activity,
        s,
        Mat,
        M[d:, :d],
        Xt,
        idr,
        number_act,
    )
    for i in range(d):
        bi, di, e, s0 = beta[i], beta_dot[i], lam_s_dot[i], s[i]
        if activity[i]:
            if abs(bi * di) > 1e-10 and bi * di > 0:
                L[i] = bi / (di * lambdamax)
        else:
            if abs(e - s0) < 1e-10 or abs(s0) > 1:
                continue
            if e > s0:
                dl = (1 + s0) / (1 + e)
            else:
                dl = (1 - s0) / (1 - e)
            L[i] = dl * lam

    dlamb = min(min(L), lam, lam - lamin)
    max_up = False
    j_switch = None
    yADl = -y * A.dot(beta_dot) * lambdamax
    if param.intercept:
        yADl += y * np.vdot(param.Abar, beta_dot) * lambdamax
    for j in range(len(r)):
        #     find if there is a  0< dl < dlamb such that F[j] and r[j]+yADl[j]*dl > 1
        # or  find if there is a  0< dl < dlamb such that not F[j] r[j]+yADl[j]*dl < 1
        #     find if there is a  0< dlhuber < dlamb such that F[j] and r[j]+yADl[j]*dl < rho
        # or  find if there is a  0< dlhuber < dlamb such that not F[j] r[j]+yADl[j]*dl > rho

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

    beta = beta - lambdamax * beta_dot * dlamb
    s = lam_s_dot + lam / (lam - dlamb) * (s - lam_s_dot)
    r = r + yADl * dlamb
    lam = lam - dlamb
    if param.intercept:
        beta0_dot = -np.vdot(np.mean(A[F], axis=0), beta_dot)
        beta0 = param.beta0 - lambdamax * beta0_dot * dlamb

    if max_up:
        F[j_switch] = not F[j_switch]
        if param.intercept:
            P = A[F] - np.mean(A[F], axis=0)
            M[:d, :][:, :d] = 2 * P.T.dot(P) + eps_L2 * np.eye(d)
        else:
            M[:d, :][:, :d] = 2 * A[F].T.dot(A[F]) + eps_L2 * np.eye(d)

    else:
        # Update matrix inverse, list of rows in C and activity
        for i in range(d):
            if L[i] < dlamb + 1e-10:
                if activity[i]:
                    activity[i], number_act = False, number_act - 1
                    if len(M) > d:
                        to_ad = next_idr2(idr, M[d:, :d][:, activity])
                        if type(to_ad) == int:
                            idr[to_ad] = False
                else:
                    # x = M[:,activity+idr][i]
                    # al = M[i,i]-np.vdot(x,Xt.dot(x))
                    # if (abs(al)<1e-10): break
                    activity[i], number_act = True, number_act + 1
                    if len(M) > d:
                        to_ad = next_idr1(idr, M[d:, :d][:, activity])
                        if type(to_ad) == int:
                            idr[to_ad] = True

    N = M[activity + idr, :][:, activity + idr]

    try:
        Xt = LA.inv(N)
    except LA.LinAlgError:
        Xt = LA.inv(N + eps_L2 * np.eye(len(N)))

    param.number_act = number_act
    param.idr = idr
    param.Xt = Xt
    param.activity = activity
    param.F = F
    param.beta = beta
    param.s = s
    param.lam = lam
    param.M = M
    param.r = r
    if param.intercept:
        param.beta0 = beta0


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
        return (y > -rho) & (y < rho)
    elif typ == "C2":
        # huber classify
        return (y > rho) & (y < 1)
    elif typ == "C1":
        return y < 1
    else:
        # no huber
        return np.ones(len(y), dtype=bool)


def find_beta0(r, dbeta0, y, rho, typ):
    gradh = lambda b0: np.sum(dbeta0 * h_prime(rho, typ)(r(b0, y)))
    beta0 = binary_search(gradh, min(y), max(y))
    return beta0


def binary_search(f, a, b, tol=1e-8):
    c = (a + b) / 2
    if f(a) * f(b) > 0:
        print("gradh(min(y)) = ", f(a))
        print("gradh(max(y)) = ", f(b))
        raise ValueError("Error in binary search for initial intercept")
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
