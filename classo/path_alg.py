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
        if typ == "C2" and (not np.isfinite(rho) or rho >= 1):
            raise ValueError(
                "For huberized hinge, rho has to be smaller than 1, but here it is :",
                rho,
            )

        if typ == "R2" and (not np.isfinite(rho) or rho <= 0):
            raise ValueError("Huber regression requires finite rho > 0")
        (self.A, self.C, self.y) = matrices
        self.lamin = lamin
        self.rho = rho
        self.formulation = typ
        self.intercept = intercept
        n, d, k = len(self.A), len(self.A[0]), len(self.C)
        self.number_act = 0
        # Classification follows the specified unregularized loss. A hidden
        # ridge changes directions and can change which margin event is next.
        if typ in ["R1", "R2", "C1", "C2"]:
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
            self.beta0 = (float(np.mean(self.y)) if typ in ["R1", "C1"]
                          else find_beta0(r_func, dr, self.y, rho, typ))
            if typ == "C2":
                # Refine the piecewise-linear intercept score equation exactly
                # within its current interval after binary search.
                for _ in range(4):
                    margin = self.y*self.beta0
                    interior = (margin > rho) & (margin < 1.)
                    if not np.any(interior): break
                    self.beta0 -= np.sum(self.y*h_prime(rho,typ)(margin))/interior.sum()
            if typ == "R2":
                for _ in range(4):
                    residual = self.beta0-self.y
                    interior = np.abs(residual) < rho
                    if not np.any(interior): break
                    self.beta0 -= np.clip(residual,-rho,rho).sum()/interior.sum()
            self.Abar = np.mean(self.A, axis=0)
        else:
            self.beta0 = 0.0

        self.F = find_F(r_func(self.beta0, self.y), rho, typ)
        P = self.A[self.F]
        if intercept and len(P):
            P = P - np.mean(P, axis=0)
        AtA = P.T.dot(P) + eps_L2 * np.eye(d)

        self.r = r_func(self.beta0, self.y)
        s = -2 * self.A.T.dot(dr * h_prime(rho, typ)(self.r))
        self.lambdamax = LA.norm(s, np.inf)
        self.lamin = lamin
        self.s = s / self.lambdamax if self.lambdamax else np.zeros_like(s)
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
        if typ in ["R1", "R2", "C1", "C2"]:
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
    if typ in ["R1", "R2", "C1", "C2"]:
        check_C1_certificate(param)
    if typ in ["R1", "R2", "C1", "C2"] and param.lambdamax == 0:
        if lamin != 1.:
            BETA.append(param.beta.copy()); LAM.append(lamin)
            if intercept: BETA0.append(param.beta0)
        return (BETA0, BETA, LAM) if intercept else (BETA, LAM)
    if param.lam < lamin:
        if intercept:
            return BETA0, BETA, LAM
        else:
            return BETA, LAM
    for i in range(d * N_frac):

        still_F = np.any(param.F)
        too_active = n_active > 0 and param.number_act >= n_active

        if (not still_F and typ not in ["R1", "R2", "C1", "C2"]) or too_active or param.lam == lamin:
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

        if typ in ["R1", "R2", "C1", "C2"]:
            # Preserve only the primal state, not a copy of the dense Hessian.
            # A numerically rejected legacy candidate must never enter BETA.
            previous = (param.beta.copy(), param.beta0, param.r.copy(), param.lam)
        up(param)
        if typ in ["C1", "C2"]:
            try:
                check_C1_certificate(param)
            except RuntimeError as error:
                if not str(error).startswith(typ+" path failed its KKT certificate"):
                    raise
                param.beta, param.beta0, param.r, param.lam = previous
                _scalar_classification_step(param)
                check_C1_certificate(param)
        if typ in ["R1", "R2"]:
            try:
                check_C1_certificate(param)
            except RuntimeError:
                param.beta, param.beta0, param.r, param.lam = previous
                from ._regression_path import step
                step(param)
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
    """Check actual scalar loss KKT conditions at a returned knot.

    First try the small active stationarity multiplier solve. If its choice
    of nonunique multipliers fails, minimize the residual over all multipliers
    with a small LP. Never confuse a failed multiplier candidate with failure
    of the primal solution. No coefficient refit or numerical ridge is used.
    """
    A, C, y = param.A, param.C, param.y
    regression = param.formulation in ["R1", "R2"]
    score = A.dot(param.beta)+param.beta0
    margin = score-y if regression else y*score
    v = 2*h_prime(param.rho, param.formulation)(margin)
    if not regression: v *= y
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
            raise RuntimeError(param.formulation+" certificate multiplier LP failed: " + fit.message)
        dual = fit.x[:-1]
        residual = stationarity_residual(dual)
    margin_error = np.max(np.abs(margin-param.r), initial=0.)
    equality_error = np.max(np.abs(C.dot(param.beta)), initial=0.)
    intercept_error = abs(v.sum()) if param.intercept else 0.
    if (residual > allowance or intercept_error > allowance
            or equality_error > tol*max(1., np.max(np.abs(param.beta), initial=0.))
            or margin_error > tol*max(1., np.max(np.abs(margin), initial=0.))):
        raise RuntimeError(
            param.formulation+" path failed its KKT certificate at lambda={:.8g}: "
            "stationarity={:.3g}, intercept={:.3g}, equality={:.3g}, margins={:.3g}"
            .format(lam, residual, intercept_error, equality_error, margin_error))
    param.certificate = dict(dual=dual, stationarity=residual,
                             intercept=intercept_error, equality=equality_error,
                             margins=margin_error)
    return param.certificate


def solve_path_Conc(matrices, stop, n_active=False, lassopath=True, true_lam=False):
    """Trace the certified R1 path until its concomitant parameter reaches stop.

    If t is the relative least-squares penalty and R its residual divided by
    ||y||, the relative concomitant penalty is t/||R||. No numerical ridge
    belongs in this auxiliary least-squares path.
    """
    A, C, y = matrices
    size = LA.norm(y)
    zero = np.zeros(A.shape[1])
    if size == 0:
        if lassopath: return [zero, zero.copy()], [1., 0.], [np.zeros(len(y))]*2
        return (zero, zero.copy()), (0., 0.), (np.zeros(len(y)), np.zeros(len(y)))
    param = parameters_for_update(matrices, 0., 0., "R1")
    check_C1_certificate(param)
    BETA, LAM, R = [param.beta.copy()], [param.lam], [-y/size]
    for _ in range(A.shape[1]*N_frac):
        if param.lam <= stop*LA.norm(R[-1]) or param.lam == 0.:
            break
        previous = (param.beta.copy(), param.beta0, param.r.copy(), param.lam)
        up(param)
        try:
            check_C1_certificate(param)
        except RuntimeError:
            param.beta, param.beta0, param.r, param.lam = previous
            from ._regression_path import step
            step(param)
            check_C1_certificate(param)
        BETA.append(param.beta.copy()); LAM.append(param.lam)
        R.append((A@param.beta-y)/size)
        if n_active and np.count_nonzero(abs(param.beta)>1e-9)>=n_active:
            break
    else:
        raise RuntimeError("Concomitant path exceeded its knot limit")
    if lassopath: return BETA, LAM, R
    if len(BETA)==1:
        return (BETA[0], BETA[0]), (1., 1.), (R[0], R[0])
    return (BETA[-2], BETA[-1]), (LAM[-2]/stop, LAM[-1]/stop), (R[-2], R[-1])


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
        while i+1<len(sp_path)-1 and (lam<sp_path[i+1] or sp_path[i]==sp_path[i+1]):
            i += 1
        span = sp_path[i]-sp_path[i+1]
        teta = (sp_path[i]-lam)/span if span else 1.
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
    if param.formulation == "R1":
        zero = np.asarray(activity)&(np.abs(beta)<=1e-9)
        if np.any(zero & (beta_dot*s>1e-10*max(1.,np.max(abs(beta_dot),initial=0.)))):
            from ._regression_path import step
            return step(param)
    for i in range(d):
        bi, di, e, s0 = beta[i], beta_dot[i], lam_s_dot[i], s[i]
        if activity[i]:
            # Sign comparisons avoid a dimensional product threshold (and
            # product underflow). The relative-lambda crossing is invariant
            # when both beta and lambdamax change response units.
            if (bi > 0 and di > 0) or (bi < 0 and di < 0):
                L[i] = (bi / lambdamax) / di
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

    if param.formulation == "R1":
        Xt = _C1_inverse(N)
    else:
        try:
            Xt = LA.inv(N)
        except LA.LinAlgError:
            Xt = LA.inv(N + eps_L2 * np.eye(len(N)))

    beta = beta - lambdamax * beta_dot * dlamb
    if dlamb < lam:
        s = lam_s_dot + lam / (lam - dlamb) * (s - lam_s_dot)
    lam -= dlamb

    param.number_act = number_act
    param.idr = idr
    param.Xt = Xt
    param.activity = activity
    param.beta = beta
    if param.formulation == "R1":
        if param.intercept: param.beta0 = np.mean(param.y)-param.Abar@beta
        param.r = param.A@beta+param.beta0-param.y
    param.s = s
    param.lam = lam


def up_huber(param):
    """Advance the exact Huber regression path, including tied/flat events."""
    from ._regression_path import step
    return step(param)


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
    # Roundoff at a saturated dual coordinate must not suppress its event.
    s = np.where(np.abs(np.abs(param.s)-1.) < 1e-10, np.sign(param.s), param.s)
    lam = param.lam
    M = param.M
    r = param.r

    d = len(activity)
    L = [lam] * d
    Mat = M[:d, :d]
    beta_dot, lam_s_dot = derivatives(activity, s, Mat, M[d:, :d], Xt, idr, number_act)
    # At a tied margin event, choose the outgoing quadratic side rather
    # than waiting for a crossing that starts exactly on the boundary.
    for _ in range(20):
        rate = -y*A.dot(beta_dot)
        if param.intercept:
            rate += y*np.vdot(np.mean(A[F],axis=0),beta_dot)
        chosen = F.copy()
        boundary = np.abs(r-1.) <= 1e-10
        chosen[boundary & (rate > 1e-12)] = False
        chosen[boundary & (rate < -1e-12)] = True
        if np.array_equal(chosen,F):
            break
        F = chosen
        if not np.any(F):
            from ._classification_path import step
            return step(param)
        P = A[F]-np.mean(A[F],axis=0) if param.intercept else A[F]
        M[:d,:d] = 2*P.T.dot(P)
        reduced = M[activity+idr,:][:,activity+idr]
        Xt = _C1_inverse(reduced)
        beta_dot,lam_s_dot = derivatives(activity,s,M[:d,:d],M[d:,:d],Xt,idr,number_act)
    else:
        from ._classification_path import step
        return step(param)
    # A dual-tight zero coefficient is a candidate face, not an automatic
    # entry. The descending direction must respect its subgradient sign.
    zero_active = np.asarray(activity) & (np.abs(beta) <= 1e-9)
    if np.any(zero_active & (beta_dot*s > 1e-10*max(1.,np.max(abs(beta_dot))))):
        from ._classification_path import step
        return step(param)
    for i in range(d):
        bi, di, e, s0 = beta[i], beta_dot[i], lam_s_dot[i], s[i]
        if activity[i]:
            # A small coefficient can still cross zero before any other
            # event. A product threshold here is not invariant to X scaling.
            if abs(bi) > 1e-12 and bi * di > 0:
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
    yADl = -y * A.dot(beta_dot) * lambdamax
    if param.intercept:
        # The intercept derivative uses the CURRENT quadratic sample set.
        # Using the original full-sample mean makes tracked margins drift.
        yADl += y * np.vdot(np.mean(A[F], axis=0), beta_dot) * lambdamax
    for j in range(len(r)):
        #     find if there is a  0< dl < dlamb such that F[j] and r[j]+yADl[j]*dl > 1
        # or  find if there is a  0< dl < dlamb such that not F[j] r[j]+yADl[j]*dl < 1
        # Skip the boundary itself, not distinct nearby margin crossings.
        # A 1e-4 band can leave a sample on the wrong loss piece.
        if abs(r[j] - 1) < 1e-10:
            continue
        if yADl[j] != 0.0:
            dl = (1 - r[j]) / yADl[j]
        else:
            dl = -1
        if dl < dlamb and dl > 0:
            dlamb = dl

    # Several samples can reach the same boundary at exactly one knot.
    # Switch all of them, including ties with a coefficient event.
    sample_switches = np.flatnonzero(
        (np.abs(r + yADl*dlamb - 1.) <= 1e-10)
        & (np.abs(r - 1.) > 1e-10))
    max_up = len(sample_switches) > 0

    beta = beta - lambdamax * beta_dot * dlamb
    if dlamb < lam:
        s = lam_s_dot + lam / (lam - dlamb) * (s - lam_s_dot)
    r = r + yADl * dlamb
    lam = lam - dlamb
    if param.intercept:
        AbarF = np.mean(A[F], axis=0)
        beta0_dot = -np.vdot(AbarF, beta_dot)
        beta0 = param.beta0 - lambdamax * beta0_dot * dlamb

    if max_up:
        F[sample_switches] = ~F[sample_switches]
        # sufficient :
        # F = F | (r < 1. - 1e-10)
        # necessary :
        # F = F & (r <= 1. + 1e-10)
        if param.intercept:
            P = A[F] - np.mean(A[F], axis=0)
            M[:d, :][:, :d] = 2 * P.T.dot(P) + eps_L2 * np.eye(d)
        else:
            M[:d, :][:, :d] = 2 * A[F].T.dot(A[F]) + eps_L2 * np.eye(d)
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

    Xt = _C1_inverse(N)

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


def _up_C2(param):
    """
    Function to call to go from a breaking point to the next one
    """
    if not np.any(param.F):
        return _classification_flat_step(param)
    if param.formulation=="C2" and getattr(param,'flat_events',None):
        # A previous flat event does not imply that the current face is flat.
        # First try a certified curved segment; pivot only if it cannot move.
        return _scalar_classification_step(param)
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
    boundaries=(param.rho,1.) if param.formulation=="C2" else (1.,)
    # Roundoff at a saturated dual coordinate must not suppress its event.
    s = np.where(np.abs(np.abs(param.s)-1.) < 1e-10, np.sign(param.s), param.s)
    lam = param.lam
    M = param.M
    r = param.r

    d = len(activity)
    L = [lam] * d
    Mat = M[:d, :d]
    beta_dot, lam_s_dot = derivatives(activity, s, Mat, M[d:, :d], Xt, idr, number_act)
    if param.formulation in ("C1","C2"):
        # At rho/one choose the outgoing one-sided quadratic region before
        # solving event times, including starts exactly at a hinge boundary.
        for _ in range(20):
            rate=-y*A.dot(beta_dot)
            if param.intercept:rate+=y*np.vdot(np.mean(A[F],axis=0),beta_dot)
            chosen=F.copy()
            at_lower=(np.abs(r-param.rho)<=1e-10 if param.formulation=="C2" else np.zeros(len(r),bool))
            at_upper=np.abs(r-1.)<=1e-10
            chosen[at_lower&(rate < -1e-12)]=False
            chosen[at_lower&(rate > 1e-12)]=True
            chosen[at_upper&(rate > 1e-12)]=False
            chosen[at_upper&(rate < -1e-12)]=True
            if np.array_equal(chosen,F):break
            F=chosen
            if not np.any(F):
                param.F=F
                return _classification_flat_step(param)
            P=A[F]-np.mean(A[F],axis=0) if param.intercept else A[F]
            M[:d,:d]=2*P.T.dot(P)
            N=M[activity+idr,:][:,activity+idr]
            Xt=_C1_inverse(N)
            beta_dot,lam_s_dot=derivatives(activity,s,M[:d,:d],M[d:,:d],Xt,idr,number_act)
        else:
            # Coupled sample boundaries may require a joint cone pivot rather
            # than alternating one-at-a-time region assignments.
            return _scalar_classification_step(param)
    # A dual-tight zero coefficient is only a candidate entering face.
    # The equality-constrained linear solve can point into its forbidden
    # sign (especially at simultaneous entries / dependent tree columns).
    # Select the outgoing critical cone instead of taking that invalid step.
    zero_active=np.asarray(activity)&(np.abs(beta)<=1e-9)
    if np.any(zero_active & (beta_dot*s>1e-10*max(1.,np.max(abs(beta_dot))))):
        return _scalar_classification_step(param)
    for i in range(d):
        bi, di, e, s0 = beta[i], beta_dot[i], lam_s_dot[i], s[i]
        if activity[i]:
            # A small coefficient can still cross zero before any other
            # event. A product threshold here is not invariant to X scaling.
            if abs(bi) > 1e-12 and bi * di > 0:
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
    yADl = -y * A.dot(beta_dot) * lambdamax
    if param.intercept:
        # The intercept derivative uses the CURRENT quadratic sample set.
        # Using the original full-sample mean makes tracked margins drift.
        yADl += y * np.vdot(np.mean(A[F], axis=0), beta_dot) * lambdamax
    for j in range(len(r)):
        for boundary in boundaries:
            if abs(r[j]-boundary)<1e-10 or yADl[j]==0.:
                continue
            crossing=(boundary-r[j])/yADl[j]
            if 0<crossing<dlamb:dlamb=crossing
    # Toggle all sample events, including ties with coefficient events.
    sample_switches=np.flatnonzero(np.any([
        (np.abs(r+yADl*dlamb-boundary)<=1e-10)
        & (np.abs(r-boundary)>1e-10) for boundary in boundaries],axis=0))
    max_up=len(sample_switches)>0

    beta = beta - lambdamax * beta_dot * dlamb
    if dlamb < lam:
        s = lam_s_dot + lam / (lam - dlamb) * (s - lam_s_dot)
    r = r + yADl * dlamb
    lam = lam - dlamb
    if param.intercept:
        AbarF = np.mean(A[F], axis=0)
        beta0_dot = -np.vdot(AbarF, beta_dot)
        beta0 = param.beta0 - lambdamax * beta0_dot * dlamb

    if max_up:
        F[sample_switches] = ~F[sample_switches]
        # sufficient :
        # F = F | (r < 1. - 1e-10)
        # necessary :
        # F = F & (r <= 1. + 1e-10)
        if param.intercept and np.any(F):
            P = A[F] - np.mean(A[F], axis=0)
            M[:d, :][:, :d] = 2 * P.T.dot(P) + eps_L2 * np.eye(d)
        else:
            M[:d, :][:, :d] = 2 * A[F].T.dot(A[F]) + eps_L2 * np.eye(d)
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

    Xt = _C1_inverse(N)

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


def _refresh_classification_state(param):
    """Rebuild the scalar active KKT state after a certified path event."""
    A,C,y=param.A,param.C,param.y
    d=A.shape[1]
    regression = param.formulation in ["R1", "R2"]
    score = A.dot(param.beta)+param.beta0
    param.r = score-y if regression else y*score
    # LP endpoints can lie a few ulps outside a margin boundary. Include
    # those boundary samples; outgoing selection chooses their future side.
    param.F=param.r<=1.+1e-9
    if param.formulation=="C2":param.F&=param.r>=param.rho-1e-9
    if param.formulation == "R1": param.F = np.ones(len(y),bool)
    if param.formulation == "R2": param.F = np.abs(param.r)<=param.rho+1e-9
    P=A[param.F]
    if param.intercept and len(P):P=P-P.mean(axis=0)
    H=2*P.T.dot(P)
    param.M=np.block([[H,C.T],[C,np.zeros((len(C),len(C)))]])
    certificate=check_C1_certificate(param)
    values=2*h_prime(param.rho,param.formulation)(param.r)
    gradient=A.T.dot(values if regression else y*values)
    absolute=param.lam*param.lambdamax
    param.s=(-(gradient+C.T.dot(certificate['dual']))/absolute if absolute>0
             else np.sign(param.beta))
    active=(np.abs(param.beta)>1e-9)
    # Tight zero coordinates can enter immediately on the next curved segment.
    if absolute>0:active|=np.abs(param.s)>=1-1e-9
    param.activity=active.tolist()
    param.number_act=int(active.sum())
    param.idr=[False]*len(C)
    if len(C) and np.any(active):
        from scipy.linalg import qr
        _,R,pivots=qr(C[:,active].T,mode='economic',pivoting=True)
        diag=np.abs(np.diag(R))
        rank=int(np.sum(diag>1e-10*max(1.,diag.max(initial=0.))))
        for j in pivots[:rank]:param.idr[int(j)]=True
    selected=param.activity+param.idr
    param.Xt=_C1_inverse(param.M[selected,:][:,selected])


def _classification_flat_step(param,allow_no_progress=False):
    """Continue a no-curvature C1/C2 state without adding a ridge.

    First minimize lambda over KKT multipliers with the primal point fixed.
    If lambda cannot decrease, maximize the L1 norm over the current optimal
    polyhedral loss cell: a lower lambda selects the largest norm on this face.
    This is a primal pivot at the SAME lambda, not interpolation across a jump.
    """
    from scipy.optimize import linprog
    A,C,y=param.A,param.C,param.y
    n,d=A.shape;k=len(C)
    margin=y*(A.dot(param.beta)+param.beta0)
    gradient=A.T.dot(2*y*h_prime(param.rho,param.formulation)(margin))
    absolute=param.lam*param.lambdamax
    target=param.lamin*param.lambdamax
    active=np.abs(param.beta)>1e-9
    # Variables [constraint multipliers, absolute lambda].
    inequalities=np.vstack([np.c_[C.T,-np.ones(d)],np.c_[-C.T,-np.ones(d)]])
    equalities=np.c_[C[:,active].T,np.sign(param.beta[active])]
    fit=linprog(np.r_[np.zeros(k),1.],A_ub=inequalities,b_ub=np.r_[-gradient,gradient],
                A_eq=equalities if np.any(active) else None,
                b_eq=-gradient[active] if np.any(active) else None,
                bounds=[(None,None)]*k+[(0.,absolute)],method='highs',
                options=dict(primal_feasibility_tolerance=1e-9,dual_feasibility_tolerance=1e-9))
    if not fit.success:raise RuntimeError('Classification flat-state certificate LP failed: '+fit.message)
    lower=float(fit.x[-1]);next_lambda=max(target,lower)
    if next_lambda<absolute-1e-10*max(1.,absolute):
        param.lam=(param.lamin if next_lambda==target or not param.lambdamax
                   else next_lambda/param.lambdamax)
        _refresh_classification_state(param)
        param.flat_events=getattr(param,'flat_events',[])+[dict(kind='constant_primal',lambda_value=next_lambda)]
        return True
    if absolute<=1e-12:
        param.lam=param.lamin
        _refresh_classification_state(param)
        return True
    # A flat optimum may occupy a whole interval of coefficient values at a
    # critical lambda. Choose its descending-path endpoint by an exact LP.
    offset=int(param.intercept);q=d+offset
    Z=np.c_[np.ones(n),A] if param.intercept else A
    signed=y[:,None]*Z
    if param.formulation=='C2':
        low=margin<=param.rho+1e-9
        high=margin>=1.-1e-9
        curved=~(low|high)
        loss_linear=2*(param.rho-1)*signed[low].sum(axis=0)
        loss_constant=low.sum()*(1-param.rho**2)+np.sum((1-margin[curved])**2)
        loss=np.where(margin<param.rho,(1-param.rho)**2+2*(1-param.rho)*(param.rho-margin),np.maximum(1-margin,0)**2).sum()
    else:
        low=np.zeros(n,bool);high=margin>=1.-1e-9;curved=~high
        loss_linear=np.zeros(q);loss_constant=np.sum((1-margin[curved])**2);loss=np.maximum(1-margin,0).dot(np.maximum(1-margin,0))
    F=np.c_[np.zeros((d,offset)),np.eye(d)]
    rows=[np.c_[F,-np.eye(d)],np.c_[-F,-np.eye(d)]]
    rhs=[np.zeros(d),np.zeros(d)]
    if np.any(low):rows.append(np.c_[signed[low],np.zeros((low.sum(),d))]);rhs.append(np.full(low.sum(),param.rho))
    if np.any(high):rows.append(np.c_[-signed[high],np.zeros((high.sum(),d))]);rhs.append(-np.ones(high.sum()))
    tree=np.c_[np.zeros((k,offset)),C,np.zeros((k,d))]
    fixed_scores=np.c_[signed[curved],np.zeros((curved.sum(),d))]
    optimality=np.r_[loss_linear,np.full(d,absolute)]
    objective_value=loss+absolute*np.abs(param.beta).sum()
    pivot=linprog(np.r_[np.zeros(q),-np.ones(d)],A_ub=np.vstack(rows),b_ub=np.concatenate(rhs),
                  A_eq=np.vstack([tree,fixed_scores,optimality]),b_eq=np.r_[np.zeros(k),margin[curved],objective_value-loss_constant],
                  bounds=[(None,None)]*q+[(0.,None)]*d,method='highs',
                  options=dict(primal_feasibility_tolerance=1e-9,dual_feasibility_tolerance=1e-9))
    if not pivot.success:raise RuntimeError('Classification optimal-face pivot LP failed: '+pivot.message)
    old=np.r_[param.beta0,param.beta] if param.intercept else param.beta.copy()
    if (np.max(np.abs(pivot.x[:q]-old),initial=0.)<=1e-9
            or (allow_no_progress and np.abs(pivot.x[offset:q]).sum()<=np.abs(param.beta).sum()+1e-9)):
        if allow_no_progress:return False
        raise RuntimeError('Classification flat-face pivot made no primal or lambda progress')
    previous_beta, previous_bias = param.beta.copy(), param.beta0
    param.beta=pivot.x[offset:q].copy()
    if param.intercept:param.beta0=float(pivot.x[0])
    try:
        _refresh_classification_state(param)
    except RuntimeError:
        param.beta, param.beta0 = previous_beta, previous_bias
        _refresh_classification_state(param)
        if allow_no_progress: return False
        raise
    if not np.any(param.F):raise RuntimeError('Classification flat-face pivot did not reach a curvature boundary')
    param.flat_events=getattr(param,'flat_events',[])+[dict(kind='linear_face_pivot',lambda_value=absolute)]
    return True


def _scalar_classification_step(param):
    from ._classification_path import step
    if param.formulation != "C2":
        return step(param)
    # Most post-flat knots have a perfectly valid curved direction. Solving
    # two LPs before every such segment dominated the long C2 path runtime.
    previous = (param.beta.copy(), param.beta0, param.lam)
    try:
        return step(param)
    except (LA.LinAlgError, RuntimeError):
        # A singular/unbounded cone or an unresolved zero-length segment can
        # require a same-lambda optimal-face pivot. Restore the primal state
        # before this fallback, including failures during state rebuilding.
        param.beta, param.beta0, param.lam = previous
        _refresh_classification_state(param)
        if _classification_flat_step(param, allow_no_progress=True):
            return
        raise


def up_huber_cl(param):
    """C2 events at both rho and one, using the current quadratic set."""
    return _up_C2(param)


def check_C2_certificate(param, tol=1e-7):
    return check_C1_certificate(param, tol)



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
