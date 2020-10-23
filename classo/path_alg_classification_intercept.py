"""
    To do to add an intercept : 

    Update it as well in parameters_for_update : init lines 70 , update  lines 173

    extend matrix M , and update it : line 65 + be cautious about indices of M everywhere

    also change residual r : lines 165-179

    MAybe some other changes... 


    How the equation will look like :
    
    Let F = y * (Abeta+betaO) < 1
    
    Let m = mean(A[F,:], axis=0)

    Let N  = A[F,:] - m
    
    betaO_dot =  - m. beta_dot and
    
    2N.T.dot(N)  beta_dot      C.T v_dot = s     on active variables (beta_dot elsewhere)
    
            C    beta_dot      0   v_dot = 0


    and
    (lambda s)_dot  = N.T.N beta_dot   + C.T v_dot              on inactive variables





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


    """

    def __init__(self, matrices, lamin, eps_L2=1e-3):

        (self.A, self.C, self.y) = matrices
        self.lamin = lamin
        n, d, k = len(self.A), len(self.A[0]), len(self.C)
        self.number_act = 0
        self.idr = [False] * k
        self.activity = [False] * d
        self.beta = np.zeros(d)
        s = 2 * self.A.T.dot(self.y)
        self.lambdamax = LA.norm(s, np.inf)
        self.lamin = lamin
        self.s = s / self.lambdamax
        self.lam = 1.0
        self.r = -self.y
        self.F = [True] * n
        self.eps_L2 = eps_L2
        AtA = self.A[self.F].T.dot(self.A[self.F]) + self.eps_L2 * np.eye(d)
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

        try:
            self.Xt = LA.inv(
                self.M[self.activity + self.idr, :][:, self.activity + self.idr]
            )
        except LA.LinAlgError:
            self.Xt = LA.inv(
                self.M[self.activity + self.idr, :][:, self.activity + self.idr]
                + np.diag([self.eps_L2] * sum(self.activity) + [0] * sum(self.idr))
            )


def solve_path(matrices, lamin, n_active):
    """
    This functions will compute the path for all the breaking points :
    beta is a piecewise linear function of lambda, and only value on the breaking points
    is computed here

    Args :
        matrices : A (aka X),C,y the matrices of the problem
        lamin : fraction of lambda that gives one criteria to stop : continue while lambda > lamin * lambda_max
        n_active : another criteria to stop

    Return :
        BETA : list of beta(lambda) for lambda in LAMBDA
        LAMBDA : list of breaking points
    """
    d = len(matrices[0][0])
    param = parameters_for_update(matrices, lamin)
    BETA, LAM = [param.beta], [param.lam]
    if param.lam < lamin:
        return BETA, LAM
    for i in range(d * N_frac):
        up(param)
        BETA.append(param.beta), LAM.append(param.lam)
        if (n_active > 0 and param.number_act >= n_active) or param.lam == lamin:
            return (BETA, LAM)

    raise ValueError(
        "The path algorithm did not finsh after %i iterations " % N,
        "and with lamin=%f" % lamin,
        " and with n_active =" + str(n_active),
    )


def up(param):
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
    C_reduc = M[d:, :][:, :d]
    Mat = M[: len(activity), :][:, : len(activity)]
    beta_dot, s_dot = derivatives(activity,s,Mat,C_reduc, Xt, idr,number_act)
    for i in range(d):
        bi, di, e, s0 = beta[i], beta_dot[i], lam_s_dot[i], s[i]
        if activity[i]:
            if abs(bi * di) > 1e-10 and bi * di > 0:
                L[i] = bi / (di * lambdamax)
        else:
            if abs(e + s0) < 1e-10 or abs(s0) > 1:
                continue
            if e < -s0:
                dl = (1 + s0) / (1 - e)
            else:
                dl = (1 - s0) / (1 + e)
            L[i] = dl * lam

    dlamb = min(min(L), lam, lam - lamin)
    max_up, yADl = False, y * A.dot(D) * lambdamax
    j_switch = None
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

    beta = beta + lambdamax * beta_dot * dlamb
    s = - lam_s_dot + lam / (lam - dlamb) * (s - lam_s_dot)
    r = r + yADl * dlamb
    lam = lam - dlamb

    if max_up:
        F[j_switch] = not F[j_switch]
        M[:d, :][:, :d] = 2 * A[F].T.dot(A[F])
    else:
        # Update matrix inverse, list of rows in C and activity
        for i in range(d):
            if L[i] < dlamb + 1e-10:
                if activity[i]:
                    activity[i], number_act = False, number_act - 1
                    if len(M) > d:
                        to_ad = next_idr2(idr, M[d:, :][:, :d][:, activity])
                        if type(to_ad) == int:
                            idr[to_ad] = False
                else:
                    # x = M[:,activity+idr][i]
                    # al = M[i,i]-np.vdot(x,Xt.dot(x))
                    # if (abs(al)<1e-10): break
                    activity[i], number_act = True, number_act + 1
                    if len(M) > d:
                        to_ad = next_idr1(idr, M[d:, :][:, :d][:, activity])
                        if type(to_ad) == int:
                            idr[to_ad] = True
    try:
        Xt = LA.inv(M[activity + idr, :][:, activity + idr])
    except LA.LinAlgError:
        Xt = LA.inv(
            M[activity + idr, :][:, activity + idr]
            + np.diag([eps_L2] * (sum(activity) + sum(idr)))
        )

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


def derivatives(activity,s,Mat,C,Inv,idr,number_act):
    """
        Compute the derivatives of the solution Beta and the derivative of lambda*subgradient
        thanks to the following equation : 

        2A[active].T A[active] beta_dot + C[active].T v_dot = s       on active variables
        C[active]   beta_dot                               = 0

         (lambda s)_dot  = 2A[inactive].T A[active] beta_dot         on inactive variables

         Then : Mat = A[inactive].T A[active]  and Inv is the inverse of a matrix of the first equaction. 

    """
    beta_dot = np.zeros(len(activity))
    beta_dot[activity]= -Inv[:number_act, :number_act].dot(s[activity])
    lam_s_dot = Mat.dot(beta_dot)
    
    if len(C)>0:
        v_dot = np.zeros(len(C))
        v_dot[idr] = -Inv[number_act:, :number_act].dot(s[activity])
        lam_s_dot += C.T.dot(v_dot)

    return (beta_dot,lam_s_dot)




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


def h_lambdamax(X, y):
    return 2 * LA.norm(X.T.dot(y), np.infty)





def pathalgo(matrix, path, n_active=False):
    """
    This function is only to interpolate the solution path between the breaking points
    """
    BETA, i = [], 0
    X, sp_path = solve_path(matrix, path[-1], n_active)
    sp_path.append(path[-1]), X.append(X[-1])
    for lam in path:
        while lam < sp_path[i + 1]:
            i += 1
        teta = (sp_path[i] - lam) / (sp_path[i] - sp_path[i + 1])
        BETA.append(X[i] * (1 - teta) + X[i + 1] * teta)
    return BETA
