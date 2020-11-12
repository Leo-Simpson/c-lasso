from .path_alg import solve_path, pathalgo_general
import numpy as np
import numpy.linalg as LA
from .solve_R1 import problem_R1, Classo_R1

"""
Problem    :   min h_rho(Ab - y) + lambda ||b||1 with C.b= 0 <=>   min ||Ab - y - r*o||^2 + lambda ||b,o||1 with C.b= 0, o in R^m
                                                                                        r = lambda / 2rho
Dimensions :   A : m*d  ;  y : m  ;  b : d   ; C : k*d

The first function compute a solution of a Lasso problem for a given lambda.
The parameters are lam (lambda/lambdamax, \in [0,1]) and pb, which has to be a 'problem_LS type',
which is defined bellow in order to contain all the important parameters of the problem.
One can initialise it this way : pb = class_of_problem.problem(data=(A,C,y),type_of_algo).
We solve the problem without normalizing anything.
"""


def Classo_R2(pb, lam, compute=True):

    pb_type = pb.type  # can be 'Path-Alg', 'P-PDS' , 'PF-PDS' or 'DR'

    (m, d, k), (A, C, y) = pb.dim, pb.matrix
    lamb, rho = lam * pb.lambdamax, pb.rho

    if lam == 0.0:
        pb_type = "DR"
        compute = "True"
        # here we simply refer to Classo_R1 that is called line 42.

    # Path-Alg
    # here we compute the path algo until our lambda, and just take the last beta

    if pb_type == "Path-Alg":
        if pb.intercept:
            AA, CC = A[:, 1:], C[:, 1:]
        else:
            AA, CC = A[:, :], C[:, :]
        out = solve_path((AA, CC, y), lam, False, rho, "R2", intercept=pb.intercept)
        if pb.intercept:
            beta0, beta = out[0][-1], out[1][-1]
            beta = np.array([beta0] + list(beta))
        else:
            beta = out[0][-1]

        return beta

    # DR :
    regpath = pb.regpath
    r = lamb / (2 * rho)
    if pb_type == "DR":
        if compute:
            pb.init_R1(r=r)
            x = Classo_R1(pb.prob_R1, lamb / pb.prob_R1.lambdamax)
            beta = x[:-m]
            if pb.intercept:
                betaO = pb.prob_R1.ybar - np.vdot(pb.prob_R1.Abar, x)
                beta = np.array([betaO] + list(beta))
            return beta
        else:
            pb.add_r(r=r)
            if len(pb.init) == 3:
                pb.prob_R1.init = pb.init
            x, warm_start = Classo_R1(pb.prob_R1, lamb / pb.prob_R1.lambdamax)
            beta = x[:-m]
            if pb.intercept:
                betaO = pb.prob_R1.ybar - np.vdot(pb.prob_R1.Abar, x)
                beta = np.array([betaO] + list(beta))
            return (beta, warm_start)

    tol = pb.tol * LA.norm(y) / LA.norm(A, "fro")  # tolerance rescaled

    # cvx
    # call to the cvx function of minimization
    if pb_type == "cvx":
        import cvxpy as cp

        x = cp.Variable(d)
        objective, constraints = (
            cp.Minimize(cp.sum(cp.huber(A * x - y, rho)) + lamb * cp.norm(x, 1)),
            [C * x == 0],
        )
        prob = cp.Problem(objective, constraints)
        result = prob.solve(warm_start=regpath, eps_abs=tol)
        if regpath:
            return (x.value, True)
        return x.value

    if compute:
        pb.compute_param()
    tau, Proj, AtA, Aty = pb.tauN, proj_c(C, d), pb.AtA, pb.Aty
    gamma = pb.gam / (2 * (pb.AtAnorm + r ** 2))
    t = lamb * gamma
    w, tm, zerom, zerod = (
        t * pb.weights,
        t * np.ones(m),
        np.zeros(m),
        np.zeros(d),
    )
    o, xbar, x, v = pb.init
    # vectors usefull to compute the prox of f(b)= sum(wi |bi|)

    # FORWARD BACKWARD
    if pb_type == "P-PDS":

        for i in range(pb.N):
            grad = AtA.dot(x) - Aty
            v = v + tau * C.dot(xbar)
            S = x - 2 * gamma * grad - 2 * gamma * r * (A.T).dot(o) - (C.T).dot(v)
            o = prox(
                o * (1 - 2 * gamma * r ** 2) + 2 * gamma * r * (y - A.dot(x)),
                tm,
                zerom,
            )
            p = prox(S, w, zerod)
            nw_x = Proj.dot(p)
            eps = nw_x - x
            xbar = p + eps

            if i % 10 == 2 and LA.norm(eps) < tol:  # 0.6
                if regpath:
                    return (x, (o, xbar, x, v))
                else:
                    return x
            x = nw_x
            if LA.norm(x) > 1e10:
                raise ValueError("The algorithm of P-PDS diverges")

        raise ValueError(
            "The algorithm of P-PDS did not converge after %i iterations " % pb.N
        )

    # NO PROJ

    if pb_type == "PF-PDS":  # y1 --> S ; p1 --> p . ; p2 --> y2
        for i in range(pb.N):
            grad = AtA.dot(x) - Aty

            S1 = x - 2 * gamma * grad - 2 * gamma * r * (A.T).dot(o) - (C.T).dot(v)
            S2 = o * (1 - 2 * gamma * r ** 2) + 2 * gamma * r * (y - A.dot(x))

            p1 = prox(S1, w, zerod)
            p2 = prox(S2, tm, zerom)

            v = v + tau * C.dot(p1)
            v2 = v + tau * C.dot(x)

            eps1 = (
                p1 + 2 * gamma * (Aty - AtA.dot(p1) - r * A.T.dot(o)) - C.T.dot(v2) - S1
            )
            eps2 = p2 + 2 * r * gamma * (y - r * p2 - A.dot(x)) - S2

            x = x + eps1
            o = o + eps2

            if i > 0 and LA.norm(eps1) + LA.norm(eps2) < tol:
                if regpath:
                    return (x, (o, xbar, x, v))
                else:
                    return x

            if LA.norm(x) + LA.norm(o) + LA.norm(v) > 1e6:
                raise ValueError("The algorithm of PF-PDS diverges")

        raise ValueError(
            "The algorithm of PF-PDS did not converge after %i iterations " % pb.N
        )


"""
This function compute the the solution for a given path of lam :
by calling the function 'algo' for each lambda with warm start,
or wuth the method ODE, by computing the whole path thanks to the ODE that rules Beta and the subgradient s,
and then to evaluate it in the given finite path.
"""


def pathlasso_R2(pb, path, n_active=False):
    n = pb.dim[0]
    BETA, tol = [], pb.tol
    if pb.type == "Path-Alg":
        (A, C, y) = pb.matrix
        if pb.intercept:
            AA, CC = A[:, 1:], C[:, 1:]
        else:
            AA, CC = A[:, :], C[:, :]
        return pathalgo_general((AA, CC, y), path, "R2", n_active, pb.rho, pb.intercept)

    # Now we are in the case where we have to do warm starts.
    save_init = pb.init
    pb.regpath = True
    pb.compute_param()
    pb.init_R1()
    if type(n_active) == int and n_active > 0:
        n_act = n_active
    else:
        n_act = n
    for lam in path:
        X = Classo_R2(pb, lam, compute=False)
        BETA.append(X[0])
        pb.init = X[1]
        if (
            sum([(abs(X[0][i]) > 1e-5) for i in range(len(X[0]))]) >= n_act
            or type(X[1]) == str
        ):
            pb.init = save_init
            BETA.extend([BETA[-1]] * (len(path) - len(BETA)))
            pb.regpath = False
            return BETA

    pb.init = save_init
    pb.regpath = False
    return BETA


"""
Class of problem : we define a type, which will contain as keys, all the parameters we need for a given problem.
"""


class problem_R2:
    def __init__(self, data, algo, rho, intercept=False):
        self.N = 500000

        (AA, CC, y) = data
        A = AA[:, :]
        C = CC[:, :]
        self.weights = np.ones(A.shape[1])
        self.intercept = intercept
        if intercept:
            # add a column of 1 in A, and change weight.
            A = np.concatenate([np.ones((len(A), 1)), A], axis=1)
            C = np.concatenate([np.zeros((len(C), 1)), C], axis=1)
            self.weights = np.concatenate([[0.0], self.weights])
            yy = y - np.mean(y)

        self.dim = (
            A.shape[0],
            A.shape[1],
            C.shape[0],
        )
        self.matrix = (A, C, y)
        (m, d, k) = self.dim
        self.init = np.zeros(m), np.zeros(d), np.zeros(d), np.zeros(k)
        self.tol = 1e-4
        self.regpath = False
        self.name = algo + " Huber"
        self.type = algo  # type of algo used
        self.rho = rho
        self.gam = 1.0
        self.tau = 0.5  # equation for the convergence of Noproj and LS algorithms : gam + tau < 1
        if not intercept:
            yy = y
        self.lambdamax = 2 * LA.norm(AA.T.dot(h_prime(yy, rho)), np.infty)

    """
    this is a method of the class pb that is used to computed the expensive multiplications only once. (espacially usefull for warm start. )
    """

    def compute_param(self):
        (A, C, y) = self.matrix
        m, d, k = self.dim
        self.c = (
            d / LA.norm(A, 2)
        ) ** 2  # parameter for Concomitant problem : the matrix is scaled as c*A^2

        self.AtA = (A.T).dot(A)
        self.Aty = (A.T).dot(y)
        self.Cnorm = LA.norm(C, 2) ** 2
        self.tauN = self.tau / self.Cnorm
        self.AtAnorm = LA.norm(self.AtA, 2)

    def init_R1(self, r=0.0):
        (AA, CC, y) = self.matrix
        A, C = AA[:, :], CC[:, :]
        (m, d, k) = self.dim
        if self.intercept:
            A, C = A[:, 1:], C[:, 1:]

        Ahuber = np.append(A, r * np.eye(m), 1)
        Chuber = np.append(C, np.zeros((k, m)), 1)
        yhuber = y
        if self.intercept:
            Abar = np.mean(Ahuber, axis=0)
            ybar = np.mean(y)
            Ahuber = Ahuber - Abar
            yhuber = yhuber - ybar
        matrices_huber = (Ahuber, Chuber, yhuber)
        prob = problem_R1(matrices_huber, self.type)
        prob.regpath = self.regpath
        prob.compute_param()
        if self.intercept:
            prob.Abar = Abar
            prob.ybar = ybar
            self.AAt = (A - np.mean(A, axis=0)).dot((A - np.mean(A, axis=0)).T)
        else:
            self.AAt = A.dot(A.T)
        self.prob_R1 = prob

    def add_r(self, r):
        prob = self.prob_R1
        A_r1 = prob.matrix[0]
        m = A_r1.shape[0]
        d = A_r1.shape[1] - m
        prob.AtA[d:, :d] = prob.matrix[0][:, :d] * r
        prob.AtA[:d, d:] = prob.AtA[d:, :d].T
        prob.Aty = np.append(prob.Aty[:d], prob.matrix[2] * r)
        prob.lambdamax = 2 * LA.norm(prob.Aty, np.infty)
        extension = np.eye(m)
        if self.intercept:
            # self.init_R1(r=r)
            extension = extension - np.mean(extension, axis=0)
        extension = r * extension
        A_r1[:, d:] = extension
        right_bottom = extension.dot(extension)
        prob.AtA[d:, d:] = right_bottom
        prob.AAt = self.AAt + right_bottom
        prob.AtAnorm = LA.norm(prob.AtA, 2)


# compute the prox of the function : f(b)= sum (wi * |bi| )
def prox(b, w, zeros):
    return np.minimum(b + w, zeros) + np.maximum(b - w, zeros)


# Compute I - C^t (C.C^t)^-1 . C : the projection on Ker(C)
def proj_c(M, d):
    if LA.matrix_rank(M) == 0:
        return np.eye(d)
    return np.eye(d) - LA.multi_dot([M.T, np.linalg.inv(M.dot(M.T)), M])


# Compute the derivative of the huber function, particulary useful for the computing of lambdamax
def h_prime(y, rho):
    m = len(y)
    lrho = rho * np.ones(m)
    return np.maximum(lrho, -y) + np.minimum(y - lrho, 0)
