from .path_alg import solve_path_Conc
import numpy as np
import numpy.linalg as LA
from .misc_functions import unpenalized

"""
Problem    :   min ||Ab - y||^2/sigma + n/2 sigma + lambda ||b||1 with C.b= 0 and sigma > 0

Dimensions :   A : m*d  ;  y : m  ;  b : d   ; C : k*d

The first function compute a solution of a Lasso problem for a given lambda.
The parameters are lam (lambda/lambdamax, \in [0,1]) and pb, which has to be a 'problem_LS type',
which is defined bellow in order to contain all the important parameters of the problem.
One can initialise it this way : pb = class_of_problem.problem(data=(A,C,y),type_of_algo).
We solve the problem without normalizing anything.
"""


def Classo_R3(pb, lam):
    pb_type = pb.type  # can be 'Path-Alg' or 'DR'
    (m, d, k), (A, C, y) = pb.dim, pb.matrix
    sigmax = pb.sigmax

    if lam == 0.0:
        beta = unpenalized(pb.matrix)
        sigma = LA.norm(A.dot(beta) - y) / np.sqrt(m / 2)
        return beta, sigma

    # Path algorithm
    # here we compute the path algo until our lambda, and just take the last beta

    # here we use our path algorithm for concomitant problem, and then only takes the last beta.
    # Actually, the function solve_path_Conc has the argument concomitant= 'fix_lam' so it means it will directly stop when it has to.
    # Then we only have to finc the solution between the last beta computed and the one before.
    if pb_type == "Path-Alg":
        (beta1, beta2), (s1, s2), (r1, r2) = solve_path_Conc(
            (A, C, y), lam, lassopath=False
        )
        dr, ds = r1 - r2, s1 - s2
        teta = root_2(
            LA.norm(dr) ** 2 - ds ** 2,
            np.vdot(dr, r2) - s2 * ds,
            LA.norm(r2) ** 2 - s2 ** 2,
        )
        sigma = (s1 * teta + s2 * (1 - teta)) * sigmax
        beta = beta1 * teta + beta2 * (1 - teta)
        return (beta, sigma)

    regpath = pb.regpath
    if not regpath:
        pb.compute_param()

    lamb = lam * pb.lambdamax
    Anorm = pb.Anorm
    tol = pb.tol * LA.norm(y) / Anorm  # tolerance rescaled
    Proj = proj_c(C, d)  # Proj = I - C^t . (C . C^t )^-1 . C
    QA = pb.QA
    Q1 = pb.Q1
    Q2 = pb.Q2
    # Save some matrix products already computed in problem.compute_param()
    gamma = pb.gam / (pb.Anorm2 * lam)  # Normalize gamma
    w = lamb * gamma * pb.weights
    zerod = np.zeros(d)
    # two vectors usefull to compute the prox of f(b)= sum(wi |bi|)
    mu, c, root = pb.mu, pb.c, 0.0
    xs, nu, o, xbar, x = pb.init

    # 2prox
    if pb_type == "DR":
        for i in range(pb.N):
            nv_b = x + Q1.dot(o) - QA.dot(x) - Q2.dot(x - xbar)
            nv_s = (xs + nu) / 2
            if i > 0 and LA.norm(b - nv_b) + LA.norm(s - nv_s) / Anorm < 2 * tol:
                if regpath:
                    return (b, (xs, nu, o, xbar, x), s)
                else:
                    return (b, s)

            s, b = nv_s, nv_b
            Ab = A.dot(b)
            p1, p2, root = prox_phi_1(xs, 2 * Ab - o - y, gamma / c, root)
            sup1 = max(0, nu) - s
            sup2 = p1 - s
            sup3 = p2 + y - Ab
            sup4 = prox(2 * b - xbar, w, zerod) - b
            sup5 = Proj.dot(2 * b - x) - b

            xs = xs + mu * sup1
            nu = nu + mu * sup2
            o = o + mu * sup3
            xbar = xbar + mu * sup4
            x = x + mu * sup5

            if LA.norm(b) + LA.norm(s) > 1e6:
                raise ValueError("The algorithm of Doulgas Rachford diverges")

        raise ValueError(
            "The algorithm of Doulgas Rachford did not converge after %i iterations "
            % pb.N
        )
    print("none of the cases ! ")


"""
This function compute the the solution for a given path of lam :
by calling the function 'algo' for each lambda with warm start,
or wuth the method ODE, by computing the whole path thanks to the ODE that rules Beta and the subgradient s,
and then to evaluate it in the given finite path.
"""


def pathlasso_R3(pb, path, n_active=False):
    n, d, k = pb.dim
    BETA, SIGMA, tol = [], [], pb.tol

    if pb.type == "Path-Alg":
        y = pb.matrix[2]
        sigmax = LA.norm(y)
        X, LAM, R = solve_path_Conc(pb.matrix, path[-1], n_active=n_active)
        LAM.append(path[-1]), X.append(X[-1]), R.append(R[-1])
        beta2, l2, r2, j = X[0], path[0] + 0.1, -y / LA.norm(y), 0
        for lam in path:
            if lam == 0:
                lam = 1e-4
            while (LA.norm(r2) < l2 / lam) and (j < len(LAM)):
                beta1, l1, r1, beta2, l2, r2, j = (
                    beta2,
                    l2,
                    r2,
                    X[j],
                    LAM[j],
                    R[j],
                    j + 1,
                )
            s1, s2 = l1 / lam, l2 / lam
            dr, ds = r1 - r2, s1 - s2
            teta = root_2(
                LA.norm(dr) ** 2 - ds ** 2,
                np.vdot(dr, r2) - s2 * ds,
                LA.norm(r2) ** 2 - s2 ** 2,
            )
            SIGMA.append((s1 * teta + s2 * (1 - teta)) * sigmax)
            BETA.append(beta1 * teta + beta2 * (1 - teta))
        return (BETA, SIGMA)

    save_init = pb.init
    pb.regpath = True
    pb.compute_param()
    if type(n_active) == int and n_active > 0:
        n_act = n_active
    else:
        n_act = n

    for lam in path:
        X = Classo_R3(pb, lam)
        BETA.append(X[0])
        SIGMA.append(X[-1])
        pb.init = X[1]

        if (
            sum([(abs(X[0][i]) > 1e-5) for i in range(len(X[0]))]) >= n_act
            or type(X[1]) == str
        ):
            pb.init = save_init
            BETA.extend([BETA[-1]] * (len(path) - len(BETA)))
            SIGMA.extend([SIGMA[-1]] * (len(path) - len(SIGMA)))
            pb.regpath = False
            return (BETA, SIGMA)

    pb.init = save_init
    pb.regpath = False
    return (BETA, SIGMA)


"""
Class of problem : we define a type, which will contain as keys, all the parameters we need for a given problem.
"""


class problem_R3:
    def __init__(self, data, algo):
        self.N = 500000

        (A, C, y) = data
        self.dim = (A.shape[0], A.shape[1], C.shape[0])
        self.matrix = (A, C, y)

        (m, d, k) = self.dim
        self.weights = np.ones(d)
        self.tol = 1e-6

        self.regpath = False
        self.name = algo + " Concomitant"
        self.type = algo  # type of algorithm used

        self.proj_sigm = lambda x: max(x, 0)
        self.mu = 1.95

        self.gam = np.sqrt(d)
        self.Aty = (A.T).dot(y)
        self.sigmax = LA.norm(y) / np.sqrt(m / 2)
        self.lambdamax = 2 * LA.norm(self.Aty, np.infty) / self.sigmax
        self.init = 1.0, 1.0, np.zeros(m), np.zeros(d), np.zeros(d)

    # Here we compute the costful matrices products and inverts in order to compute it only once, which is especially helpful for warmstarts.
    def compute_param(self):
        (A, C, y) = self.matrix
        m, d, k = self.dim
        self.Anorm = LA.norm(A, "fro")
        self.Anorm2 = self.Anorm ** 2
        c = (d / LA.norm(A, 2)) ** 2
        # parameter for Concomitant problem : the matrix is scaled as c*A^2
        self.c = c
        self.Q1, self.Q2 = QQ(c, A)
        self.QA = self.Q1.dot(A)


"""
Functions used in the algorithms, modules needed :
"""


# compute the prox of the function : f(b)= sum (wi * |bi| )
def prox(b, w, zeros):
    return np.minimum(b + w, zeros) + np.maximum(b - w, zeros)


# Compute I - C^t (C.C^t)^-1 . C : the projection on Ker(C)
def proj_c(M, d):
    if LA.matrix_rank(M) == 0:
        return np.eye(d)
    return np.eye(d) - LA.multi_dot([M.T, np.linalg.inv(M.dot(M.T)), M])


# Compute the real positive root of a polynomial of degree 3 in the form :
#  X^3 + a*X - b with Newton method and a warm start (for Comcomitant problem)
def calc_Newton(a, b, root):
    er = -b
    while abs(er) > 1e-6:
        root = root - er / (3 * root ** 2 + a)
        er = root ** 3 + a * root - b
    return root


def QQ(coef, A):
    # compute QQ = coef A^t (2.I.+coef A A^t )^-1 , (2.I.+coef A^t A)^-1
    return (
        coef * (A.T).dot(LA.inv(2 * np.eye(A.shape[0]) + coef * A.dot(A.T))),
        LA.inv(2 * np.eye(A.shape[1]) + coef * (A.T).dot(A)),
    )


# Return the cost function of some Beta for a given Lasso problem : L_LS = ||y-Ab||2^2  + lambda* ||b||1
def L_LS(pb, lam, sol):
    return LA.norm(
        pb.matrix[0].dot(sol) - pb.matrix[2]
    ) ** 2 + lam * pb.lambdamax * LA.norm(sol, 1)


# Return the prox operator for the function
# phi = ||(y-Ab)||2^2 /sigma + 1/2 * sigma with the warm start on the computation of the root
def prox_phi_1(sig, u, gamma, warm_start):
    l2u = LA.norm(u)
    Po0 = 4 * gamma * sig + l2u ** 2
    if Po0 < 2 * gamma ** 2:
        return (0.0, 0.0, 0.0)
    else:
        root = calc_Newton((4 * sig / gamma + 6.0), 8 * l2u / gamma, warm_start)
        return (
            sig + 0.5 * gamma * (0.5 * root ** 2 - 1.0),
            u * (1 - gamma * root / l2u),
            root,
        )


# Return the positive root in [0,1] of the polynomial aX^2 + 2bX + c if it exists, 1. if not
def root_2(a, b, c):

    if a == 0.0:
        return c / (2 * b)
    root = (-np.sqrt(b ** 2 - a * c) - b) / a
    if root > 1:
        return 1.0
    return root


# The aim :
# solve the equation :
# |dr|^2 teta^2 + 2 (dr.r2) teta  + |r2|^2 =
# |ds|^2 teta^2 + 2 (ds.s2) teta  + |s2|^2

# Which comes from the equation :
# | teta r1 + (1-teta) r2 | = | teta s1 + (1-teta) s2 |

#  Which comes from the equation :
# lam * | teta r1 + (1-teta) r2 | = | teta l1 + (1-teta) l2 |

# lam * | y - X.( teta beta1 + (1-teta)beta2 ) | = | teta l1 + (1-teta) l2 |

# in other word : $lam.|y-X.beta(gamma)| = gamma $

# so find a gamma such that sigma(gamma) * lam = gamma
