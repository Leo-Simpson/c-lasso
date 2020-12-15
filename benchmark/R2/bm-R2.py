import numpy as np
import numpy.linalg as LA
from classo import classo_problem, random_data
import cvxpy as cp
from time import time

l = [1, 2, 5, 7]

def huber(r, rho):
    F = abs(r) >= rho
    h = r**2
    h[F] =  2*rho*abs(r)[F] - rho**2
    return t
    
    

def loss(X, y, lam, rho, beta):
    lamb = lam*2*LA.norm(X.T.dot(y),np.infty)
    r = X.dot(beta) - y
    return np.sum(huber(r,rho)) + lamb*np.sum(abs(beta))


d_nonzero = 5
sigma = 0.5
lam = 0.1

N_per_data = 5
N_data = 10
SIZES = [
    (50, 100),
    (100, 100),
    (100, 200),
    (200, 200)
]


N_sizes = len(SIZES)

T_pa = np.zeros((N_sizes, N_data))
L_pa = np.zeros((N_sizes, N_data))

T_pds = np.zeros((N_sizes, N_data))
L_pds = np.zeros((N_sizes, N_data))

T_dr = np.zeros((N_sizes, N_data))
L_dr = np.zeros((N_sizes, N_data))

T_cvx = np.zeros((N_sizes, N_data))
L_cvx = np.zeros((N_sizes, N_data))



for s in range(N_sizes):

    m, d = SIZES[s]


    for i in range(N_data):
        (X, C, y), sol = random_data(m, d, d_nonzero, 1, sigma, zerosum=True, seed=i)
        rho = 1.345 * np.sqrt(np.mean(y**2))
        lamb = lam*2*LA.norm(X.T.dot(y),np.infty)

        t0 = time()
        # classo Path-Alg
        b_pa = []
        for j in range(N_per_data):
            problem = classo_problem(X, y, C)
            problem.formulation.concomitant = False
            problem.formulation.huber = True
            problem.formulation.scale_rho = False
            problem.formulation.rho = rho
            problem.model_selection.StabSel = False
            problem.model_selection.LAMfixed = True
            problem.model_selection.LAMfixedparameters.rescaled_lam = True
            problem.model_selection.LAMfixedparameters.lam = lam
            problem.model_selection.LAMfixedparameters.numerical_method = 'Path-Alg'
            problem.solve()
            b_pa.append(problem.solution.LAMfixed.beta)
        b_pa = np.array(b_pa)

        t1 = time()
        # classo P-PDS
        b_pds = []
        for j in range(N_per_data):
            problem = classo_problem(X, y, C)
            problem.formulation.concomitant = False
            problem.formulation.huber = True
            problem.formulation.scale_rho = False
            problem.formulation.rho = rho
            problem.model_selection.StabSel = False
            problem.model_selection.LAMfixed = True
            problem.model_selection.LAMfixedparameters.rescaled_lam = True
            problem.model_selection.LAMfixedparameters.lam = lam
            problem.model_selection.LAMfixedparameters.numerical_method = 'P-PDS'
            problem.solve()
            b_pds.append(problem.solution.LAMfixed.beta)
        b_pds = np.array(b_pds)

        t2 = time()
        # classo DR
        b_dr = []
        for j in range(N_per_data):
            problem = classo_problem(X, y, C)
            problem.formulation.concomitant = False
            problem.formulation.huber = True
            problem.formulation.scale_rho = False
            problem.formulation.rho = rho
            problem.model_selection.StabSel = False
            problem.model_selection.LAMfixed = True
            problem.model_selection.LAMfixedparameters.rescaled_lam = True
            problem.model_selection.LAMfixedparameters.lam = lam
            problem.model_selection.LAMfixedparameters.numerical_method = 'P-PDS'
            problem.solve()
            b_dr.append(problem.solution.LAMfixed.beta)
        b_dr = np.array(b_dr)

        t3 = time()
        # cvx
        b_cvx = []
        for j in range(N_per_data):
            beta = cp.Variable(d)
            objective, constraints = cp.Minimize(cp.sum_squares(cp.huber(X@beta-y), rho)+ lamb*cp.norm(beta, 1)), [C@beta == 0]
            prob = cp.Problem(objective, constraints)
            result = prob.solve(warm_start = False,eps_abs = 1e-5)
            b_cvx.append(beta.value)
        b_cvx = np.array(b_cvx)
        
        t4 = time()


        T_pa[s, i] = (t1 - t0) / N_per_data
        L_pa[s, i] = loss(X, y, lam, rho, np.mean(b_pa, axis=0))
        C_pa[s, i] = np.linalg.norm(C.dot(np.mean(b_pa, axis=0)))

        T_pds[s, i] = (t2 - t1) / N_per_data
        L_pds[s, i] = loss(X, y, lam, rho, np.mean(b_pds, axis=0))
        C_pds[s, i] = np.linalg.norm(C.dot(np.mean(b_pds, axis=0)))

        T_dr[s, i] = (t3 - t0) / N_per_data
        L_dr[s, i] = loss(X, y, lam, rho, np.mean(b_dr, axis=0))
        C_dr[s, i] = np.linalg.norm(C.dot(np.mean(b_ds, axis=0)))

        T_cvx[s, i] = (t4 - t3) / N_per_data  
        L_cvx[s, i] = loss(X, y, lam, rho, np.mean(b_cvx, axis=0))
        C_cvx[s, i] = np.linalg.norm(C.dot(np.mean(b_cvx, axis=0)))

np.savez(
    'bm-R1.npz',
    T_pa = T_pa,
    L_pa = L_pa,
    C_pa = C_pa, 
    T_pds = T_pds,
    L_pds = L_pds,
    C_pds = C_pds,
    T_dr = T_dr,
    L_dr = L_dr,
    C_dr = C_dr,
    T_cvx = T_cvx,
    L_cvx = L_cvx,
    C_cvx = C_cvx,
    SIZES = np.array(SIZES)
)
