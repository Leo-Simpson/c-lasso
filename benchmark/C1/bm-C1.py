import numpy as np
import numpy.linalg as LA
from classo import classo_problem, random_data
import cvxpy as cp
from time import time

import os
my_path = os.path.dirname(__file__) 

l = [1, 2, 5, 7]



def loss(X, y, lamb, beta):
    return np.sum(np.maximum(1 - y*X.dot(beta), 0)**2) + lamb*np.sum(abs(beta))


d_nonzero = 5
sigma = 0.5
lam = 0.1

N_per_data = 5
N_data = 20
S = [50, 100, 200, 500]


SIZES = []
for i in range(len(S)):
    SIZES.append((S[i], S[i]))
    if i+1<len(S):
        SIZES.append((S[i], S[i+1]))

N_sizes = len(SIZES)

T_pa = np.zeros((N_sizes, N_data))
L_pa = np.zeros((N_sizes, N_data))
C_pa = np.zeros((N_sizes, N_data))

T_cvx = np.zeros((N_sizes, N_data))
L_cvx = np.zeros((N_sizes, N_data))
C_cvx = np.zeros((N_sizes, N_data))



for s in range(N_sizes):

    m, d = SIZES[s]


    for i in range(N_data):
        (X, C, y), sol = random_data(m, d, d_nonzero, 1, sigma, zerosum=True, seed=i)
        y = np.sign(y)
        lamb = lam*2*LA.norm(X.T.dot(y),np.infty)

        t0 = time()
        # classo Path-Alg
        b_pa = []
        for j in range(N_per_data):
            problem = classo_problem(X, y, C)
            problem.formulation.concomitant = False
            problem.formulation.classification =True
            problem.model_selection.StabSel = False
            problem.model_selection.LAMfixed = True
            problem.model_selection.LAMfixedparameters.rescaled_lam = False
            problem.model_selection.LAMfixedparameters.lam = lamb
            problem.model_selection.LAMfixedparameters.numerical_method = 'Path-Alg'
            problem.solve()
            b_pa.append(problem.solution.LAMfixed.beta)
        b_pa = np.array(b_pa)

        t1 = time()

        # cvx
        b_cvx = []
        for j in range(N_per_data):
            beta = cp.Variable(d)
            cp.pos(1 - cp.multiply(y, X @ beta))
            objective, constraints = cp.Minimize(cp.sum_squares(cp.pos(1 - cp.multiply(y, X @ beta)))+ lamb*cp.norm(beta, 1)), [C@beta == 0]
            prob = cp.Problem(objective, constraints)
            result = prob.solve(warm_start = False, eps_abs = 1e-5)
            b_cvx.append(beta.value)
        b_cvx = np.array(b_cvx)
        
        t2 = time()


        T_pa[s, i] = (t1 - t0) / N_per_data
        L_pa[s, i] = loss(X, y, lamb, np.mean(b_pa, axis=0))
        C_pa[s, i] = np.linalg.norm(C.dot(np.mean(b_pa, axis=0)))

        T_cvx[s, i] = (t2 - t1) / N_per_data  
        L_cvx[s, i] = loss(X, y, lamb, np.mean(b_cvx, axis=0))
        C_cvx[s, i] = np.linalg.norm(C.dot(np.mean(b_cvx, axis=0)))

np.savez(
    os.path.join(my_path, 'bm-C1.npz'),
    T_pa = T_pa,
    L_pa = L_pa,
    C_pa = C_pa, 
    T_cvx = T_cvx,
    L_cvx = L_cvx,
    C_cvx = C_cvx,
    SIZES = np.array(SIZES)
)
