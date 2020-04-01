path = '/Users/lsimpson/Desktop/GitHub/Figures/examplePH/'
from CLasso import *
import numpy as np
from copy import deepcopy as dc

pH = sio.loadmat('data/pHData.mat')
tax = sio.loadmat('data/taxTablepHData.mat')['None'][0]

X,Y_uncent = pH['X'],pH['Y'].T[0]
y = Y_uncent-np.mean(Y_uncent) # Center Y
problem = classo_problem(X,y) # zero sum is default C

# Solve the entire path
problem.model_selection.PATH = True
problem.solve()
problem.solution.PATH.save = path+'R3_'
problem.solution.StabSel.save1 = path+'R3_StabSel'
problem.solution.StabSel.save3 = path+'R3_StabSel-beta'
problem1 = dc(problem)

problem.formulation.huber = True

problem.solve()
problem.solution.PATH.save = path+'R4_'
problem.solution.StabSel.save1 = path+'R4_StabSel'
problem.solution.StabSel.save3 = path+'R4_StabSel-beta'
problem2 = dc(problem)

print(problem1, problem1.solution)
print(problem2, problem2.solution)