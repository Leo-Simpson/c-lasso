path = '/Users/lsimpson/Documents/GitHub/Figures/examplePH/'
from CLasso import *
import numpy as np
from copy import deepcopy as dc
from time import time
pH = sio.loadmat('data/pHData.mat')
tax = sio.loadmat('data/taxTablepHData.mat')['None'][0]

X,Y_uncent = pH['X'],pH['Y'].T[0]
y = Y_uncent-np.mean(Y_uncent) # Center Y
problem = classo_problem(X,y) # zero sum is default C

#not doing stability selection
problem.model_selection.StabSel = False
# Solve the entire path
problem.model_selection.PATH = True
problem.model_selection.PATHparameters.plot_sigma = True
problem.solve()
problem.solution.PATH.save = path+'Concomitant '
problem1 = dc(problem)
problem.formulation.huber = True
problem.solve()
problem.solution.save = path+'Concomitant Huber '
problem2 = dc(problem)

print(problem1, problem1.solution)
print(problem2, problem2.solution)