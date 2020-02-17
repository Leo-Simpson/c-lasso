from CLasso import *
import numpy as np
from copy import deepcopy as dc
from time import time
pH = sio.loadmat('Data/pHData.mat')
tax = sio.loadmat('Data/taxTablepHData.mat')['None'][0]

X,Y_uncent = pH['X'],pH['Y'].T[0]
y = Y_uncent-np.mean(Y_uncent) # Center Y
problem = classo_problem(X,y) # zero sum is default C

#not doing stability selection
problem.model_selection.StabSel = False
# Solve the entire path
problem.model_selection.PATH = True
problem.model_selection.PATHparameters.plot_sigma = True
problem.solve()
problem.solution.PATH.save = 'figures_examplePH/Concomitant '
problem1 = dc(problem)
problem.formulation.huber = True
problem.solve()
problem.solution.save = 'figures_examplePH/Concomitant Huber '
problem2 = dc(problem)

print(problem1, problem1.solution)
print(problem2, problem2.solution)