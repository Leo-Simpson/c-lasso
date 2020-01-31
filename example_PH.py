from CLasso import *
import numpy as np
pH = sio.loadmat('Data/pHData.mat')
tax = sio.loadmat('Data/taxTablepHData.mat')['None'][0]
X,Y_uncent, header = pH['X'],pH['Y'].T[0] , pH['__header__']
y = Y_uncent-np.mean(Y_uncent) # Center Y
problem = classo_problem(X,y) # zero sum is default C

problem.model_selection.SSparameters.seed = 4
# Solve the problem for a fixed lambda (by default, it will use the theoritical lambda)
problem.model_selection.LAMfixed                    = True
# Solve the stability selection : (by default, it will use the theoritical lambda)
problem.model_selection.SS                       = True
problem.model_selection.SSparameters.method      = 'lam'
problem.model_selection.SSparameters.threshold   = 0.7
# Solve the entire path
problem.model_selection.PATH = True
problem.model_selection.PATHparameters.plot_sigma = True



problem.solve()
print(problem)
print(problem.solution)
