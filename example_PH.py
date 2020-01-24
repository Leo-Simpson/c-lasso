from CLasso import *
import numpy as np

PHData  = sio.loadmat('data/pHData.mat')

X,Y = PHData['X'],PHData['Y'][:,0]

y = Y - np.mean(Y)
C = np.ones((1,len(X[0])))



problem = classo_problem(X,y,C)

problem.model_selection.LAMfixed = True
problem.model_selection.SS       = True
problem.model_selection.SSparameters.method = 'lam'
problem.model_selection.PATH = True
problem.formulation.huber = False
problem.data.rescale = False

problem.solve()
print(problem)
print(problem.solution)

