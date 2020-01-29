from CLasso import *
import numpy as np
m,d,d_nonzero,k,sigma =100,90,5,1,0.5
(X,C,y),sol = random_data(m,d,d_nonzero,k,sigma,zerosum=True,seed=4)
problem = classo_problem(X,y,C)
problem.model_selection.SSparameters.method = 'lam'
problem.formulation.huber=True


problem.model_selection.LAMfixedparameters.numerical_method = 'ODE'
problem.solve()
print(problem)
print(problem.solution)
