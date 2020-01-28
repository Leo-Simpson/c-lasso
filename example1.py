from CLasso import *
import numpy as np
m,d,d_nonzero,k,sigma =100,50,5,1,0.5
(X,C,y),sol = random_data(m,d,d_nonzero,k,sigma,zerosum=True)
problem = classo_problem(X,y,C)
problem.model_selection.SS = False
problem.model_selection.PATH = False
problem.model_selection.LAMfixed = True


problem.solve()
sigma = problem.solution.LAMfixed.sigma
print(problem)
print(problem.solution)
problem.model_selection.LAMfixedparameters.numerical_method = '2prox'
problem.solve()
sigma2 = problem.solution.LAMfixed.sigma
print(problem)
print(problem.solution)