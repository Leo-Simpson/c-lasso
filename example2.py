from CLasso import *
m,d,d_nonzero,k,sigma =100,100,5,1,0.5
(X,C,y),sol = random_data(m,d,d_nonzero,k,sigma,zerosum=True, seed = 2 )
problem = classo_problem(X,y,C)

problem                                     = classo_problem(X,y,C)
problem.formulation.huber                   = True
problem.formulation.concomitant             = False
problem.model_selection.CV                  = True
problem.model_selection.SS                  = True
problem.model_selection.LAMfixed            = True
problem.model_selection.PATH                = False
problem.model_selection.SSparameters.method = 'max'

problem.solve()
print(problem)
print(problem.solution)

problem.solution.CV.graphic(mse_max = 1.)