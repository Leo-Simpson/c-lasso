from CLasso import *
m,d,d_nonzero,k,sigma =100,100,5,1,0.5
(X,C,y),sol = random_data(m,d,d_nonzero,k,sigma,zerosum=True, seed = 2 )
problem = classo_problem(X,y,C)

problem                                     = classo_problem(X,y,C)
problem.formulation.huber                   = False
problem.formulation.concomitant             = False
problem.model_selection.CV                  = True
problem.model_selection.StabSel              = True
problem.model_selection.LAMfixed            = True
problem.model_selection.PATH                = True
problem.model_selection.StabSelparameters.method = 'max'

problem.solve()
print(problem)
problem.solution.StabSel.save1 = 'example2/StabSel'
problem.solution.StabSel.save3 = 'example2/StabSel-beta'
problem.solution.CV.save = 'example2/CV-beta'
problem.solution.LAMfixed.save = 'example2/LAM-beta'
print(problem.solution)

problem.solution.CV.graphic(mse_max = 1.,save='example2/CV-graph')