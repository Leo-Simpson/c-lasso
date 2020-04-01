path = '/Users/lsimpson/Desktop/GitHub/Figures/example2/'
from CLasso import *
m,d,d_nonzero,k,sigma =100,100,5,1,0.5
(X,C,y),sol = random_data(m,d,d_nonzero,k,sigma,zerosum=True, seed = 4 )

problem                                     = classo_problem(X,y,C)
problem.formulation.huber                   = False
problem.formulation.concomitant             = True
problem.model_selection.CV                  = True
problem.model_selection.LAMfixed            = True
problem.model_selection.PATH                = True
problem.model_selection.StabSelparameters.method = 'max'

problem.solve()
print(problem)

problem.solution.StabSel.save1 = path+'StabSel'
problem.solution.StabSel.save3 = path+'StabSel-beta'
problem.solution.CV.save = path+'CV-beta'
problem.solution.LAMfixed.save = path+'LAM-beta'
problem.solution.PATH.save = path+'PATH'

print(problem.solution)

problem.solution.CV.graphic(mse_max = 1.,save=path+'CV-graph')