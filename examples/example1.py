path = '/Users/lsimpson/Desktop/GitHub/Figures/example1/'
from CLasso import *

m,d,d_nonzero,k,sigma =100,100,5,1,0.5
(X,C,y),sol = random_data(m,d,d_nonzero,k,sigma,zerosum=True,seed= 4)
problem = classo_problem(X,y,C)
problem.solve()
print(problem)
problem.solution.StabSel.save1 = path + 'StabSel'
problem.solution.StabSel.save2 = path + 'StabSel-path'
problem.solution.StabSel.save3 = path + 'StabSel-beta'
print(problem.solution)


