path = '/Users/lsimpson/Desktop/GitHub/c-lasso/paper/figures/'
from classo import *
m,d,d_nonzero,k,sigma =100,100,5,1,0.5
(X,C,y),sol = random_data(m,d,d_nonzero,k,sigma,zerosum=True, seed = 123 )

import numpy
print( list(numpy.nonzero(sol)[0]) )



# let's define a c-lasso problem instance with the previously generated data and parameters set to default values
problem  = classo_problem(X,y,C)
problem.numerical_method = 'choose'

# let's change the optimization formulation of problem instance
problem.formulation.huber  = True
problem.formulation.concomitant = False
problem.formulation.rho = 1.5

# let's add a computation of the lambda-path
problem.model_selection.PATH = True

# let's then add a computation of beta for a fixed lambda 
problem.model_selection.LAMfixed = True
# and set it to to 0.1*lambdamax
problem.model_selection.LAMfixedparameters.rescaled_lam = True
problem.model_selection.LAMfixedparameters.lam = 0.1



# Finally one can compute the solutions of those optimization problems
problem.solve()

print(" \n Here is the problem instance plot : \n ")
print(problem)


problem.solution.StabSel.save1 = path+problem.model_selection.StabSelparameters.numerical_method+'-StabSel'
problem.solution.StabSel.save2 = path+problem.model_selection.StabSelparameters.numerical_method+'-StabSel-path'
problem.solution.StabSel.save3 = path+problem.model_selection.StabSelparameters.numerical_method+'-StabSel-beta'
#problem.solution.CV.save = path+'CV-beta'
problem.solution.LAMfixed.save = path+problem.model_selection.PATHparameters.numerical_method+'-LAM-beta'
problem.solution.PATH.save = path+problem.model_selection.LAMfixedparameters.numerical_method+'-PATH'


print(" \n Here is the solution instance plot : \n ")
print(problem.solution)


#problem.solution.CV.graphic(mse_max = 1.,save=path+'CV-graph')


