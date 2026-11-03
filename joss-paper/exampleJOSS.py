import sys, os
from os.path import join, dirname, abspath
figure_dir = join(dirname(abspath(__file__)), "figures")


import numpy
from classo import classo_problem, random_data

n, d, d_nonzero, k, sigma = 100, 100, 5, 1, 0.5
(X, C, y), sol = random_data(n, d, d_nonzero, k, sigma, zerosum=True, seed=123)
print("Relevant variables  : {}".format(numpy.nonzero(sol)[0]))

problem = classo_problem(X, y, C)

problem.formulation.huber = True
problem.formulation.concomitant = False
problem.formulation.rho = 1.5

problem.model_selection.LAMfixed = True
problem.model_selection.PATH = True
problem.model_selection.LAMfixedparameters.rescaled_lam = True
problem.model_selection.LAMfixedparameters.lam = 0.1

problem.solve()

name_method = problem.model_selection.LAMfixedparameters.numerical_method
problem.solution.StabSel.save1 = f"{figure_dir}/{name_method}-StabSel"
problem.solution.StabSel.save2 = f"{figure_dir}/{name_method}-StabSel-path"
problem.solution.StabSel.save3 = f"{figure_dir}/{name_method}-StabSel-beta"
# problem.solution.CV.save = path+'CV-beta
problem.solution.LAMfixed.save = f"{figure_dir}/{problem.model_selection.PATHparameters.numerical_method}-LAM-beta"
problem.solution.PATH.save = f"{figure_dir}/{problem.model_selection.LAMfixedparameters.numerical_method}-PATH"

print(problem.solution)
