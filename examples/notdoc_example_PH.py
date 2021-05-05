path = "../../figures/examplePH/"
import sys, os
from os.path import join, dirname
classo_dir = dirname(dirname(__file__))
sys.path.append(classo_dir)

from classo import classo_problem
import numpy as np
from copy import deepcopy as dc
import scipy.io as sio

pH = sio.loadmat("pH_data/matlab/pHData.mat")
tax = sio.loadmat("pH_data/matlab/taxTablepHData.mat")["None"][0]

X, Y_uncent = pH["X"], pH["Y"].T[0]
y = Y_uncent - np.mean(Y_uncent)  # Center Y
problem = classo_problem(X, y)  # zero sum is default C

# Solve the entire path
problem.model_selection.PATH = True
problem.solve()
problem.solution.PATH.save = path + "R3-"
problem.solution.StabSel.save1 = path + "R3-StabSel"
problem.solution.StabSel.save3 = path + "R3-StabSel-beta"
problem1 = dc(problem)

# problem.formulation.huber = True

problem.solve()
problem.solution.PATH.save = path + "R4-"
problem.solution.StabSel.save1 = path + "R4-StabSel"
problem.solution.StabSel.save3 = path + "R4-StabSel-beta"
problem2 = dc(problem)

print(problem1, problem1.solution)
print(problem2, problem2.solution)
