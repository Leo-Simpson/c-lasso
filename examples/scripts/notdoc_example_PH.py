import sys, os
from os.path import join, dirname, abspath
from pathlib import Path
if "__file__" in globals():
    this_dir = Path(__file__).resolve().parent
else:
    this_dir = Path.cwd()

# this file is in classo/examples/scripts, so we need to go up three levels to get the classo directory
classo_dir = dirname(dirname(this_dir))
sys.path.append(classo_dir)
from classo import classo_problem
import numpy as np

figure_dir = join(classo_dir, "figures/examplePH")


from classo import classo_problem
import numpy as np
from copy import deepcopy as dc
import scipy.io as sio

data_dir = join(classo_dir, "examples", "datasets", "pH_data/matlab")

pH = sio.loadmat(join(data_dir, "pHData.mat"))
# tax = sio.loadmat(join(data_dir, "taxTablepHData.mat"))["None"][0]

X, Y_uncent = pH["X"], pH["Y"].T[0]
y = Y_uncent - np.mean(Y_uncent)  # Center Y
problem = classo_problem(X, y)  # zero sum is default C

# Solve the entire path
problem.model_selection.PATH = True
problem.solve()
problem.solution.PATH.save = join(figure_dir, "R3-")
problem.solution.StabSel.save1 = join(figure_dir, "R3-StabSel")
problem.solution.StabSel.save3 = join(figure_dir, "R3-StabSel-beta")
problem1 = dc(problem)

# problem.formulation.huber = True

problem.solve()
problem.solution.PATH.save = join(figure_dir, "R4-")
problem.solution.StabSel.save1 = join(figure_dir, "R4-StabSel")
problem.solution.StabSel.save3 = join(figure_dir, "R4-StabSel-beta")
problem2 = dc(problem)

print(problem1, problem1.solution)
problem1.solution.plot()
print(problem2, problem2.solution)
problem2.solution.plot()
