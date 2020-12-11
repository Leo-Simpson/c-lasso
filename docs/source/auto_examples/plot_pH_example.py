r"""
pH prediction using the Central Park soil dataset 
===================================================

The next microbiome example considers the 
`Central Park Soil dataset <https://github.com/Leo-Simpson/c-lasso/tree/master/examples/pH_data>`_ from
`Ramirez et al. <https://royalsocietypublishing.org/doi/full/10.1098/rspb.2014.1988>`_. The sample locations are shown in the Figure on the right.
The task is to predict pH concentration in the soil from microbial abundance data. This task was also considered in `Tree-Aggregated Predictive Modeling of Microbiome Data <https://www.biorxiv.org/content/10.1101/2020.09.01.277632v1>`_.
"""

from classo import classo_problem
import numpy as np
from copy import deepcopy as dc
import scipy.io as sio

# %%
#  Load data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

pH = sio.loadmat("pH_data/pHData.mat")
tax = sio.loadmat("pH_data/taxTablepHData.mat")["None"][0]

X, Y_uncent = pH["X"], pH["Y"].T[0]
y = Y_uncent - np.mean(Y_uncent)  # Center Y
print(X.shape)

# %%
# Set up c-lassso problem
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^

problem = classo_problem(X, y) 

problem.model_selection.StabSelparameters.method      = 'lam'
problem.model_selection.PATH = True
problem.model_selection.LAMfixed = True
problem.model_selection.PATHparameters.n_active = X.shape[1] + 1

# %%
# Solve for R1
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
problem.formulation.concomitant = False
problem.solve()
print(problem, problem.solution)

# %%
# Solve for R2
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
problem.formulation.huber = True
problem.solve()
print(problem, problem.solution)


# %%
# Solve for R3
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
problem.formulation.concomitant = True
problem.formulation.huber = False
problem.solve()
print(problem, problem.solution)


# %%
# Solve for R4
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Remark : we reset the numerical method here, 
# because it has been automatically set to '¨Path-Alg'
# for previous computations, but for R4, "DR" is much better
# as explained in the documentation, R4 "Path-Alg" is a method for fixed lambda
# but is (paradoxically) bad to compute the lambda-path 
# because of the absence of possible warm-start in this method

problem.model_selection.PATHparameters.numerical_method = "DR"
problem.formulation.huber = True
problem.solve()
print(problem, problem.solution)


